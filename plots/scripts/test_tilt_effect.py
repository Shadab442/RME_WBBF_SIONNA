"""Per-UE SINR vs. electrical downtilt.

SINR = signal power received in the SSB resource block, divided by
interference from all other sectors in that same resource block, plus noise.
Computed for every UE (dropped uniformly at random over the whole hex-grid
area, not a fixed count per sector -- see functions/uniform_ue_drop.py),
for several common downtilt settings (applied identically to every sector),
compared as SINR CDFs.

The channel is frequency-selective: Sionna's own GenerateOFDMChannel
produces the full per-element, per-subcarrier channel (pathloss, shadow
fading, AND multipath fast fading all included, exactly as the 3GPP model
generates them) for the per-sector antenna array and
each sector's own current ElectricalDowntilt.weights() combines that
sector's elements into its port, per subcarrier, before power/SINR are
computed. No fixed rx-per-tx association anywhere (see
functions/kpi_calculator.py): every sector is evaluated as a candidate
server for every UE.

Style follows sionna-sls/e2e_example.py (config.seed/precision, antenna
array/ResourceGrid construction) where it fits;
the parts of that example
that don't apply here (scheduler, link adaptation, power control, per-UE
precoding) are skipped -- this project is single-port/rank-1, no MAC layer.

Run: python plots/scripts/test_tilt_effect.py
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

import sionna
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi, UMa, RMa
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from sionna.phy.ofdm import ResourceGrid
from sionna.phy.constants import BOLTZMANN_CONSTANT

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from functions.uniform_ue_drop import UniformDropTopology
from functions.hex_grid_view import HexGridView
from functions.electrical_downtilt import ElectricalDowntilt
from functions.kpi_calculator import KpiCalculator

sionna.phy.config.seed = 42
sionna.phy.config.precision = "single"
DEVICE = "cuda:0"  # "cpu" or "cuda:0" -- every Sionna object below (Object subclasses:
                # HexGrid, AntennaArray, UMi/UMa/RMa, GenerateOFDMChannel, ResourceGrid, ...)
                # follows this automatically whenever its own device=None (the default)
sionna.phy.config.device = DEVICE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tilt_effect")
os.makedirs(OUT_DIR, exist_ok=True)

# parameters
CARRIER_FREQUENCY = 3.5e9  # Hz -- n78 FR1 mid-band
SCENARIO = "umi"
NUM_RINGS = 2
NUM_UT = 100  # total UEs, dropped uniformly over the whole area
UT_HEIGHT = 1.5  # m
M = 8  # elements per sector antenna array
ANTENNA_WINDOW = "rectangular"  # "rectangular" (eq. 7.3-1 default) or "hanning"
BS_TX_POWER_DBM = 40.0  # total conducted power per sector, over the full channel BW
CHANNEL_BANDWIDTH_PRB = 50  # ~20 MHz @ 30 kHz SCS -- the power pool BS_TX_POWER_DBM
                            # is spread over (SSB REs share this, not a private budget)
TEMPERATURE = 294  # K
DOWNTILT_SWEEP_DEG = list(range(2, 13, 2))  # deg -- one sweep, used by every plot
TOTAL_REALIZATIONS = 500  # total independent (topology, channel) draws to average over
MAX_REALIZATION_CUDA = 1  # realizations processed in one batch

assert TOTAL_REALIZATIONS % MAX_REALIZATION_CUDA == 0, \
    "TOTAL_REALIZATIONS must be a multiple of MAX_REALIZATION_CUDA"
NUM_CHUNKS = TOTAL_REALIZATIONS // MAX_REALIZATION_CUDA

_SCENARIO_CHANNEL_MODEL = {"umi": UMi, "uma": UMa, "rma": RMa}

# topology parameters
min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, indoor_probability, \
    min_ut_velocity, max_ut_velocity = set_3gpp_scenario_parameters(SCENARIO)
scenario_params = {"isd": isd, "bs_height": bs_height, "min_bs_ut_dist": min_bs_ut_dist,
                   "min_ut_height": min_ut_height}

# Uniform drop topology with hexagonal cellular scenario
topo = UniformDropTopology(scenario_params, num_rings=NUM_RINGS,
                           batch_size=MAX_REALIZATION_CUDA)
# Number of BS
num_bs = topo.num_bs

# UT locations
ut_loc = topo.sample_ut_loc(NUM_UT, UT_HEIGHT, dtype=topo.bs_loc.dtype,
                           device=topo.bs_loc.device, batch_size=MAX_REALIZATION_CUDA)

# Zero antenna orientation, all outdoor, no movement of UTs
ut_orientations = torch.zeros(MAX_REALIZATION_CUDA, NUM_UT, 3, dtype=topo.bs_loc.dtype,
                              device=topo.bs_loc.device)
ut_velocities = torch.zeros(MAX_REALIZATION_CUDA, NUM_UT, 3, dtype=topo.bs_loc.dtype,
                            device=topo.bs_loc.device)
in_state = torch.zeros(MAX_REALIZATION_CUDA, NUM_UT, dtype=torch.bool,
                       device=topo.bs_loc.device)  # all outdoor

# Visualize the hexaogonal scenario
HexGridView(topo.grid, ut_loc[0:1]).save(os.path.join(OUT_DIR, "scenario.png"))
print(f"Scenario: {NUM_RINGS} ring(s), {topo.num_cells} sites, {num_bs} sectors, "
     f"{NUM_UT} UEs (uniform over the whole area)")

# Antenna array
bs_array = AntennaArray(num_rows=M, num_cols=1, polarization="single",
                        polarization_type="V", antenna_pattern="38.901",
                        carrier_frequency=CARRIER_FREQUENCY)
ut_array = Antenna(polarization="single", polarization_type="V",
                   antenna_pattern="omni", carrier_frequency=CARRIER_FREQUENCY)

# Sector E-tilts
sector_etilts = [ElectricalDowntilt(bs_array, carrier_frequency=CARRIER_FREQUENCY,
                                    window=ANTENNA_WINDOW)
                for _ in range(num_bs)]

# Channel model
channel_model_cls = _SCENARIO_CHANNEL_MODEL[SCENARIO]
channel_model_kwargs = dict(carrier_frequency=CARRIER_FREQUENCY, bs_array=bs_array,
                           ut_array=ut_array, direction="downlink",
                           enable_pathloss=True, enable_shadow_fading=True)
if SCENARIO != "rma": channel_model_kwargs["o2i_model"] = "low"
channel_model = channel_model_cls(**channel_model_kwargs)

# SSB-like resource grid (Section 7.4.3, 3GPP TR 38.211)
# FR1 mid-band, 240 subcarriers (20 PRB), 30 kHz SCS, 4 OFDM symbols
resource_grid = ResourceGrid(num_ofdm_symbols=4, fft_size=240,
                             subcarrier_spacing=30e3, num_tx=1,
                             num_streams_per_tx=1)


# Tx power
bs_tx_power_w = 10 ** ((BS_TX_POWER_DBM - 30) / 10)
num_subcarriers_full_bw = CHANNEL_BANDWIDTH_PRB * 12
tx_power_per_subcarrier_w = bs_tx_power_w / num_subcarriers_full_bw

# Noise power
noise_power_w = BOLTZMANN_CONSTANT * TEMPERATURE * resource_grid.subcarrier_spacing

# KPI computation
kpi_calc = KpiCalculator(channel_model, resource_grid, sector_etilts,
                        tx_power_per_subcarrier_w, noise_power_w)

# Plots
def ecdf(x: np.ndarray):
    x_sorted = np.sort(x)
    return x_sorted, np.arange(1, len(x_sorted) + 1) / len(x_sorted)


def sample_sinr_db_crn(tilt_values, total_realizations, max_realization_cuda, resample_topology=True):
    """Evaluate all tilt values using common random realizations
    in fixed-size batches.

    Each batch contains independent topology and channel realizations
    shared across all tilt values. Results are accumulated into preallocated
    arrays while limiting peak memory to max_realization_cuda.

    :return: (sinr_by_tilt, rsrp_by_tilt) -- each a dict mapping every tilt
        value to pooled per-UE samples (SINR in dB, RSRP in dBm for the
        same argmax-SINR serving sector), shape [total_realizations * num_ue].
    """
    num_chunks = total_realizations // max_realization_cuda
    n_samples = total_realizations * NUM_UT
    sinr_by_tilt = {t: np.zeros(n_samples) for t in tilt_values}
    rsrp_by_tilt = {t: np.zeros(n_samples) for t in tilt_values}

    for chunk_idx in range(num_chunks):
        if resample_topology:
            new_ut_loc = topo.sample_ut_loc(
                NUM_UT, UT_HEIGHT, dtype=topo.bs_loc.dtype, device=topo.bs_loc.device,
                batch_size=max_realization_cuda,
            )
            new_bs_virtual_loc = topo.mirror_bs_loc(new_ut_loc)
            channel_model.set_topology(new_ut_loc, topo.bs_loc, ut_orientations,
                                       topo.bs_orientations, ut_velocities, in_state,
                                       None, new_bs_virtual_loc)
        h_freq = kpi_calc.generate_h_freq(batch_size=max_realization_cuda)  # one chunk's channel draw
        start = chunk_idx * max_realization_cuda * NUM_UT
        end = start + max_realization_cuda * NUM_UT
        for downtilt_deg in tilt_values:
            for etilt in sector_etilts:
                etilt.set_tilt(downtilt_deg)
            sinr_db, rsrp_dbm = kpi_calc.compute_ue_sinr_rsrp(h_freq)
            sinr_by_tilt[downtilt_deg][start:end] = sinr_db.reshape(-1)
            rsrp_by_tilt[downtilt_deg][start:end] = rsrp_dbm.reshape(-1)
    return sinr_by_tilt, rsrp_by_tilt


# One common-random-numbers sweep, reused by all plots below.
sinr_by_tilt, rsrp_by_tilt = sample_sinr_db_crn(DOWNTILT_SWEEP_DEG, TOTAL_REALIZATIONS,
                                                MAX_REALIZATION_CUDA, resample_topology=True)

# --------------------------------SINR CDF for different tilts-----------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for downtilt_deg in DOWNTILT_SWEEP_DEG:
    sinr_db = sinr_by_tilt[downtilt_deg]
    x, y = ecdf(sinr_db)
    ax.plot(x, y, label=f"downtilt = {downtilt_deg:+.0f}°")
    print(f"downtilt {downtilt_deg:+.0f}deg: median SINR = {np.median(sinr_db):.1f} dB, "
         f"5th pct = {np.percentile(sinr_db, 5):.1f} dB")

ax.set_xlabel("SINR (dB)")
ax.set_ylabel("CDF")
ax.set_title(f"Per-UE SS-SINR CDF vs. electrical downtilt\n"
            f"({NUM_RINGS} ring(s), {num_bs} sectors, M={M}, frequency-selective)")
ax.grid(True, alpha=0.3)
ax.legend()
out_path = os.path.join(OUT_DIR, "sinr_cdf_vs_downtilt.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# ------------------cell-edge (5th pct) vs. tilt----------------------------
medians_db, p5_db, p1_db = [], [], []
for downtilt_deg in DOWNTILT_SWEEP_DEG:
    sinr_db = sinr_by_tilt[downtilt_deg]
    medians_db.append(np.median(sinr_db))
    p5_db.append(np.percentile(sinr_db, 5))
    p1_db.append(np.percentile(sinr_db, 1))
    print(f"downtilt {downtilt_deg:+3.0f}deg ({len(sinr_db)} samples): "
         f"median = {medians_db[-1]:5.1f} dB, "
         f"5th pct = {p5_db[-1]:5.1f} dB, 1st pct = {p1_db[-1]:5.1f} dB")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(DOWNTILT_SWEEP_DEG, medians_db, "o-", label="median (50th pct)")
ax.plot(DOWNTILT_SWEEP_DEG, p5_db, "o-", label="cell-edge (5th pct)")
ax.plot(DOWNTILT_SWEEP_DEG, p1_db, "o-", label="1st pct")
ax.set_xlabel("Electrical downtilt (deg)")
ax.set_ylabel("SINR (dB)")
ax.set_title(f"SINR vs. electrical downtilt, by percentile\n"
            f"({NUM_RINGS} ring(s), {num_bs} sectors, M={M}, frequency-selective)")
ax.grid(True, alpha=0.3)
ax.legend()
out_path = os.path.join(OUT_DIR, "cell_edge_sinr_vs_downtilt.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# ------------------RSRP (serving-sector received power) vs. tilt-----------
# Same argmax-SINR serving sector as the SINR plots above -- this shows
# whether extreme tilts create a real coverage hole (absolute power) even
# where SINR looked comparable (a ratio, less sensitive to a shared tilt
# suppressing signal and interference together).
rsrp_medians_dbm, rsrp_p5_dbm, rsrp_p1_dbm = [], [], []
for downtilt_deg in DOWNTILT_SWEEP_DEG:
    rsrp_dbm = rsrp_by_tilt[downtilt_deg]
    rsrp_medians_dbm.append(np.median(rsrp_dbm))
    rsrp_p5_dbm.append(np.percentile(rsrp_dbm, 5))
    rsrp_p1_dbm.append(np.percentile(rsrp_dbm, 1))
    print(f"downtilt {downtilt_deg:+3.0f}deg: RSRP median = {rsrp_medians_dbm[-1]:6.1f} dBm, "
         f"5th pct = {rsrp_p5_dbm[-1]:6.1f} dBm, 1st pct = {rsrp_p1_dbm[-1]:6.1f} dBm")

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(DOWNTILT_SWEEP_DEG, rsrp_medians_dbm, "o-", label="median (50th pct)")
ax.plot(DOWNTILT_SWEEP_DEG, rsrp_p5_dbm, "o-", label="cell-edge (5th pct)")
ax.plot(DOWNTILT_SWEEP_DEG, rsrp_p1_dbm, "o-", label="1st pct")
ax.set_xlabel("Electrical downtilt (deg)")
ax.set_ylabel("RSRP (dBm)")
ax.set_title(f"Serving-sector RSRP vs. electrical downtilt, by percentile\n"
            f"({NUM_RINGS} ring(s), {num_bs} sectors, M={M}, frequency-selective)")
ax.grid(True, alpha=0.3)
ax.legend()
out_path = os.path.join(OUT_DIR, "rsrp_vs_downtilt.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# --------------------------------RSRP CDF for different tilts-----------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for downtilt_deg in DOWNTILT_SWEEP_DEG:
    rsrp_dbm = rsrp_by_tilt[downtilt_deg]
    x, y = ecdf(rsrp_dbm)
    ax.plot(x, y, label=f"downtilt = {downtilt_deg:+.0f}°")

ax.set_xlabel("RSRP (dBm)")
ax.set_ylabel("CDF")
ax.set_title(f"Serving-sector RSRP CDF vs. electrical downtilt\n"
            f"({NUM_RINGS} ring(s), {num_bs} sectors, M={M}, frequency-selective)")
ax.grid(True, alpha=0.3)
ax.legend()
out_path = os.path.join(OUT_DIR, "rsrp_cdf_vs_downtilt.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
