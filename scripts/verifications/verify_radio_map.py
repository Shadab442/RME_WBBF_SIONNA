"""Verification: does helpers/radio_map.py's precomputed power map agree
with live per-UE power computed the normal way (helpers.kpi_calculator.
KpiCalculator.compute_power_table), and does it show the physical trends
already independently verified for the antenna pattern itself (power falls
off with distance; downtilt suppresses far-field power more than
near-field power, per verify_tilt_effect.py)?

Run: python scripts/verifications/verify_radio_map.py

Saves to results/verifications/radio_map/:
  1. grid_scenario.png        -- the radio map's grid points over the hex
                                 topology -- sanity-checks coverage/filtering.
  2. power_heatmap_sector0.png -- interpolated dB heatmap of sector 0's
                                 received power at one representative tilt --
                                 should show a clear lobe toward that
                                 sector's boresight, decaying with distance.
  3. tilt_sensitivity.png     -- a near and a far grid point's power across
                                 the full tilt sweep -- downtilt should
                                 suppress the far point more than the near one.
  4. live_vs_map_agreement.png -- live per-UE power (freshly drawn UEs,
                                 standard pipeline) vs. the radio map's
                                 power at each UE's nearest grid point, same
                                 tilt -- should track the y=x line if the
                                 map is correct.
"""

import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sionna
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from sionna.phy.constants import BOLTZMANN_CONSTANT

from helpers.cellular_topology import CellularTopology
from helpers.config import load_config
from helpers.electrical_downtilt import ElectricalDowntilt
from helpers.kpi_calculator import KpiCalculator
from helpers.radio_map import build_radio_map, nearest_grid_power
from helpers.scenario_view import save_scenario
from helpers.ue_drop import sample_grid_points, sample_uniform_ut_loc

sionna.phy.config.seed = 42
sionna.phy.config.precision = "single"
DEVICE = "cuda:0"
sionna.phy.config.device = DEVICE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "verifications", "radio_map")
os.makedirs(OUT_DIR, exist_ok=True)

CFG = load_config("verify_radio_map")

CARRIER_FREQUENCY = CFG["carrier_frequency"]
SCENARIO = CFG["scenario"]
NUM_RINGS = CFG["num_rings"]  # matches the other scripts
UT_HEIGHT = CFG["ut_height"]
M = CFG["m"]
ANTENNA_WINDOW = CFG["antenna_window"]
BS_TX_POWER_DBM = CFG["bs_tx_power_dbm"]
CHANNEL_BANDWIDTH_PRB = CFG["channel_bandwidth_prb"]
TEMPERATURE = CFG["temperature"]
DOWNTILT_SWEEP_DEG = CFG["downtilt_sweep_deg"]  # same sweep as the tilts_effect scripts

GRID_SPACING_M = CFG["grid_spacing_m"]  # radio map lattice spacing
NUM_REALIZATIONS = CFG["num_realizations"]  # averaged per grid point, to smooth out shadow fading
MAX_REALIZATION_CUDA = CFG["max_realization_cuda"]
GRID_CHUNK_SIZE = CFG["grid_chunk_size"]  # grid points per channel computation -- a radio map's
                       # grid is denser than a normal UE drop, so it's
                       # chunked too (see build_radio_map), not just realizations

NUM_LIVE_UT = CFG["num_live_ut"]  # freshly-drawn UEs for the live-vs-map agreement check
LIVE_CHECK_TILT_DEG = CFG["live_check_tilt_deg"]  # single tilt used for that check

assert NUM_REALIZATIONS % MAX_REALIZATION_CUDA == 0, \
    "NUM_REALIZATIONS must be a multiple of MAX_REALIZATION_CUDA"

min_bs_ut_dist, isd, bs_height, min_ut_height, _, _, _, _ = set_3gpp_scenario_parameters(SCENARIO)
scenario_params = {"isd": isd, "bs_height": bs_height, "min_bs_ut_dist": min_bs_ut_dist,
                   "min_ut_height": min_ut_height}
topo = CellularTopology(scenario_params, num_rings=NUM_RINGS, batch_size=MAX_REALIZATION_CUDA)
num_bs = topo.num_bs

grid_loc = sample_grid_points(topo, GRID_SPACING_M, UT_HEIGHT, dtype=topo.bs_loc.dtype, device=topo.bs_loc.device)
num_grid_points = grid_loc.shape[0]
print(f"Grid: {num_grid_points} points at {GRID_SPACING_M:g} m spacing "
     f"({NUM_RINGS} ring(s), {topo.num_cells} sites, {num_bs} sectors)")

save_scenario(os.path.join(OUT_DIR, "grid_scenario.png"), topo.grid, grid_loc, colors="tab:green")

bs_array = AntennaArray(num_rows=M, num_cols=1, polarization="single", polarization_type="V",
                        antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY)
ut_array = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                   carrier_frequency=CARRIER_FREQUENCY)
sector_etilts = [ElectricalDowntilt(bs_array, carrier_frequency=CARRIER_FREQUENCY, window=ANTENNA_WINDOW)
                for _ in range(num_bs)]
channel_model = UMi(carrier_frequency=CARRIER_FREQUENCY, o2i_model="low", bs_array=bs_array, ut_array=ut_array,
                    direction="downlink", enable_pathloss=True, enable_shadow_fading=True)

bs_tx_power_w = 10 ** ((BS_TX_POWER_DBM - 30) / 10)
channel_bandwidth_hz = CHANNEL_BANDWIDTH_PRB * 12 * 30e3  # 30 kHz SCS
noise_power_w = BOLTZMANN_CONSTANT * TEMPERATURE * channel_bandwidth_hz
kpi_calc = KpiCalculator(channel_model, sector_etilts, bs_tx_power_w, noise_power_w)

ut_orientations = torch.zeros(MAX_REALIZATION_CUDA, num_grid_points, 3, dtype=topo.bs_loc.dtype, device=topo.bs_loc.device)
ut_velocities = torch.zeros_like(ut_orientations)
in_state = torch.zeros(MAX_REALIZATION_CUDA, num_grid_points, dtype=torch.bool, device=topo.bs_loc.device)

# -------------------------- build the radio map -----------------------------
radio_map = build_radio_map(
    kpi_calc, channel_model, topo, grid_loc, DOWNTILT_SWEEP_DEG,
    NUM_REALIZATIONS, MAX_REALIZATION_CUDA, GRID_CHUNK_SIZE, ut_orientations, ut_velocities, in_state,
)  # [num_tilts, num_bs, num_grid_points]
print(f"Radio map: {radio_map.shape} (tilts x sectors x grid points)")

grid_xy = grid_loc[:, :2].detach().cpu().numpy()
site_xy = topo.site_loc.detach().cpu().numpy()

# ---------------- realization-to-realization variance check -----------------
# How much does a single (grid point, tilt, sector) power value vary draw to
# draw, before averaging? Large-scale only: the sole remaining source of
# variation across draws at the SAME position is shadow fading, redrawn
# each set_topology() call -- see build_radio_map's docstring.
_probe_tilt_idx = 0
_probe_sector = 0
_probe_points = grid_loc[:1]  # a single point is plenty for this probe --
                              # no need to run the full grid through the channel again
_probe_batch = _probe_points.unsqueeze(0).repeat(MAX_REALIZATION_CUDA, 1, 1)  # see build_radio_map's comment on why not .expand()
_probe_ut_orientations = ut_orientations[:, :1]
_probe_ut_velocities = ut_velocities[:, :1]
_probe_in_state = in_state[:, :1]
_single_draws = []
for _ in range(6):
    # build_radio_map's last grid chunk left the scenario at a different
    # shape -- reset before this probe's own (smaller) shape.
    channel_model.reset_topology()
    channel_model.set_topology(
        _probe_batch, topo.bs_loc, _probe_ut_orientations,
        topo.bs_orientations, _probe_ut_velocities, _probe_in_state, None,
        topo.mirror_bs_loc(_probe_batch),
    )
    state = kpi_calc.generate_large_scale_state()
    pt = kpi_calc.compute_power_table([DOWNTILT_SWEEP_DEG[_probe_tilt_idx]], state)
    pt = pt.reshape(1, num_bs, MAX_REALIZATION_CUDA, 1)
    _single_draws.append(pt[0, _probe_sector, 0, 0])  # first realization, the one probe point
_single_draws = np.asarray(_single_draws)
_draws_db = 10 * np.log10(_single_draws)
print(f"Realization-to-realization spread at one (grid point, sector, tilt): "
     f"{_draws_db.min():.1f} to {_draws_db.max():.1f} dBW across {len(_single_draws)} draws "
     f"(std={_draws_db.std():.2f} dB) -- spread here is expected (shadow fading); "
     f"averaging {NUM_REALIZATIONS} such draws is what stabilizes the map.")

# ------------------------- power vs. distance sanity ------------------------
sector0_xy = topo.bs_loc[0, 0, :2].detach().cpu().numpy()
dist_to_sector0 = np.linalg.norm(grid_xy - sector0_xy, axis=-1)
mid_tilt_idx = len(DOWNTILT_SWEEP_DEG) // 2
power_sector0_db = 10 * np.log10(radio_map[mid_tilt_idx, 0, :])
corr = np.corrcoef(dist_to_sector0, power_sector0_db)[0, 1]
print(f"Sector 0 power vs. distance correlation (at {DOWNTILT_SWEEP_DEG[mid_tilt_idx]:g}° downtilt): "
     f"{corr:+.3f} (expect clearly negative -- pathloss should dominate)")

fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(dist_to_sector0, power_sector0_db, s=6, alpha=0.5)
ax.set_xlabel("Distance to sector 0's site (m)")
ax.set_ylabel("Sector 0 received power (dBW)")
ax.set_title(f"Power vs. distance, sector 0 ({DOWNTILT_SWEEP_DEG[mid_tilt_idx]:g}° downtilt)\n"
            f"correlation={corr:+.3f}")
ax.grid(True, alpha=0.3)
path = os.path.join(OUT_DIR, "power_vs_distance_sector0.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ---------------------- sector 0 power heatmap (dB) --------------------------
fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(grid_xy[:, 0], grid_xy[:, 1], c=power_sector0_db, cmap="viridis", s=25)
ax.scatter(site_xy[:, 0], site_xy[:, 1], marker="^", color="red", s=80, label="sites", zorder=5)
fig.colorbar(sc, ax=ax, label="Sector 0 received power (dBW)")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Sector 0 power heatmap ({DOWNTILT_SWEEP_DEG[mid_tilt_idx]:g}° downtilt)\n"
            "expect a clear lobe toward sector 0's boresight, decaying outward")
ax.set_aspect("equal")
ax.legend()
path = os.path.join(OUT_DIR, "power_heatmap_sector0.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------- tilt sensitivity -------------------------------
# Near point: close to sector 0's site. Far point: the grid point farthest
# from sector 0's site with above-median power at the middle tilt (i.e.
# actually roughly in sector 0's coverage wedge, not just far away in some
# unrelated direction).
near_idx = int(np.argmin(dist_to_sector0))
plausible_far = dist_to_sector0 > np.median(dist_to_sector0)
far_candidates = np.where(plausible_far & (power_sector0_db > np.median(power_sector0_db)))[0]
far_idx = far_candidates[np.argmax(dist_to_sector0[far_candidates])] if len(far_candidates) else int(np.argmax(dist_to_sector0))

near_power_db = 10 * np.log10(radio_map[:, 0, near_idx])
far_power_db = 10 * np.log10(radio_map[:, 0, far_idx])
near_drop = near_power_db[0] - near_power_db[-1]
far_drop = far_power_db[0] - far_power_db[-1]
print(f"Tilt sensitivity, sector 0: near point ({dist_to_sector0[near_idx]:.0f} m) drops "
     f"{near_drop:+.1f} dB from {DOWNTILT_SWEEP_DEG[0]:g}° to {DOWNTILT_SWEEP_DEG[-1]:g}°; "
     f"far point ({dist_to_sector0[far_idx]:.0f} m) drops {far_drop:+.1f} dB "
     f"(expect far point to drop MORE -- downtilt suppresses far-field power more than near-field)")

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(DOWNTILT_SWEEP_DEG, near_power_db, "o-", label=f"near ({dist_to_sector0[near_idx]:.0f} m)")
ax.plot(DOWNTILT_SWEEP_DEG, far_power_db, "o-", label=f"far ({dist_to_sector0[far_idx]:.0f} m)")
ax.set_xlabel("Downtilt (deg)")
ax.set_ylabel("Sector 0 received power (dBW)")
ax.set_title("Tilt sensitivity: near vs. far grid point")
ax.grid(True, alpha=0.3)
ax.legend()
path = os.path.join(OUT_DIR, "tilt_sensitivity.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ------------------------- live vs. map agreement ----------------------------
# Freshly-drawn UEs, evaluated the STANDARD way (KpiCalculator directly, no
# radio map involved), all sectors at the SAME fixed tilt -- compared
# against the radio map's power at each UE's nearest grid point, same tilt,
# same sector. This is the strongest check: it ties the whole pipeline
# together and would catch real bugs (wrong units, wrong indexing,
# coordinate mismatches) that the sanity checks above wouldn't.
live_ut_loc = sample_uniform_ut_loc(topo, NUM_LIVE_UT, UT_HEIGHT, dtype=topo.bs_loc.dtype,
                                    device=topo.bs_loc.device, batch_size=MAX_REALIZATION_CUDA)
live_bs_virtual_loc = topo.mirror_bs_loc(live_ut_loc)
# Own correctly-sized tensors -- ut_orientations/ut_velocities/in_state above
# are sized to num_grid_points, which need not match NUM_LIVE_UT.
live_ut_orientations = torch.zeros(MAX_REALIZATION_CUDA, NUM_LIVE_UT, 3, dtype=topo.bs_loc.dtype, device=topo.bs_loc.device)
live_ut_velocities = torch.zeros_like(live_ut_orientations)
live_in_state = torch.zeros(MAX_REALIZATION_CUDA, NUM_LIVE_UT, dtype=torch.bool, device=topo.bs_loc.device)
channel_model.reset_topology()  # shape differs from the probe block above
channel_model.set_topology(live_ut_loc, topo.bs_loc, live_ut_orientations, topo.bs_orientations,
                           live_ut_velocities, live_in_state, None, live_bs_virtual_loc)
state_live = kpi_calc.generate_large_scale_state()
for etilt in sector_etilts:
    etilt.set_tilt(LIVE_CHECK_TILT_DEG)
live_power_w = kpi_calc.compute_power_matrix_w(state_live)  # [num_sectors, batch, num_ue]
live_sector0_power_db = 10 * np.log10(live_power_w[0].detach().cpu().numpy().reshape(-1))
live_ut_xy = live_ut_loc[:, :, :2].detach().cpu().numpy().reshape(-1, 2)

live_check_tilt_idx = DOWNTILT_SWEEP_DEG.index(LIVE_CHECK_TILT_DEG)
map_power_at_ues = nearest_grid_power(grid_xy, radio_map, live_ut_xy, live_check_tilt_idx, sector_idx=0)
map_sector0_power_db = 10 * np.log10(map_power_at_ues)

mae_db = np.mean(np.abs(live_sector0_power_db - map_sector0_power_db))
agreement_corr = np.corrcoef(live_sector0_power_db, map_sector0_power_db)[0, 1]
print(f"Live vs. map agreement (sector 0, {LIVE_CHECK_TILT_DEG:g}°, {NUM_LIVE_UT} live UEs): "
     f"mean abs error={mae_db:.2f} dB, correlation={agreement_corr:+.3f} "
     f"(expect correlation clearly positive; some spread is expected -- live UEs get their OWN "
     f"shadow fading draw, the map only gives the nearest grid cell's AVERAGE)")

fig, ax = plt.subplots(figsize=(6.5, 6.5))
lo = min(live_sector0_power_db.min(), map_sector0_power_db.min())
hi = max(live_sector0_power_db.max(), map_sector0_power_db.max())
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
ax.scatter(map_sector0_power_db, live_sector0_power_db, s=10, alpha=0.5)
ax.set_xlabel("Radio map power at nearest grid point (dBW)")
ax.set_ylabel("Live per-UE power (dBW)")
ax.set_title(f"Live vs. map agreement, sector 0 @ {LIVE_CHECK_TILT_DEG:g}°\n"
            f"MAE={mae_db:.2f} dB, correlation={agreement_corr:+.3f}")
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
ax.legend()
path = os.path.join(OUT_DIR, "live_vs_map_agreement.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")
