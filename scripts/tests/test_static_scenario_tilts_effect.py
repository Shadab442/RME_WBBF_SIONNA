"""Spatial-variability sanity check: does the coverage-maximizing tilt
differ from sector to sector, for a static (non-mobile) UE population?

UEs are dropped in clusters (same generative model as
test_dynamic_scenario_tilts_effect.py's RPGM population: NUM_GROUPS clusters
split evenly across sites via helpers.ue_drop.sample_cluster_start_xy, each
cluster's members within DEVIATION_RADIUS_FRAC_AREA of its center), not
uniformly at random -- so this script's population matches the mobility
script's population structure exactly, just held at one snapshot instead of
evolving over time. Every pooled realization draws a FRESH set of cluster
centers and members (not one fixed layout reused across realizations), for
the same statistical-independence reason the old uniform-drop version
resampled UE positions every chunk.

Two baselines, both maximizing pooled coverage (fraction of UEs with
best-serving-sector SINR > COVERAGE_THRESHOLD_DB):

* Static Global: one common tilt, applied to every sector -- exhaustive
  search over DOWNTILT_SWEEP_DEG (helpers.tilt_controller.GlobalTiltSelector).
* Static Local: per-sector tilts, found by coordinate ascent
  (helpers.tilt_controller.LocalTiltSelector) starting from Static Global --
  guaranteed no worse than Static Global's coverage, since "keep the current
  tilt" is always an available move each round.

Both assignments are evaluated off a single shared power table
(helpers.kpi_calculator.KpiCalculator.compute_power_table): each candidate
tilt is swept once (applied to every sector, exactly like a normal tilt
sweep) and each sector's own row kept, since a sector's received power
depends only on its own antenna weights and the channel to each UE -- never
on any other sector's tilt. Any per-sector assignment can then be evaluated
by array lookup alone (KpiCalculator.coverage_from_assignment), no repeated
channel computation.

This script only computes and saves data (results/tests/static_scenario_tilts_effect/
data.npz, plus scenario.png) -- run plot_static_scenario_tilts_effect.py
separately to produce the plots.

Run: python scripts/tests/test_static_scenario_tilts_effect.py
"""

import os
import sys

import numpy as np
import torch

import sionna
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi, UMa, RMa
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from sionna.phy.constants import BOLTZMANN_CONSTANT

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from helpers.cellular_topology import CellularTopology
from helpers.config import load_config
from helpers.ue_drop import sample_cluster_start_xy, sample_clustered_ut_loc
from helpers.scenario_view import save_scenario, compute_cell_colors
from helpers.electrical_downtilt import ElectricalDowntilt
from helpers.kpi_calculator import KpiCalculator
from helpers.tilt_controller import GlobalTiltSelector, LocalTiltSelector

sionna.phy.config.seed = 42
sionna.phy.config.precision = "single"
DEVICE = "cuda:0"
sionna.phy.config.device = DEVICE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "static_scenario_tilts_effect")
os.makedirs(OUT_DIR, exist_ok=True)

CFG = load_config("test_static_scenario_tilts_effect")

# parameters
CARRIER_FREQUENCY = CFG["carrier_frequency"]  # Hz -- n78 FR1 mid-band
SCENARIO = CFG["scenario"]
NUM_RINGS = CFG["num_rings"]  # matches the mobility scripts
NUM_UT = CFG["num_ut"]  # total UEs
UT_HEIGHT = CFG["ut_height"]  # m
# Clustered population -- same values as test_dynamic_scenario_tilts_effect.py,
# so the two scripts' populations only differ in whether they move over time.
NUM_GROUPS = CFG["num_groups"]
MEMBERS_PER_GROUP = NUM_UT // NUM_GROUPS
DEVIATION_RADIUS_FRAC_AREA = CFG["deviation_radius_frac_area"]
M = CFG["m"]  # elements per sector antenna array
ANTENNA_WINDOW = CFG["antenna_window"]
BS_TX_POWER_DBM = CFG["bs_tx_power_dbm"]
CHANNEL_BANDWIDTH_PRB = CFG["channel_bandwidth_prb"]  # ~20 MHz @ 30 kHz SCS
TEMPERATURE = CFG["temperature"]  # K
DOWNTILT_SWEEP_DEG = CFG["downtilt_sweep_deg"]
TOTAL_REALIZATIONS = CFG["total_realizations"]  # total independent (topology, channel) draws pooled together
MAX_REALIZATION_CUDA = CFG["max_realization_cuda"]
# Coverage objective's threshold -- this drives the tilt SEARCH itself (both
# baselines maximize coverage AT this threshold), unlike plot-only
# tunables (e.g. moving-average windows) that only affect display.
COVERAGE_THRESHOLD_DB = CFG["coverage_threshold_db"]
COORDINATE_ASCENT_MAX_ROUNDS = CFG["coordinate_ascent_max_rounds"]

assert TOTAL_REALIZATIONS % MAX_REALIZATION_CUDA == 0, \
    "TOTAL_REALIZATIONS must be a multiple of MAX_REALIZATION_CUDA"
assert NUM_UT % NUM_GROUPS == 0, \
    "NUM_UT must be divisible by NUM_GROUPS for equal-sized clusters"
NUM_CHUNKS = TOTAL_REALIZATIONS // MAX_REALIZATION_CUDA

_SCENARIO_CHANNEL_MODEL = {"umi": UMi, "uma": UMa, "rma": RMa}

# topology parameters
min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, indoor_probability, \
    min_ut_velocity, max_ut_velocity = set_3gpp_scenario_parameters(SCENARIO)
scenario_params = {"isd": isd, "bs_height": bs_height, "min_bs_ut_dist": min_bs_ut_dist,
                   "min_ut_height": min_ut_height}

topo = CellularTopology(scenario_params, num_rings=NUM_RINGS, batch_size=MAX_REALIZATION_CUDA)
num_bs = topo.num_bs

deviation_radius = DEVIATION_RADIUS_FRAC_AREA * topo.default_drop_radius


def sample_clustered_ut_loc_batch(batch_size):
    """[batch_size, NUM_UT, 3] -- batch_size independent clustered drops,
    each with its own FRESH cluster centers (not shared across the batch),
    matching how sample_uniform_ut_loc(..., batch_size=...) drew an
    independent uniform snapshot per batch element."""
    ut_locs = []
    for _ in range(batch_size):
        start_xy_list = sample_cluster_start_xy(topo, NUM_GROUPS)
        ut_loc_b, _ = sample_clustered_ut_loc(
            topo, start_xy_list, MEMBERS_PER_GROUP, deviation_radius, UT_HEIGHT,
            dtype=topo.bs_loc.dtype, device=topo.bs_loc.device,
        )
        ut_locs.append(ut_loc_b)
    return torch.stack(ut_locs, dim=0)


ut_orientations = torch.zeros(MAX_REALIZATION_CUDA, NUM_UT, 3, dtype=topo.bs_loc.dtype,
                              device=topo.bs_loc.device)
ut_velocities = torch.zeros(MAX_REALIZATION_CUDA, NUM_UT, 3, dtype=topo.bs_loc.dtype,
                            device=topo.bs_loc.device)
in_state = torch.zeros(MAX_REALIZATION_CUDA, NUM_UT, dtype=torch.bool,
                       device=topo.bs_loc.device)

# Representative snapshot for the scenario.png setup plot -- the pooled
# sweep below draws its own fresh clustered UE positions every chunk.
_snapshot_start_xy = sample_cluster_start_xy(topo, NUM_GROUPS)
_snapshot_ut_loc, _snapshot_member_group_idx = sample_clustered_ut_loc(
    topo, _snapshot_start_xy, MEMBERS_PER_GROUP, deviation_radius, UT_HEIGHT,
    dtype=topo.bs_loc.dtype, device=topo.bs_loc.device,
)
_cluster_colors, _ = compute_cell_colors(_snapshot_start_xy, topo.site_loc, topo.num_sectors_per_site)
_snapshot_colors = [_cluster_colors[g] for g in _snapshot_member_group_idx.tolist()]
save_scenario(os.path.join(OUT_DIR, "scenario.png"), topo.grid, _snapshot_ut_loc, colors=_snapshot_colors,
             cluster_centers=_snapshot_start_xy, cluster_radius=deviation_radius)
print(f"Scenario: {NUM_RINGS} ring(s), {topo.num_cells} sites, {num_bs} sectors, "
     f"{NUM_UT} UEs ({NUM_GROUPS} clusters x {MEMBERS_PER_GROUP} UEs)")

bs_array = AntennaArray(num_rows=M, num_cols=1, polarization="single",
                        polarization_type="V", antenna_pattern="38.901",
                        carrier_frequency=CARRIER_FREQUENCY)
ut_array = Antenna(polarization="single", polarization_type="V",
                   antenna_pattern="omni", carrier_frequency=CARRIER_FREQUENCY)

sector_etilts = [ElectricalDowntilt(bs_array, carrier_frequency=CARRIER_FREQUENCY,
                                    window=ANTENNA_WINDOW)
                for _ in range(num_bs)]

channel_model_cls = _SCENARIO_CHANNEL_MODEL[SCENARIO]
channel_model_kwargs = dict(carrier_frequency=CARRIER_FREQUENCY, bs_array=bs_array,
                           ut_array=ut_array, direction="downlink",
                           enable_pathloss=True, enable_shadow_fading=True)
if SCENARIO != "rma": channel_model_kwargs["o2i_model"] = "low"
channel_model = channel_model_cls(**channel_model_kwargs)

bs_tx_power_w = 10 ** ((BS_TX_POWER_DBM - 30) / 10)
channel_bandwidth_hz = CHANNEL_BANDWIDTH_PRB * 12 * 30e3  # 30 kHz SCS
noise_power_w = BOLTZMANN_CONSTANT * TEMPERATURE * channel_bandwidth_hz

kpi_calc = KpiCalculator(channel_model, sector_etilts, bs_tx_power_w, noise_power_w)


def build_power_table_crn(tilt_values, total_realizations, max_realization_cuda):
    """Pooled [num_tilts, num_sectors, total_realizations*num_ue] power
    table -- fresh clustered UE positions and channel every chunk,
    concatenated along the sample axis (a plain tilt sweep at every chunk,
    per helpers.kpi_calculator.KpiCalculator.compute_power_table)."""
    num_chunks = total_realizations // max_realization_cuda
    chunks = []
    for chunk_idx in range(num_chunks):
        new_ut_loc = sample_clustered_ut_loc_batch(max_realization_cuda)
        new_bs_virtual_loc = topo.mirror_bs_loc(new_ut_loc)
        channel_model.set_topology(new_ut_loc, topo.bs_loc, ut_orientations,
                                   topo.bs_orientations, ut_velocities, in_state,
                                   None, new_bs_virtual_loc)
        state = kpi_calc.generate_large_scale_state()
        chunks.append(kpi_calc.compute_power_table(tilt_values, state))
    return np.concatenate(chunks, axis=2)


power_table = build_power_table_crn(DOWNTILT_SWEEP_DEG, TOTAL_REALIZATIONS, MAX_REALIZATION_CUDA)
print(f"Power table: {power_table.shape} (tilts x sectors x pooled samples)")

global_selector = GlobalTiltSelector()
init_assignment = global_selector.select(kpi_calc, power_table, COVERAGE_THRESHOLD_DB)
static_global_tilt_idx = int(init_assignment[0])
global_coverage = kpi_calc.coverage_from_assignment(power_table, init_assignment, COVERAGE_THRESHOLD_DB)
print(f"Static Global: {DOWNTILT_SWEEP_DEG[static_global_tilt_idx]:g}° "
     f"-> coverage={global_coverage:.4f}")

local_selector = LocalTiltSelector(max_rounds=COORDINATE_ASCENT_MAX_ROUNDS)
local_tilt_idx_per_sector = local_selector.select(
    kpi_calc, power_table, COVERAGE_THRESHOLD_DB, warm_start=init_assignment
)
num_rounds, coverage_trace = local_selector.last_num_rounds, local_selector.last_coverage_trace
local_coverage = float(coverage_trace[-1])
converged = num_rounds < COORDINATE_ASCENT_MAX_ROUNDS or coverage_trace[-1] == coverage_trace[-2]
print(f"Static Local: converged after {num_rounds} round(s) "
     f"({'converged' if converged else 'hit max_rounds, may not have converged'}) "
     f"-> coverage={local_coverage:.4f} (gap over Global: {local_coverage - global_coverage:+.4f})")

global_coverage_check, sinr_db_global, serving_idx_global = kpi_calc.coverage_from_assignment(
    power_table, init_assignment, COVERAGE_THRESHOLD_DB, return_details=True
)
local_coverage_check, sinr_db_local, serving_idx_local = kpi_calc.coverage_from_assignment(
    power_table, local_tilt_idx_per_sector, COVERAGE_THRESHOLD_DB, return_details=True
)
assert np.isclose(global_coverage_check, global_coverage)
assert np.isclose(local_coverage_check, local_coverage)

per_sector_coverage_global = np.full(num_bs, np.nan)
per_sector_coverage_local = np.full(num_bs, np.nan)
for s in range(num_bs):
    mask_global = serving_idx_global == s
    if mask_global.any():
        per_sector_coverage_global[s] = np.mean(sinr_db_global[mask_global] > COVERAGE_THRESHOLD_DB)
    mask_local = serving_idx_local == s
    if mask_local.any():
        per_sector_coverage_local[s] = np.mean(sinr_db_local[mask_local] > COVERAGE_THRESHOLD_DB)

local_tilt_deg_per_sector = np.asarray(DOWNTILT_SWEEP_DEG)[local_tilt_idx_per_sector]
print(f"Static Local per-sector tilts: {sorted(set(local_tilt_deg_per_sector.tolist()))} "
     f"(distinct values in use)")

data_path = os.path.join(OUT_DIR, "data.npz")
np.savez(
    data_path,
    downtilt_sweep_deg=np.asarray(DOWNTILT_SWEEP_DEG),
    coverage_threshold_db=COVERAGE_THRESHOLD_DB,
    num_rings=NUM_RINGS,
    num_bs=num_bs,
    num_ut=NUM_UT,
    total_realizations=TOTAL_REALIZATIONS,
    static_global_tilt_deg=DOWNTILT_SWEEP_DEG[static_global_tilt_idx],
    global_coverage=global_coverage,
    local_tilt_deg_per_sector=local_tilt_deg_per_sector,
    local_coverage=local_coverage,
    coverage_trace=coverage_trace,
    num_rounds=num_rounds,
    sinr_db_global=sinr_db_global,
    sinr_db_local=sinr_db_local,
    serving_idx_global=serving_idx_global,
    serving_idx_local=serving_idx_local,
    per_sector_coverage_global=per_sector_coverage_global,
    per_sector_coverage_local=per_sector_coverage_local,
)
print(f"Saved: {data_path}")
