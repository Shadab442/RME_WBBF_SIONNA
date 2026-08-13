"""Tilt control window calibration: how much does a single per-sector tilt
assignment, chosen non-causally to be the BEST possible for a window of W
consecutive slots, lose relative to the finest-possible per-slot oracle
(W=1), as W grows?

This isolates pure windowing/quantization loss from causal staleness: both
the per-slot oracle and the windowed assignment are non-causal (each sees
its own window's full data, including "future" slots relative to the
window's start) and use the exact same search (LocalTiltSelector,
coordinate ascent, always warm_start=None so nothing carries momentum
across window sizes) -- the ONLY thing varying across the sweep is W
itself. At W=1 this reduces to exactly the per-slot oracle, so the gap at
W=1 must be exactly zero -- a built-in correctness check, not just an
expectation.

Mobility is fixed to the primary 1-2 m/s target for this run (see
MOBILITY_SPEED_SETTINGS) -- the window size doesn't need to work for every
possible mobility, just this one; higher speeds are a separate follow-up
check once a candidate W is chosen here, not something this sweep needs to
be robust to.

Every candidate window size in CANDIDATE_WINDOW_SLOTS must evenly divide
the largest one -- slots are processed in "mega-chunks" of
max(CANDIDATE_WINDOW_SLOTS), so every smaller candidate's windows align
cleanly within each chunk and only one chunk's worth of power tables is
ever held in memory at a time.

Saves to results/tests/tilt_window_calibration/data.npz -- run
plot_tilt_window_calibration.py separately to produce the plots (coverage
vs. window size, and per-sector tilt heatmaps comparing the windowed
assignment against the per-slot oracle).

Run: python scripts/tests/test_tilt_window_calibration.py
"""

import os
import sys

import numpy as np
import torch

import sionna
from sionna.phy.channel.tr38901 import Antenna, AntennaArray, RMa, UMa, UMi
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from sionna.phy.constants import BOLTZMANN_CONSTANT

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from helpers.cellular_topology import CellularTopology
from helpers.config import load_config
from helpers.electrical_downtilt import ElectricalDowntilt
from helpers.kpi_calculator import KpiCalculator
from helpers.mobility import ReferencePointGroupMobility
from helpers.ue_drop import sample_cluster_start_xy, sample_clustered_ut_loc
from helpers.tilt_controller import LocalTiltSelector

sionna.phy.config.seed = 42
sionna.phy.config.precision = "single"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cuda:0"
sionna.phy.config.device = DEVICE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "tilt_window_calibration")
os.makedirs(OUT_DIR, exist_ok=True)

CFG = load_config("test_tilt_window_calibration")

CARRIER_FREQUENCY = CFG["carrier_frequency"]
SCENARIO = CFG["scenario"]
NUM_RINGS = CFG["num_rings"]
NUM_UT = CFG["num_ut"]
UT_HEIGHT = CFG["ut_height"]
M = CFG["m"]
DOWNTILT_SWEEP_DEG = CFG["downtilt_sweep_deg"]
ANTENNA_WINDOW = CFG["antenna_window"]
BS_TX_POWER_DBM = CFG["bs_tx_power_dbm"]
CHANNEL_BANDWIDTH_PRB = CFG["channel_bandwidth_prb"]
TEMPERATURE = CFG["temperature"]

COVERAGE_THRESHOLD_DB = CFG["coverage_threshold_db"]
COORDINATE_ASCENT_MAX_ROUNDS = CFG["coordinate_ascent_max_rounds"]

NUM_GROUPS = CFG["num_groups"]
MEMBERS_PER_GROUP = NUM_UT // NUM_GROUPS
DEVIATION_RADIUS_FRAC_AREA = CFG["deviation_radius_frac_area"]
MEMBER_JITTER_SPEED = CFG["member_jitter_speed"]

MEASUREMENT_INTERVAL_S = CFG["measurement_interval_s"]
NUM_REALIZATIONS_PER_SLOT = CFG["num_realizations_per_slot"]
MAX_REALIZATION_CUDA = CFG["max_realization_cuda"]

CANDIDATE_WINDOW_SLOTS = CFG["candidate_window_slots"]
NUM_WINDOWS_PER_CANDIDATE = CFG["num_windows_per_candidate"]
MOBILITY_SPEED_SETTINGS = CFG["mobility_speed_settings"]

assert NUM_UT % NUM_GROUPS == 0, "NUM_UT must be divisible by NUM_GROUPS for equal-sized RPGM groups"
assert NUM_REALIZATIONS_PER_SLOT % MAX_REALIZATION_CUDA == 0, \
    "NUM_REALIZATIONS_PER_SLOT must be a multiple of MAX_REALIZATION_CUDA"

MEGA_CHUNK_SLOTS = max(CANDIDATE_WINDOW_SLOTS)
for w in CANDIDATE_WINDOW_SLOTS:
    assert MEGA_CHUNK_SLOTS % w == 0, \
        f"every candidate window ({w} slots) must evenly divide the largest ({MEGA_CHUNK_SLOTS} slots)"

_SCENARIO_CHANNEL_MODEL = {"umi": UMi, "uma": UMa, "rma": RMa}

NUM_SLOTS = MEGA_CHUNK_SLOTS * NUM_WINDOWS_PER_CANDIDATE


def configure_simulation():
    """Build one persistent topology, channel, and KPI path -- shared
    across every mobility speed setting swept below."""
    min_bs_ut_dist, isd, bs_height, min_ut_height, _, _, _, _ = set_3gpp_scenario_parameters(SCENARIO)
    scenario_params = {
        "isd": isd, "bs_height": bs_height,
        "min_bs_ut_dist": min_bs_ut_dist, "min_ut_height": min_ut_height,
    }
    topology = CellularTopology(scenario_params, num_rings=NUM_RINGS, batch_size=MAX_REALIZATION_CUDA)

    bs_array = AntennaArray(num_rows=M, num_cols=1, polarization="single", polarization_type="V",
                            antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY)
    ut_array = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                       carrier_frequency=CARRIER_FREQUENCY)
    sector_etilts = [
        ElectricalDowntilt(bs_array, carrier_frequency=CARRIER_FREQUENCY,
                           downtilt_deg=DOWNTILT_SWEEP_DEG[0], window=ANTENNA_WINDOW)
        for _ in range(topology.num_bs)
    ]

    channel_model_cls = _SCENARIO_CHANNEL_MODEL[SCENARIO]
    channel_model_kwargs = dict(
        carrier_frequency=CARRIER_FREQUENCY, bs_array=bs_array, ut_array=ut_array,
        direction="downlink", enable_pathloss=True, enable_shadow_fading=True,
    )
    if SCENARIO != "rma":
        channel_model_kwargs["o2i_model"] = "low"
    channel_model = channel_model_cls(**channel_model_kwargs)

    bs_tx_power_w = 10 ** ((BS_TX_POWER_DBM - 30.0) / 10.0)
    channel_bandwidth_hz = CHANNEL_BANDWIDTH_PRB * 12 * 30e3  # 30 kHz SCS
    noise_power_w = BOLTZMANN_CONSTANT * TEMPERATURE * channel_bandwidth_hz
    kpi_calculator = KpiCalculator(channel_model, sector_etilts, bs_tx_power_w, noise_power_w)

    ut_orientations = torch.zeros(MAX_REALIZATION_CUDA, NUM_UT, 3,
                                  dtype=topology.bs_loc.dtype, device=topology.bs_loc.device)
    ut_velocities = torch.zeros_like(ut_orientations)
    in_state = torch.zeros(MAX_REALIZATION_CUDA, NUM_UT, dtype=torch.bool, device=topology.bs_loc.device)

    return topology, channel_model, kpi_calculator, ut_orientations, ut_velocities, in_state


def build_mobility_list(topology, min_speed, max_speed):
    """NUM_REALIZATIONS_PER_SLOT independent RPGM populations -- one per
    pooled realization, matching test_dynamic_scenario_tilts_effect.py's
    pattern."""
    mobility_list = []
    for _ in range(NUM_REALIZATIONS_PER_SLOT):
        start_xy_list = sample_cluster_start_xy(topology, NUM_GROUPS)
        deviation_radius = DEVIATION_RADIUS_FRAC_AREA * topology.default_drop_radius
        initial_ut_loc, member_group_idx = sample_clustered_ut_loc(
            topology, start_xy_list, MEMBERS_PER_GROUP, deviation_radius, UT_HEIGHT,
            dtype=topology.bs_loc.dtype, device=topology.bs_loc.device,
        )
        mobility_list.append(ReferencePointGroupMobility(
            initial_ut_loc, member_group_idx, start_xy_list,
            deviation_radius=deviation_radius, topo=topology,
            min_speed=min_speed, max_speed=max_speed,
            member_jitter_speed=MEMBER_JITTER_SPEED,
        ))
    return mobility_list


def build_power_table_slot(topology, mobility_list, channel_model, kpi_calculator,
                           ut_orientations, ut_velocities, in_state):
    """One slot's pooled [num_tilts, num_sectors, NUM_REALIZATIONS_PER_SLOT*NUM_UT]
    power table, from every realization's CURRENT position."""
    num_chunks = NUM_REALIZATIONS_PER_SLOT // MAX_REALIZATION_CUDA
    chunks = []
    for chunk in range(num_chunks):
        chunk_mobility = mobility_list[chunk * MAX_REALIZATION_CUDA:(chunk + 1) * MAX_REALIZATION_CUDA]
        ut_loc = torch.stack([mobility.ut_loc for mobility in chunk_mobility], dim=0)
        bs_virtual_loc = topology.mirror_bs_loc(ut_loc)
        channel_model.set_topology(
            ut_loc, topology.bs_loc, ut_orientations, topology.bs_orientations,
            ut_velocities, in_state, None, bs_virtual_loc,
        )
        state = kpi_calculator.generate_large_scale_state()
        chunks.append(kpi_calculator.compute_power_table(DOWNTILT_SWEEP_DEG, state))
    return np.concatenate(chunks, axis=2)


def run_speed_setting(topology, channel_model, kpi_calculator, ut_orientations, ut_velocities, in_state,
                      min_speed, max_speed, num_bs):
    """Per-slot oracle coverage/tilt history, plus per-candidate windowed
    coverage/tilt history (broadcast across each window's own slots, so
    it's directly comparable slot-by-slot against the oracle)."""
    mobility_list = build_mobility_list(topology, min_speed, max_speed)

    oracle_coverage = np.zeros(NUM_SLOTS)
    oracle_tilt_deg_history = np.zeros((NUM_SLOTS, num_bs))
    windowed_coverage = {w: np.zeros(NUM_SLOTS) for w in CANDIDATE_WINDOW_SLOTS}
    windowed_tilt_deg_history = {w: np.zeros((NUM_SLOTS, num_bs)) for w in CANDIDATE_WINDOW_SLOTS}

    for chunk_start in range(0, NUM_SLOTS, MEGA_CHUNK_SLOTS):
        chunk_power_tables = []  # held only for this mega-chunk, then discarded
        for offset in range(MEGA_CHUNK_SLOTS):
            slot = chunk_start + offset
            power_table = build_power_table_slot(
                topology, mobility_list, channel_model, kpi_calculator,
                ut_orientations, ut_velocities, in_state,
            )
            chunk_power_tables.append(power_table)

            oracle_assignment = LocalTiltSelector(max_rounds=COORDINATE_ASCENT_MAX_ROUNDS).select(
                kpi_calculator, power_table, COVERAGE_THRESHOLD_DB, warm_start=None
            )
            oracle_coverage[slot] = kpi_calculator.coverage_from_assignment(
                power_table, oracle_assignment, COVERAGE_THRESHOLD_DB
            )
            oracle_tilt_deg_history[slot] = np.asarray(DOWNTILT_SWEEP_DEG)[oracle_assignment]

            if slot + 1 < NUM_SLOTS:
                for mobility in mobility_list:
                    mobility.step(MEASUREMENT_INTERVAL_S)

            if slot % 100 == 0 or slot == NUM_SLOTS - 1:
                print(f"  slot {slot:4d}/{NUM_SLOTS - 1}: oracle={oracle_coverage[slot]:.3f}")

        # This mega-chunk is complete: every candidate window evenly divides
        # it, so pool and score each candidate's windows that fall inside it.
        for w in CANDIDATE_WINDOW_SLOTS:
            for window_start in range(0, MEGA_CHUNK_SLOTS, w):
                window_tables = chunk_power_tables[window_start:window_start + w]
                pooled_table = np.concatenate(window_tables, axis=2)
                windowed_assignment = LocalTiltSelector(max_rounds=COORDINATE_ASCENT_MAX_ROUNDS).select(
                    kpi_calculator, pooled_table, COVERAGE_THRESHOLD_DB, warm_start=None
                )
                pooled_coverage = kpi_calculator.coverage_from_assignment(
                    pooled_table, windowed_assignment, COVERAGE_THRESHOLD_DB
                )
                tilt_deg = np.asarray(DOWNTILT_SWEEP_DEG)[windowed_assignment]
                abs_start = chunk_start + window_start
                windowed_coverage[w][abs_start:abs_start + w] = pooled_coverage
                windowed_tilt_deg_history[w][abs_start:abs_start + w] = tilt_deg

    return oracle_coverage, oracle_tilt_deg_history, windowed_coverage, windowed_tilt_deg_history


def main():
    topology, channel_model, kpi_calculator, ut_orientations, ut_velocities, in_state = configure_simulation()
    num_bs = topology.num_bs
    print(f"Device={DEVICE}; {NUM_UT} UEs, {num_bs} sectors; {NUM_SLOTS} slots x {MEASUREMENT_INTERVAL_S:g} s; "
         f"candidate windows (slots)={CANDIDATE_WINDOW_SLOTS}")

    save_kwargs = dict(
        candidate_window_slots=np.asarray(CANDIDATE_WINDOW_SLOTS),
        measurement_interval_s=MEASUREMENT_INTERVAL_S,
        num_slots=NUM_SLOTS,
        num_bs=num_bs,
        downtilt_sweep_deg=np.asarray(DOWNTILT_SWEEP_DEG),
        coverage_threshold_db=COVERAGE_THRESHOLD_DB,
        mobility_speed_names=np.asarray([s["name"] for s in MOBILITY_SPEED_SETTINGS]),
        mobility_speed_min=np.asarray([s["min_ut_speed"] for s in MOBILITY_SPEED_SETTINGS]),
        mobility_speed_max=np.asarray([s["max_ut_speed"] for s in MOBILITY_SPEED_SETTINGS]),
    )
    for setting in MOBILITY_SPEED_SETTINGS:
        name = setting["name"]
        print(f"\n=== mobility speed setting: {name} "
             f"({setting['min_ut_speed']:g}-{setting['max_ut_speed']:g} m/s) ===")
        oracle_coverage, oracle_tilt_deg_history, windowed_coverage, windowed_tilt_deg_history = run_speed_setting(
            topology, channel_model, kpi_calculator, ut_orientations, ut_velocities, in_state,
            setting["min_ut_speed"], setting["max_ut_speed"], num_bs,
        )

        # Sanity check, not just an expectation: at W=1 the windowed search
        # IS the per-slot oracle (same data, same warm_start=None), so the
        # gap must be exactly zero.
        gap_at_w1 = np.abs(oracle_coverage - windowed_coverage[1]).max()
        assert gap_at_w1 < 1e-9, f"W=1 must equal the per-slot oracle exactly; max |gap|={gap_at_w1}"

        mean_gap_msg = ", ".join(
            f"W={w}: mean_gap={np.mean(oracle_coverage - windowed_coverage[w]):.4f}"
            for w in CANDIDATE_WINDOW_SLOTS
        )
        print(f"{name}: {mean_gap_msg}")

        save_kwargs[f"oracle_coverage_{name}"] = oracle_coverage
        save_kwargs[f"oracle_tilt_deg_history_{name}"] = oracle_tilt_deg_history
        for w in CANDIDATE_WINDOW_SLOTS:
            save_kwargs[f"windowed_coverage_{name}_{w}"] = windowed_coverage[w]
            save_kwargs[f"windowed_tilt_deg_history_{name}_{w}"] = windowed_tilt_deg_history[w]

    data_path = os.path.join(OUT_DIR, "data.npz")
    np.savez(data_path, **save_kwargs)
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
