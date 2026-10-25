"""
One measurement (mobility position + channel draw) per tilt-control
interval (see config.yaml's `simulation` section).

Adaptive Legacy: Reactive rule-based per-sector RET controller, scored at the current tilt
and updated once per interval using that interval's own measurement.

DRL: Independent per-sector controllers, scored at the current tilt and
updated from that interval's own measurement to select tilts for the next
interval -- either Double-DQN (discrete tilt_step_deg-spaced actions),
RandomPolicy (untrained baseline, --model random), or PPO (continuous
action, floor-quantized to algorithms.drl.ppo.tilt_quantization_deg --
--model ppo, see drl/ppo.py).

Exactly these two algorithms exist as of 2026-09-22 -- Dynamic Local Causal,
No Tilt, and both the old coordinate-ascent Oracle and its CP-SAT
replacement (helpers/tilt_controller.py's own CpSatTiltController -- built,
then deleted the same day: it wasn't earning its own real-world cost
relative to the baseline pipeline) are all deleted.

Run: python main.py
"""

import argparse
import os
from pathlib import Path

import numpy as np
import sionna
import torch

from helpers.utils import LiveMetricsPlot, RepoLogging, get_logger, load_config, save_mobility_animation
import helpers.simulation_engine as _se_mod
from helpers.simulation_engine import SimulationEngine

# Per-scenario overrides for geometry knobs tuned for RMa's isd=5000m --
# UMa's own near/mid hotspot crossover (174.2m) was empirically calibrated
# via scripts/radial_swap_plot_scenario.py (seed-averaged distance-binned
# coverage swap, 2026-10-05); the naive isd-ratio rescale (90m) was wrong
# because the crossover is governed by antenna geometry (mechanical pitch),
# which doesn't scale with isd the way cell_radius does. far_dist_m,
# spatial_grid_cell_size_m, deviation_radius_m, cluster_radius_max_m DO use
# the isd-ratio rescale (0.1x RMa's own values) -- those scale with
# cell_radius, which is exactly isd/sqrt(3) for every 3GPP scenario.
# downtilt sweep ranges are each scenario's own empirically-found best
# tradeoff window (RMa -1..+1/0.1 deg committed; UMa -2..+2/0.2 deg, coarser
# step to keep the DRL action-grid size comparable to RMa's 21 actions).
_SCENARIO_OVERRIDES = {
    "uma": {
        "near_dist_m": 174.2, "far_dist_m": 230.0,
        "spatial_grid_cell_size_m": 5.0, "deviation_radius_m": 4.0, "cluster_radius_max_m": 10.0,
        "downtilt_sweep_start_deg": -2.0, "downtilt_sweep_end_deg": 2.0, "tilt_step_deg": 0.2,
    },
}

# Named WESN-readout x mobility experiment presets -- each is DRL-only.
# Adaptive Legacy is still cheaply computed by SimulationEngine regardless,
# since it doesn't currently support skipping it, but it doesn't add
# meaningful compute or affect DRL's own training -- see
# helpers/simulation_engine.py's execute_algorithms).
_EXPERIMENTS = {
    "wesn-no-mlp-default": {"mlp_hidden_sizes": []},
    "wesn-early-default": {},  # config.yaml's own mlp_hidden_sizes/mlp_position as-is
}

# Logging (before anything else logs)
_parser = argparse.ArgumentParser()
_parser.add_argument("--model", choices=["mlp", "wesn", "random", "ppo"], default=None,
                     help="Override the DRL policy from config.yaml -- 'mlp'/'wesn' select "
                          "algorithms.drl.dqn.model (policy stays dqn); 'random' "
                          "switches algorithms.drl.policy_name to the untrained RandomPolicy "
                          "baseline instead (default choice for the trained/untrained comparison); "
                          "'ppo' switches algorithms.drl.policy_name to the continuous-action PPO "
                          "policy (see drl/ppo.py, algorithms.drl.ppo in config.yaml). "
                          "Each gets its own OUT_DIR so mlp/wesn/random/ppo can run concurrently "
                          "without editing the shared config file.")
_parser.add_argument("--experiment", choices=list(_EXPERIMENTS.keys()), default=None,
                     help="Run one of the named WESN-readout x mobility presets instead of "
                          "plain config.yaml -- overrides model to wesn, applies that preset's "
                          "mlp_hidden_sizes/cluster_mobility_mode. Mutually exclusive with "
                          "--model. Own OUT_DIR per name.")
_parser.add_argument("--async-schedule", choices=["sync", "async"], default=None,
                     help="Override algorithms.drl.async_schedule from config.yaml -- same "
                          "per-run OUT_DIR pattern as --model, so e.g. --model mlp "
                          "--async-schedule sync and --model wesn --async-schedule async can run "
                          "concurrently without editing the shared config file.")
_parser.add_argument("--reward-type", choices=["global", "delta_covered"], default=None,
                     help="Override algorithms.drl.reward_type from config.yaml -- same "
                          "per-run OUT_DIR pattern as --model/--async-schedule, so a reward-type "
                          "comparison run doesn't need editing the shared config file.")
_parser.add_argument("--state-type",
                     choices=["region_aligned", "cell_aligned", "polar_whole_neighbor", "polar_region_neighbor"],
                     default=None,
                     help="Override algorithms.drl.state_type from config.yaml (2026-10-15 addition -- "
                          "polar_whole_neighbor/polar_region_neighbor are own-sector polar-grid (radial x "
                          "azimuthal) state representations, see helpers/simulation_engine.py's own "
                          "_calibrate_polar_lut). Same per-run OUT_DIR pattern as --model.")
_parser.add_argument("--outdoor-gnss-prob", type=float, default=None,
                     help="Override topology.outdoor_gnss_prob from config.yaml -- the fraction of "
                          "outdoor UEs that count as anchor (location-known) this run (2026-10-15 "
                          "addition, for an all-anchor vs. partial-anchor state-accuracy comparison "
                          "-- see config.yaml's own comment on the GNSS-based anchor mechanism). "
                          "Same per-run OUT_DIR pattern as --model.")
_parser.add_argument("--indoor-gnss-prob", type=float, default=None,
                     help="Override topology.indoor_gnss_prob from config.yaml -- the fraction of "
                          "INDOOR UEs that count as anchor this run (2026-10-23 addition). "
                          "config.yaml's own committed default (0.0) means indoor UEs are NEVER "
                          "anchor regardless of --outdoor-gnss-prob -- for a genuine uniform "
                          "'X% of EVERYONE is anchor' comparison (not just X% of the outdoor "
                          "slice), pass this equal to --outdoor-gnss-prob.")
_parser.add_argument("--non-anchor-region-handling", choices=["exclude", "rme", "radio_map"], default=None,
                     help="Override algorithms.drl.non_anchor_region_handling from config.yaml -- how "
                          "a non-anchor UE's own region is decided for own-region state (2026-10-23 "
                          "'radio_map' addition: SimulationEngine._build_region_idx_radio_map, see its "
                          "own docstring). Same per-run OUT_DIR pattern as --model.")
_parser.add_argument("--anchor-selection-mode", choices=["random", "balanced", "mid_prioritized_balanced"],
                     default=None,
                     help="Override topology.anchor_selection_mode from config.yaml -- 'random' "
                          "(default) draws anchor clusters uniformly; 'balanced' stratifies by "
                          "(sector, region-type) slot so every currently-occupied region stays "
                          "anchor-represented (2026-10-15 addition). Same per-run OUT_DIR pattern "
                          "as --model.")
_parser.add_argument("--cluster-mobility-mode", choices=["hotspot_demand-periodic", "hotspot_demand-markov"],
                     default=None,
                     help="Override mobility.common.cluster_mobility_mode from config.yaml -- "
                          "'-periodic' (default) advances each sector's demand state deterministically "
                          "(+1 mod num_states every hotspot_hold_steps intervals); '-markov' draws the "
                          "next state randomly from transition_matrix instead, making demand genuinely "
                          "unpredictable from history alone (2026-10-19 addition, for testing whether a "
                          "recurrent DQN can 'learn the clock' under -periodic and therefore not need "
                          "real-time own-region sensing -- see anchor-fraction sweep discussion). Same "
                          "per-run OUT_DIR pattern as --model.")
_parser.add_argument("--ppo-model", choices=["mlp", "wesn"], default=None,
                     help="Override algorithms.drl.ppo.model -- only meaningful together with "
                          "--model ppo (policy_name=ppo); --model itself only ever touches dqn.model. "
                          "Same per-run OUT_DIR pattern as --model.")
_parser.add_argument("--scenario", choices=["rma", "uma"], default=None,
                     help="Override topology.scenario from config.yaml, plus every geometry knob "
                          "that depends on it (hotspot near/mid/far cuts, spatial_grid_cell_size_m, "
                          "mobility deviation_radius_m/cluster_radius_max_m, downtilt sweep range) -- "
                          "see _SCENARIO_OVERRIDES. Same per-run OUT_DIR pattern as --model.")
_parser.add_argument("--tilt-step-deg", type=float, default=None,
                     help="Override downtilt.tilt_step_deg (applied AFTER --scenario's own preset, "
                          "if both given) -- coarsens/refines the action grid within the SAME sweep "
                          "range without changing downtilt_sweep_start_deg/end_deg. Same per-run "
                          "OUT_DIR pattern as --model.")
_parser.add_argument("--downtilt-start-deg", type=float, default=None,
                     help="Override downtilt.downtilt_sweep_start_deg (applied AFTER --scenario's "
                          "own preset). Same per-run OUT_DIR pattern as --model.")
_parser.add_argument("--downtilt-end-deg", type=float, default=None,
                     help="Override downtilt.downtilt_sweep_end_deg (applied AFTER --scenario's "
                          "own preset). Same per-run OUT_DIR pattern as --model.")
_parser.add_argument("--device", choices=["cuda:0", "cpu"], default=None,
                     help="Force the Sionna channel-simulation device, overriding the default "
                          "cuda:0-if-available auto-detect (2026-10-15 addition, for falling back "
                          "to CPU when the shared GPU is memory-saturated by other users' "
                          "processes). The per-sector DQN networks already run on CPU regardless "
                          "(see drl.factory) -- this only affects the channel/ray-tracing layer.")
_parser.add_argument("--cpu-threads", type=int, default=8,
                     help="torch.set_num_threads() cap (default 8) -- the per-sector DQN networks "
                          "run on CPU (device=cpu, see drl.factory), and PyTorch otherwise grabs "
                          "every visible core by default. On this machine (64 cores, shared with "
                          "other users) two concurrent runs left uncapped were measured fighting "
                          "each other for CPU and slowing ~30x versus running alone (2026-09-29/30) "
                          "-- capped, they coexist without starving each other or the machine's "
                          "other users.")
RepoLogging.add_argument(_parser)
_args = _parser.parse_args()
assert _args.model is None or _args.experiment is None, "--model and --experiment are mutually exclusive"
torch.set_num_threads(_args.cpu_threads)
RepoLogging.configure(_args.log_level, overrides=RepoLogging.parse_overrides(_args.log_level_override))
logger = get_logger(__name__)

sionna.phy.config.precision = "single"
DEVICE = _args.device if _args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
sionna.phy.config.device = DEVICE

# Load config
CFG = load_config()
if _args.scenario is not None:
    CFG["topology"]["scenario"] = _args.scenario
    if _args.scenario in _SCENARIO_OVERRIDES:
        ov = _SCENARIO_OVERRIDES[_args.scenario]
        CFG["downtilt"]["downtilt_sweep_start_deg"] = ov["downtilt_sweep_start_deg"]
        CFG["downtilt"]["downtilt_sweep_end_deg"] = ov["downtilt_sweep_end_deg"]
        CFG["downtilt"]["tilt_step_deg"] = ov["tilt_step_deg"]
        CFG["topology"]["spatial_grid_cell_size_m"] = ov["spatial_grid_cell_size_m"]
        CFG["mobility"]["hotspot"]["deviation_radius_m"] = ov["deviation_radius_m"]
        CFG["mobility"]["rpgm"]["cluster_radius_max_m"] = ov["cluster_radius_max_m"]
        # _calibrate_hotspot_region_lut (called inside SimulationEngine.
        # __init__, below) reads these module-level globals at call time.
        _se_mod._HOTSPOT_NEAR_DIST_M = ov["near_dist_m"]
        _se_mod._HOTSPOT_FAR_DIST_M = ov["far_dist_m"]
if _args.downtilt_start_deg is not None:
    CFG["downtilt"]["downtilt_sweep_start_deg"] = _args.downtilt_start_deg
if _args.downtilt_end_deg is not None:
    CFG["downtilt"]["downtilt_sweep_end_deg"] = _args.downtilt_end_deg
if _args.tilt_step_deg is not None:
    CFG["downtilt"]["tilt_step_deg"] = _args.tilt_step_deg
TOPOLOGY_CFG = CFG["topology"]
ANTENNA_CFG = CFG["antenna"]
DOWNTILT_CFG = CFG["downtilt"]
DOWNTILT_SWEEP_DEG = np.arange(
    DOWNTILT_CFG["downtilt_sweep_start_deg"],
    DOWNTILT_CFG["downtilt_sweep_end_deg"] + DOWNTILT_CFG["tilt_step_deg"],
    DOWNTILT_CFG["tilt_step_deg"],
)
KPI_CFG = CFG["kpi"]
MOBILITY_CFG = CFG["mobility"]
SIMULATION_CFG = CFG["simulation"]
DRL_CFG = CFG["algorithms"]["drl"]
if _args.model == "random":
    DRL_CFG["policy_name"] = "random"
elif _args.model == "ppo":
    DRL_CFG["policy_name"] = "ppo"
elif _args.model is not None:
    DRL_CFG["dqn"]["model"] = _args.model
elif _args.experiment is not None:
    preset = _EXPERIMENTS[_args.experiment]
    DRL_CFG["dqn"]["model"] = "wesn"
    if "mlp_hidden_sizes" in preset:
        DRL_CFG["dqn"]["wesn"]["mlp_hidden_sizes"] = preset["mlp_hidden_sizes"]
if _args.async_schedule is not None:
    DRL_CFG["async_schedule"] = _args.async_schedule
if _args.reward_type is not None:
    DRL_CFG["reward_type"] = _args.reward_type
if _args.state_type is not None:
    DRL_CFG["state_type"] = _args.state_type
if _args.ppo_model is not None:
    DRL_CFG["ppo"]["model"] = _args.ppo_model
if _args.outdoor_gnss_prob is not None:
    CFG["topology"]["outdoor_gnss_prob"] = _args.outdoor_gnss_prob
if _args.indoor_gnss_prob is not None:
    CFG["topology"]["indoor_gnss_prob"] = _args.indoor_gnss_prob
if _args.anchor_selection_mode is not None:
    CFG["topology"]["anchor_selection_mode"] = _args.anchor_selection_mode
if _args.non_anchor_region_handling is not None:
    DRL_CFG["non_anchor_region_handling"] = _args.non_anchor_region_handling
if _args.cluster_mobility_mode is not None:
    MOBILITY_CFG["common"]["cluster_mobility_mode"] = _args.cluster_mobility_mode

if _args.experiment is not None:
    _out_name = f"experiment_{_args.experiment}"
else:
    _out_name_parts = ["main"]
    if _args.scenario is not None:
        _out_name_parts.append(_args.scenario)
    if _args.downtilt_start_deg is not None or _args.downtilt_end_deg is not None:
        _out_name_parts.append(f"tilt{_args.downtilt_start_deg:g}to{_args.downtilt_end_deg:g}")
    if _args.tilt_step_deg is not None:
        _out_name_parts.append(f"step{_args.tilt_step_deg:g}")
    if _args.model is not None:
        _out_name_parts.append(_args.model)
    if _args.ppo_model is not None:
        _out_name_parts.append(_args.ppo_model)
    if _args.async_schedule is not None:
        _out_name_parts.append(_args.async_schedule)
    if _args.reward_type is not None:
        _out_name_parts.append(_args.reward_type)
    if _args.state_type is not None:
        _out_name_parts.append(_args.state_type)
    if _args.outdoor_gnss_prob is not None:
        _out_name_parts.append(f"gnss{_args.outdoor_gnss_prob:g}")
    if _args.indoor_gnss_prob is not None:
        _out_name_parts.append(f"ignss{_args.indoor_gnss_prob:g}")
    if _args.anchor_selection_mode is not None:
        _out_name_parts.append(_args.anchor_selection_mode)
    if _args.non_anchor_region_handling is not None:
        _out_name_parts.append(_args.non_anchor_region_handling)
    if _args.cluster_mobility_mode is not None:
        _out_name_parts.append(_args.cluster_mobility_mode.split("-", 1)[1])
    _out_name = "_".join(_out_name_parts)
OUT_DIR = os.path.join(os.path.dirname(__file__), "results", _out_name)

# ENVIRONMENT seed (topology/mobility/UE-drop/pathloss/shadow-fading)
sionna.phy.config.seed = SIMULATION_CFG["environment_seed"]


def main():
    logger.function("main start")

    # Print configs
    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info(
        f"Device={DEVICE}; scenario={TOPOLOGY_CFG['scenario']}; {TOPOLOGY_CFG['num_rings']} ring(s); "
        f"{TOPOLOGY_CFG['num_ut']} UEs; tilts={DOWNTILT_SWEEP_DEG.tolist()}; "
        f"threshold={KPI_CFG['coverage_threshold_db']:g} dB"
    )
    logger.info(
        f"History={SIMULATION_CFG['num_tilt_control_intervals']} tilt control intervals "
        f"(one measurement/interval); mobility={MOBILITY_CFG['common']['cluster_mobility_mode']}"
    )

    # Configure environment
    engine = SimulationEngine(CFG)
    logger.debug(f"main: engine constructed, num_bs={engine.num_bs}")
    # num_ues_per_cluster is DERIVED (a near-even per-cluster split of
    # num_ut, see SimulationEngine.__init__) -- not a config.yaml key, so
    # only readable post-construction; a list (one entry/cluster), not
    # necessarily uniform, so logged as its own min/max range.
    ues_lo, ues_hi = min(engine.num_ues_per_cluster), max(engine.num_ues_per_cluster)
    ues_desc = f"{ues_lo}" if ues_lo == ues_hi else f"{ues_lo}-{ues_hi}"
    logger.info(f"{MOBILITY_CFG['common']['num_clusters_per_sector']} clusters/sector x "
               f"{ues_desc} UEs/cluster")

    # Implements and evaluate tilt control algorithms
    live_plot = LiveMetricsPlot(OUT_DIR)
    checkpoint_path = os.path.join(OUT_DIR, "data.npz")
    results = engine.run_simulation(live_plot=live_plot, checkpoint_path=checkpoint_path)
    live_plot.close()
    logger.debug("main: run_simulation done, live_plot closed")

    if engine.diagnostics_logger is not None:
        diagnostics_out_dir = os.path.join(OUT_DIR, engine.diagnostics_out_subdir)
        engine.diagnostics_logger.flush(diagnostics_out_dir)
        logger.info(f"main: DRL diagnostics flushed to {diagnostics_out_dir}")

    # Extract results
    coverage_adaptive_legacy = results["coverage_adaptive_legacy"]
    coverage_drl = results["coverage_drl"]
    sinr_percentiles_adaptive_legacy = results["sinr_percentiles_adaptive_legacy"]
    sinr_percentiles_drl = results["sinr_percentiles_drl"]
    adaptive_legacy_tilt_deg_history = results["adaptive_legacy_tilt_deg_history"]
    drl_tilt_deg_history = results["drl_tilt_deg_history"]
    drl_reward_history = results["drl_reward_history"]
    drl_overshoot_history = results["drl_overshoot_history"]
    drl_loss_per_interval = results["drl_loss_per_interval"]
    position_history = results["position_history"]
    ref_xy_history = results["ref_xy_history"]
    drl_policy = results["drl_policy"]
    logger.debug("main: results dict unpacked")

    # Save UE mobility animation for first realization -- deviation_radius
    # read directly from the live mobility object (correct for whichever
    # mode is active: hotspot_demand's own absolute deviation_radius_m, or
    # RPGM's deviation_radius_frac_area-derived one), not recomputed from
    # config here (which would silently use the wrong value for
    # hotspot_demand -- see helpers/mobility.py's own self.deviation_radius).
    topology = engine.topology
    deviation_radius = engine.mobility.deviation_radius
    member_group_idx = getattr(engine.mobility, "member_group_idx", None)
    save_mobility_animation(CFG, OUT_DIR, topology, position_history, ref_xy_history, deviation_radius,
                            member_group_idx)

    # Print evaluation results (time evolution)
    logger.info(
        f"Mean coverage (full run): "
        f"adaptive_legacy={coverage_adaptive_legacy.mean():.4f}, "
        f"drl={coverage_drl.mean():.4f}"
    )
    adaptive_legacy_changes_per_sector = np.count_nonzero(np.diff(adaptive_legacy_tilt_deg_history, axis=0), axis=0)
    drl_changes_per_sector = np.count_nonzero(np.diff(drl_tilt_deg_history, axis=0), axis=0)
    num_transitions = SIMULATION_CFG["num_tilt_control_intervals"] - 1
    logger.debug(f"main: tilt-change counts computed over {num_transitions} transitions")
    logger.info(
        f"Adaptive Legacy tilt changes per sector: min={adaptive_legacy_changes_per_sector.min()}, "
        f"max={adaptive_legacy_changes_per_sector.max()}, mean={adaptive_legacy_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )
    logger.info(
        f"DRL tilt changes per sector: min={drl_changes_per_sector.min()}, "
        f"max={drl_changes_per_sector.max()}, mean={drl_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )

    # Print steady-state comparison results
    steady_state_episodes = SIMULATION_CFG["steady_state_episodes"]
    steady = slice(-steady_state_episodes, None)
    logger.info(f"Mean coverage (last {steady_state_episodes} intervals -- steady state): "
               f"adaptive_legacy={coverage_adaptive_legacy[steady].mean():.4f}, "
               f"drl={coverage_drl[steady].mean():.4f}")

    # Save DRL policy
    drl_policy.save(Path(OUT_DIR) / "drl_policy.pt")
    logger.info(f"Saved DRL policy: {Path(OUT_DIR) / 'drl_policy.pt'}")

    # Save evaluation results to results/tests/dynamic_scenario_tilts_effect/
    # -- same path run_simulation's own checkpoint_path was overwriting
    # throughout the run; this final save just adds metadata on top.
    np.savez(
        checkpoint_path,
        coverage_adaptive_legacy=coverage_adaptive_legacy,
        coverage_drl=coverage_drl,
        sinr_percentiles_adaptive_legacy=sinr_percentiles_adaptive_legacy,
        sinr_percentiles_drl=sinr_percentiles_drl,
        sinr_percentiles=np.asarray(KPI_CFG["sinr_percentiles"]),
        adaptive_legacy_tilt_deg_history=adaptive_legacy_tilt_deg_history,
        drl_tilt_deg_history=drl_tilt_deg_history,
        drl_reward_history=drl_reward_history,
        drl_overshoot_history=drl_overshoot_history,
        drl_loss_per_interval=drl_loss_per_interval,
        downtilt_sweep_deg=DOWNTILT_SWEEP_DEG,
        coverage_threshold_db=KPI_CFG["coverage_threshold_db"],
        num_tilt_control_intervals=SIMULATION_CFG["num_tilt_control_intervals"],
        steady_state_episodes=steady_state_episodes,
        policy_name=DRL_CFG["policy_name"],
        mobility_model=MOBILITY_CFG["common"]["cluster_mobility_mode"],
        num_bs=topology.num_bs,
    )
    logger.info(f"Saved: {checkpoint_path}")
    logger.function("main end")


if __name__ == "__main__":
    main()
