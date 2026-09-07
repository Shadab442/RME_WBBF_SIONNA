"""
Measurements and mobility update every 1 s;
tilt updates every 300 s using pooled measurements.
350 control intervals = 105,000 measurement samples (see config.yaml's
`simulation` section for the current values).

Dynamic Local Oracle: Per-interval coordinate-ascent tilt optimization,
warm-started from the previous solution, with decisions and scoring performed
on the same interval’s pooled measurements; non-causal upper bound.

Dynamic Local Causal: Scores the previous interval’s tilt decision on the current
pooled measurements, then optimizes the next tilt assignment from the current interval
for use in the next interval.

Adaptive Legacy: Reactive rule-based per-sector RET controller, scored at the current tilt
and updated once per interval using measurements pooled over that interval.

No Tilt: Fixed 0∘ boresight tilt for all sectors throughout the simulation; no adaptation.

DRL: Independent per-sector Double-DQN controllers, scored at the current tilt and updated
from realization-0 pooled measurements to select tilts for the next interval.

Run: python main.py
"""

import argparse
import os
from pathlib import Path

import numpy as np
import sionna
import torch

from helpers.utils import LiveMetricsPlot, RepoLogging, get_logger, load_config, save_mobility_animation
from helpers.simulation_engine import SimulationEngine

# Named WESN-readout x mobility experiment presets -- each is DRL-only
# (optimization.enabled=False skips the expensive Oracle/Causal
# coordinate-ascent sweep; Adaptive Legacy and No Tilt are still cheaply
# computed by SimulationEngine regardless, since it doesn't currently
# support skipping them, but neither adds meaningful compute or affects
# DRL's own training -- see helpers/simulation_engine.py's execute_algorithms).
_EXPERIMENTS = {
    "wesn-no-mlp-default": {"mlp_hidden_sizes": []},
    "wesn-early-default": {},  # config.yaml's own mlp_hidden_sizes/mlp_position as-is
    # "-transition-random": cluster_mobility_mode stays "periodic" (config.yaml's
    # own value) -- only transition changes, jump -> smooth.
    "wesn-no-mlp-transition-random": {"mlp_hidden_sizes": [], "transition": "smooth"},
    "wesn-early-transition-random": {"transition": "smooth"},
}

# Logging (before anything else logs)
_parser = argparse.ArgumentParser()
_parser.add_argument("--model", choices=["mlp", "wesn", "random"], default=None,
                     help="Override the DRL policy from config.yaml -- 'mlp'/'wesn' select "
                          "algorithms.drl.dqn.model (policy stays dqn); 'random' "
                          "switches algorithms.drl.policy_name to the untrained RandomPolicy "
                          "baseline instead (default choice for the trained/untrained comparison). "
                          "Each gets its own OUT_DIR so mlp/wesn/random can run concurrently "
                          "without editing the shared config file.")
_parser.add_argument("--experiment", choices=list(_EXPERIMENTS.keys()), default=None,
                     help="Run one of the named WESN-readout x mobility presets instead of "
                          "plain config.yaml -- overrides model to wesn, applies that preset's "
                          "mlp_hidden_sizes/cluster_mobility_mode, and disables optimization "
                          "(Oracle/Causal). Mutually exclusive with --model. Own OUT_DIR per name.")
RepoLogging.add_argument(_parser)
_args = _parser.parse_args()
assert _args.model is None or _args.experiment is None, "--model and --experiment are mutually exclusive"
RepoLogging.configure(_args.log_level, overrides=RepoLogging.parse_overrides(_args.log_level_override))
logger = get_logger(__name__)

sionna.phy.config.precision = "single"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
sionna.phy.config.device = DEVICE

# Load config
CFG = load_config()
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
elif _args.model is not None:
    DRL_CFG["dqn"]["model"] = _args.model
elif _args.experiment is not None:
    preset = _EXPERIMENTS[_args.experiment]
    DRL_CFG["dqn"]["model"] = "wesn"
    if "mlp_hidden_sizes" in preset:
        DRL_CFG["dqn"]["wesn"]["mlp_hidden_sizes"] = preset["mlp_hidden_sizes"]
    if "transition" in preset:
        MOBILITY_CFG["transition"] = preset["transition"]
    CFG["algorithms"]["optimization"]["enabled"] = False

if _args.experiment is not None:
    _out_name = f"experiment_{_args.experiment}"
elif _args.model is not None:
    _out_name = f"main_{_args.model}"
else:
    _out_name = "main"
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
    measurement_slots_per_interval = round(
        SIMULATION_CFG["tilt_control_interval_s"] / SIMULATION_CFG["measurement_interval_s"])
    logger.info(
        f"History={SIMULATION_CFG['num_tilt_control_intervals']} tilt control intervals x "
        f"{SIMULATION_CFG['tilt_control_interval_s']:g} s "
        f"({measurement_slots_per_interval} x {SIMULATION_CFG['measurement_interval_s']:g} s measurement draws/interval) x "
        f"{SIMULATION_CFG['num_realizations_per_slot']} realizations/draw; "
        f"mobility={MOBILITY_CFG['cluster_mobility_mode']}; speed={MOBILITY_CFG['min_ut_speed']:g}-{MOBILITY_CFG['max_ut_speed']:g} m/s"
    )
    logger.info(f"RPGM={MOBILITY_CFG['num_groups']} groups x "
               f"{TOPOLOGY_CFG['num_ut'] // MOBILITY_CFG['num_groups']} UEs")

    # Configure environment
    engine = SimulationEngine(CFG)
    logger.debug(f"main: engine constructed, num_bs={engine.num_bs}")

    # Implements and evaluate tilt control algorithms
    live_plot = LiveMetricsPlot(OUT_DIR)
    checkpoint_path = os.path.join(OUT_DIR, "data.npz")
    results = engine.run_simulation(live_plot=live_plot, checkpoint_path=checkpoint_path)
    live_plot.close()
    logger.debug("main: run_simulation done, live_plot closed")

    # Extract results
    coverage_dynamic_local_oracle = results["coverage_dynamic_local_oracle"]
    coverage_dynamic_local_causal = results["coverage_dynamic_local_causal"]
    coverage_adaptive_legacy = results["coverage_adaptive_legacy"]
    coverage_no_tilt = results["coverage_no_tilt"]
    coverage_drl = results["coverage_drl"]
    sinr_percentiles_dynamic_local_oracle = results["sinr_percentiles_dynamic_local_oracle"]
    sinr_percentiles_dynamic_local_causal = results["sinr_percentiles_dynamic_local_causal"]
    sinr_percentiles_adaptive_legacy = results["sinr_percentiles_adaptive_legacy"]
    sinr_percentiles_no_tilt = results["sinr_percentiles_no_tilt"]
    sinr_percentiles_drl = results["sinr_percentiles_drl"]
    dynamic_local_oracle_tilt_deg_history = results["dynamic_local_oracle_tilt_deg_history"]
    dynamic_local_causal_tilt_deg_history = results["dynamic_local_causal_tilt_deg_history"]
    adaptive_legacy_tilt_deg_history = results["adaptive_legacy_tilt_deg_history"]
    drl_tilt_deg_history = results["drl_tilt_deg_history"]
    drl_reward_history = results["drl_reward_history"]
    drl_overshoot_history = results["drl_overshoot_history"]
    drl_loss_per_interval = results["drl_loss_per_interval"]
    no_tilt_deg = results["no_tilt_deg"]
    position_history = results["position_history"]
    ref_xy_history = results["ref_xy_history"]
    drl_policy = results["drl_policy"]
    logger.debug("main: results dict unpacked")

    # Save UE mobility animation for first realization
    topology = engine.topology
    deviation_radius = MOBILITY_CFG["deviation_radius_frac_area"] * topology.default_drop_radius
    member_group_idx = getattr(engine.mobility_list[0], "member_group_idx", None)
    save_mobility_animation(CFG, OUT_DIR, topology, position_history, ref_xy_history, deviation_radius,
                            member_group_idx)

    # Print evaluation results (time evolution)
    causal_field = (f"dynamic_local_causal={coverage_dynamic_local_causal.mean():.4f}, "
                   if engine.run_causal else "")
    logger.info(
        f"Mean coverage (full run): dynamic_local_oracle={coverage_dynamic_local_oracle.mean():.4f}, "
        f"{causal_field}"
        f"adaptive_legacy={coverage_adaptive_legacy.mean():.4f}, "
        f"no_tilt={coverage_no_tilt.mean():.4f}, "
        f"drl={coverage_drl.mean():.4f}"
    )
    oracle_changes_per_sector = np.count_nonzero(np.diff(dynamic_local_oracle_tilt_deg_history, axis=0), axis=0)
    adaptive_legacy_changes_per_sector = np.count_nonzero(np.diff(adaptive_legacy_tilt_deg_history, axis=0), axis=0)
    drl_changes_per_sector = np.count_nonzero(np.diff(drl_tilt_deg_history, axis=0), axis=0)
    num_transitions = SIMULATION_CFG["num_tilt_control_intervals"] - 1
    logger.debug(f"main: tilt-change counts computed over {num_transitions} transitions")
    logger.info(
        f"Dynamic Local Oracle tilt changes per sector: min={oracle_changes_per_sector.min()}, "
        f"max={oracle_changes_per_sector.max()}, mean={oracle_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )
    if engine.run_causal:
        causal_changes_per_sector = np.count_nonzero(np.diff(dynamic_local_causal_tilt_deg_history, axis=0), axis=0)
        logger.info(
            f"Dynamic Local Causal tilt changes per sector: min={causal_changes_per_sector.min()}, "
            f"max={causal_changes_per_sector.max()}, mean={causal_changes_per_sector.mean():.1f} "
            f"(over {num_transitions} interval transitions)"
        )
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
    steady_causal_field = (f"dynamic_local_causal={coverage_dynamic_local_causal[steady].mean():.4f}, "
                          if engine.run_causal else "")
    logger.info(f"Mean coverage (last {steady_state_episodes} intervals -- steady state): "
               f"dynamic_local_oracle={coverage_dynamic_local_oracle[steady].mean():.4f}, "
               f"{steady_causal_field}"
               f"adaptive_legacy={coverage_adaptive_legacy[steady].mean():.4f}, "
               f"no_tilt={coverage_no_tilt[steady].mean():.4f}, "
               f"drl={coverage_drl[steady].mean():.4f}")

    # Save DRL policy
    drl_policy.save(Path(OUT_DIR) / "drl_policy.pt")
    logger.info(f"Saved DRL policy: {Path(OUT_DIR) / 'drl_policy.pt'}")

    # Save evaluation results to results/tests/dynamic_scenario_tilts_effect/
    # -- same path run_simulation's own checkpoint_path was overwriting
    # throughout the run; this final save just adds metadata on top.
    np.savez(
        checkpoint_path,
        coverage_dynamic_local_oracle=coverage_dynamic_local_oracle,
        coverage_dynamic_local_causal=coverage_dynamic_local_causal,
        coverage_adaptive_legacy=coverage_adaptive_legacy,
        coverage_no_tilt=coverage_no_tilt,
        coverage_drl=coverage_drl,
        sinr_percentiles_dynamic_local_oracle=sinr_percentiles_dynamic_local_oracle,
        sinr_percentiles_dynamic_local_causal=sinr_percentiles_dynamic_local_causal,
        sinr_percentiles_adaptive_legacy=sinr_percentiles_adaptive_legacy,
        sinr_percentiles_no_tilt=sinr_percentiles_no_tilt,
        sinr_percentiles_drl=sinr_percentiles_drl,
        sinr_percentiles=np.asarray(KPI_CFG["sinr_percentiles"]),
        dynamic_local_oracle_tilt_deg_history=dynamic_local_oracle_tilt_deg_history,
        dynamic_local_causal_tilt_deg_history=dynamic_local_causal_tilt_deg_history,
        adaptive_legacy_tilt_deg_history=adaptive_legacy_tilt_deg_history,
        drl_tilt_deg_history=drl_tilt_deg_history,
        drl_reward_history=drl_reward_history,
        drl_overshoot_history=drl_overshoot_history,
        drl_loss_per_interval=drl_loss_per_interval,
        no_tilt_deg=no_tilt_deg,
        downtilt_sweep_deg=DOWNTILT_SWEEP_DEG,
        coverage_threshold_db=KPI_CFG["coverage_threshold_db"],
        num_tilt_control_intervals=SIMULATION_CFG["num_tilt_control_intervals"],
        tilt_control_interval_s=SIMULATION_CFG["tilt_control_interval_s"],
        measurement_interval_s=SIMULATION_CFG["measurement_interval_s"],
        steady_state_episodes=steady_state_episodes,
        policy_name=DRL_CFG["policy_name"],
        mobility_model=MOBILITY_CFG["cluster_mobility_mode"],
        num_bs=topology.num_bs,
    )
    logger.info(f"Saved: {checkpoint_path}")
    logger.function("main end")


if __name__ == "__main__":
    main()
