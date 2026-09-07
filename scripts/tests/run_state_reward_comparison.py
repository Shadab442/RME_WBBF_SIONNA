"""One run of the DRL state/reward taxonomy comparison -- same engine and
per-interval loop as main.py, parameterized by state_type/reward_type/
optimization_enabled/tag so 8 combinations can run as separate processes in
parallel. See scripts/plots/plot_state_reward_comparison.py for the
combined comparison plots once all runs finish.

Oracle/Causal (coordinate-ascent search) is identical across every
combination here -- it doesn't depend on state_type/reward_type at all --
so it's only computed when optimization_enabled=1 (pass this for exactly
one of the 8 runs; the others reuse it at plotting time). Adaptive Legacy/
No Tilt are likewise identical across runs (DRL doesn't feed back into
mobility/channel), so any one run's copy works for comparison.

Run: python scripts/tests/run_state_reward_comparison.py <tag> <state_type> <reward_type> <optimization_enabled 0|1> [policy_name]

policy_name defaults to whatever config.yaml has (dqn) --
pass "random" to run the no-learning baseline instead.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import sionna
import torch

from helpers.utils import LiveMetricsPlot, RepoLogging, get_logger, load_config
from helpers.simulation_engine import SimulationEngine

sionna.phy.config.precision = "single"
sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

_parser = argparse.ArgumentParser()
_parser.add_argument("tag")
_parser.add_argument("state_type")
_parser.add_argument("reward_type")
_parser.add_argument("optimization_enabled", type=int)
_parser.add_argument("policy_name", nargs="?", default=None)
RepoLogging.add_argument(_parser)
_args = _parser.parse_args()

TAG, STATE_TYPE, REWARD_TYPE, OPTIMIZATION_ENABLED = _args.tag, _args.state_type, _args.reward_type, bool(_args.optimization_enabled)
POLICY_NAME = _args.policy_name

RepoLogging.configure(_args.log_level, overrides=RepoLogging.parse_overrides(_args.log_level_override))
logger = get_logger(__name__)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_reward_comparison", TAG)

CFG = load_config()
CFG["algorithms"]["drl"]["state_type"] = STATE_TYPE
CFG["algorithms"]["drl"]["reward_type"] = REWARD_TYPE
CFG["algorithms"]["optimization"]["enabled"] = OPTIMIZATION_ENABLED
if POLICY_NAME is not None:
    CFG["algorithms"]["drl"]["policy_name"] = POLICY_NAME
SIMULATION_CFG = CFG["simulation"]
ANTENNA_CFG = CFG["antenna"]
KPI_CFG = CFG["kpi"]
MOBILITY_CFG = CFG["mobility"]
DRL_CFG = CFG["algorithms"]["drl"]

sionna.phy.config.seed = SIMULATION_CFG["environment_seed"]


def main():
    logger.function(f"main start: tag={TAG}")
    os.makedirs(OUT_DIR, exist_ok=True)
    logger.info(f"[{TAG}] policy_name={DRL_CFG['policy_name']} state_type={STATE_TYPE} reward_type={REWARD_TYPE} "
               f"optimization_enabled={OPTIMIZATION_ENABLED} intervals={SIMULATION_CFG['num_tilt_control_intervals']}")

    engine = SimulationEngine(CFG)
    logger.debug(f"[{TAG}] engine constructed, num_bs={engine.num_bs}")
    live_plot = LiveMetricsPlot(OUT_DIR)
    results = engine.run_simulation(live_plot=live_plot)
    live_plot.close()
    logger.debug(f"[{TAG}] run_simulation done, live_plot closed")

    coverage_dynamic_local_oracle = results["coverage_dynamic_local_oracle"]
    coverage_dynamic_local_causal = results["coverage_dynamic_local_causal"]
    coverage_adaptive_legacy = results["coverage_adaptive_legacy"]
    coverage_no_tilt = results["coverage_no_tilt"]
    coverage_drl = results["coverage_drl"]
    drl_reward_history = results["drl_reward_history"]
    drl_overshoot_history = results["drl_overshoot_history"]
    drl_loss_per_interval = results["drl_loss_per_interval"]
    drl_tilt_deg_history = results["drl_tilt_deg_history"]
    drl_policy = results["drl_policy"]
    logger.debug(f"[{TAG}] results dict unpacked")

    steady_state_episodes = SIMULATION_CFG["steady_state_episodes"]
    steady = slice(-steady_state_episodes, None)
    logger.info(
        f"[{TAG}] Mean coverage (full run): adaptive_legacy={coverage_adaptive_legacy.mean():.4f}, "
        f"no_tilt={coverage_no_tilt.mean():.4f}, drl={coverage_drl.mean():.4f}"
    )
    logger.info(
        f"[{TAG}] Mean coverage (last {steady_state_episodes} intervals -- steady state): "
        f"adaptive_legacy={coverage_adaptive_legacy[steady].mean():.4f}, "
        f"no_tilt={coverage_no_tilt[steady].mean():.4f}, drl={coverage_drl[steady].mean():.4f}"
    )

    drl_policy.save(Path(OUT_DIR) / "drl_policy.pt")
    logger.info(f"[{TAG}] Saved DRL policy: {Path(OUT_DIR) / 'drl_policy.pt'}")

    data_path = os.path.join(OUT_DIR, "data.npz")
    np.savez(
        data_path,
        coverage_dynamic_local_oracle=coverage_dynamic_local_oracle,
        coverage_dynamic_local_causal=coverage_dynamic_local_causal,
        coverage_adaptive_legacy=coverage_adaptive_legacy,
        coverage_no_tilt=coverage_no_tilt,
        coverage_drl=coverage_drl,
        drl_tilt_deg_history=drl_tilt_deg_history,
        drl_reward_history=drl_reward_history,
        drl_overshoot_history=drl_overshoot_history,
        drl_loss_per_interval=drl_loss_per_interval,
        downtilt_sweep_deg=np.asarray(ANTENNA_CFG["downtilt_sweep_deg"]),
        coverage_threshold_db=KPI_CFG["coverage_threshold_db"],
        num_tilt_control_intervals=SIMULATION_CFG["num_tilt_control_intervals"],
        tilt_control_interval_s=SIMULATION_CFG["tilt_control_interval_s"],
        measurement_interval_s=SIMULATION_CFG["measurement_interval_s"],
        steady_state_episodes=steady_state_episodes,
        state_type=STATE_TYPE,
        reward_type=REWARD_TYPE,
        optimization_enabled=OPTIMIZATION_ENABLED,
        policy_name=DRL_CFG["policy_name"],
        mobility_model=MOBILITY_CFG["cluster_mobility_mode"],
        num_bs=engine.num_bs,
    )
    logger.info(f"[{TAG}] Saved: {data_path}")
    logger.function(f"main end: tag={TAG}")


if __name__ == "__main__":
    main()
