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

policy_name defaults to whatever config.yaml has (independent-dqn) --
pass "random" to run the no-learning baseline instead.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import sionna
import torch

from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine

sionna.phy.config.precision = "single"
sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

TAG, STATE_TYPE, REWARD_TYPE, OPTIMIZATION_ENABLED = sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4]))
POLICY_NAME = sys.argv[5] if len(sys.argv) > 5 else None

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
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[{TAG}] policy_name={DRL_CFG['policy_name']} state_type={STATE_TYPE} reward_type={REWARD_TYPE} "
         f"optimization_enabled={OPTIMIZATION_ENABLED} intervals={SIMULATION_CFG['num_tilt_control_intervals']}")

    engine = SimulationEngine(CFG)
    results = engine.run_simulation()

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

    steady_state_episodes = SIMULATION_CFG["steady_state_episodes"]
    steady = slice(-steady_state_episodes, None)
    print(
        f"[{TAG}] Mean coverage (full run): adaptive_legacy={coverage_adaptive_legacy.mean():.4f}, "
        f"no_tilt={coverage_no_tilt.mean():.4f}, drl={coverage_drl.mean():.4f}"
    )
    print(
        f"[{TAG}] Mean coverage (last {steady_state_episodes} intervals -- steady state): "
        f"adaptive_legacy={coverage_adaptive_legacy[steady].mean():.4f}, "
        f"no_tilt={coverage_no_tilt[steady].mean():.4f}, drl={coverage_drl[steady].mean():.4f}"
    )

    drl_policy.save(Path(OUT_DIR) / "drl_policy.pt")

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
        mobility_model=MOBILITY_CFG["mobility_model"],
        num_bs=engine.num_bs,
    )
    print(f"[{TAG}] Saved: {data_path}")


if __name__ == "__main__":
    main()
