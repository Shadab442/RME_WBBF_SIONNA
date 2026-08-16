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

import os
from pathlib import Path

import numpy as np
import sionna
import torch

from helpers.utils import load_config, save_mobility_animation
from helpers.simulation_engine import SimulationEngine

sionna.phy.config.precision = "single"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cuda:0"
sionna.phy.config.device = DEVICE

OUT_DIR = os.path.join(os.path.dirname(__file__), "results", "main")

# Load config
CFG = load_config()
TOPOLOGY_CFG = CFG["topology"]
ANTENNA_CFG = CFG["antenna"]
KPI_CFG = CFG["kpi"]
MOBILITY_CFG = CFG["mobility"]
SIMULATION_CFG = CFG["simulation"]
DRL_CFG = CFG["algorithms"]["drl"]

# ENVIRONMENT seed (topology/mobility/UE-drop/pathloss/shadow-fading)
sionna.phy.config.seed = SIMULATION_CFG["environment_seed"]


def main():
    # Print configs
    os.makedirs(OUT_DIR, exist_ok=True)
    print(
        f"Device={DEVICE}; scenario={TOPOLOGY_CFG['scenario']}; {TOPOLOGY_CFG['num_rings']} ring(s); "
        f"{TOPOLOGY_CFG['num_ut']} UEs; tilts={ANTENNA_CFG['downtilt_sweep_deg']}; "
        f"threshold={KPI_CFG['coverage_threshold_db']:g} dB"
    )
    measurement_slots_per_interval = round(
        SIMULATION_CFG["tilt_control_interval_s"] / SIMULATION_CFG["measurement_interval_s"])
    print(
        f"History={SIMULATION_CFG['num_tilt_control_intervals']} tilt control intervals x "
        f"{SIMULATION_CFG['tilt_control_interval_s']:g} s "
        f"({measurement_slots_per_interval} x {SIMULATION_CFG['measurement_interval_s']:g} s measurement draws/interval) x "
        f"{SIMULATION_CFG['num_realizations_per_slot']} realizations/draw; "
        f"mobility={MOBILITY_CFG['mobility_model']}; speed={MOBILITY_CFG['min_ut_speed']:g}-{MOBILITY_CFG['max_ut_speed']:g} m/s"
    )
    if MOBILITY_CFG["mobility_model"] == "rpgm":
        print(f"RPGM={MOBILITY_CFG['num_groups']} groups x {TOPOLOGY_CFG['num_ut'] // MOBILITY_CFG['num_groups']} UEs")

    # Configure environment
    engine = SimulationEngine(CFG)

    # Implements and evaluate tilt control algorithms
    results = engine.run_simulation()

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

    # Save UE mobility animation for first realization
    topology = engine.topology
    deviation_radius = MOBILITY_CFG["deviation_radius_frac_area"] * topology.default_drop_radius
    member_group_idx = getattr(engine.mobility_list[0], "member_group_idx", None)
    save_mobility_animation(CFG, OUT_DIR, topology, position_history, ref_xy_history, deviation_radius,
                            member_group_idx)

    # Print evaluation results (time evolution)
    print(
        f"Mean coverage (full run): dynamic_local_oracle={coverage_dynamic_local_oracle.mean():.4f}, "
        f"dynamic_local_causal={coverage_dynamic_local_causal.mean():.4f}, "
        f"adaptive_legacy={coverage_adaptive_legacy.mean():.4f}, "
        f"no_tilt={coverage_no_tilt.mean():.4f}, "
        f"drl={coverage_drl.mean():.4f}"
    )
    oracle_changes_per_sector = np.count_nonzero(np.diff(dynamic_local_oracle_tilt_deg_history, axis=0), axis=0)
    causal_changes_per_sector = np.count_nonzero(np.diff(dynamic_local_causal_tilt_deg_history, axis=0), axis=0)
    adaptive_legacy_changes_per_sector = np.count_nonzero(np.diff(adaptive_legacy_tilt_deg_history, axis=0), axis=0)
    drl_changes_per_sector = np.count_nonzero(np.diff(drl_tilt_deg_history, axis=0), axis=0)
    num_transitions = SIMULATION_CFG["num_tilt_control_intervals"] - 1
    print(
        f"Dynamic Local Oracle tilt changes per sector: min={oracle_changes_per_sector.min()}, "
        f"max={oracle_changes_per_sector.max()}, mean={oracle_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )
    print(
        f"Dynamic Local Causal tilt changes per sector: min={causal_changes_per_sector.min()}, "
        f"max={causal_changes_per_sector.max()}, mean={causal_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )
    print(
        f"Adaptive Legacy tilt changes per sector: min={adaptive_legacy_changes_per_sector.min()}, "
        f"max={adaptive_legacy_changes_per_sector.max()}, mean={adaptive_legacy_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )
    print(
        f"DRL tilt changes per sector: min={drl_changes_per_sector.min()}, "
        f"max={drl_changes_per_sector.max()}, mean={drl_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )

    # Print steady-state comparison results
    steady_state_episodes = SIMULATION_CFG["steady_state_episodes"]
    steady = slice(-steady_state_episodes, None)
    print(f"Mean coverage (last {steady_state_episodes} intervals -- steady state): "
         f"dynamic_local_oracle={coverage_dynamic_local_oracle[steady].mean():.4f}, "
         f"dynamic_local_causal={coverage_dynamic_local_causal[steady].mean():.4f}, "
         f"adaptive_legacy={coverage_adaptive_legacy[steady].mean():.4f}, "
         f"no_tilt={coverage_no_tilt[steady].mean():.4f}, "
         f"drl={coverage_drl[steady].mean():.4f}")

    # Save DRL policy
    drl_policy.save(Path(OUT_DIR) / "drl_policy.pt")

    # Save evaluation results to results/tests/dynamic_scenario_tilts_effect/
    data_path = os.path.join(OUT_DIR, "data.npz")
    np.savez(
        data_path,
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
        downtilt_sweep_deg=np.asarray(ANTENNA_CFG["downtilt_sweep_deg"]),
        coverage_threshold_db=KPI_CFG["coverage_threshold_db"],
        num_tilt_control_intervals=SIMULATION_CFG["num_tilt_control_intervals"],
        tilt_control_interval_s=SIMULATION_CFG["tilt_control_interval_s"],
        measurement_interval_s=SIMULATION_CFG["measurement_interval_s"],
        steady_state_episodes=steady_state_episodes,
        policy_name=DRL_CFG["policy_name"],
        mobility_model=MOBILITY_CFG["mobility_model"],
        num_bs=topology.num_bs,
    )
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
