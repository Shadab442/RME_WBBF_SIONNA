"""Oracle-driven state/action/reward diagnostic for the DRL tilt
controller: uses the repository's existing Dynamic Local Oracle
(DynamicTiltController + LocalTiltSelector coordinate ascent, UNMODIFIED)
as ground truth, instead of random/counterfactual actions, to test
whether the CURRENT DRL state and reward formulation are appropriate.
Does NOT train any DQN and does NOT redefine state, reward, topology,
mobility, association, or KPI computation.

Timescale: real production timescale -- tilt_control_interval_s=300,
measurement_interval_s=1 (300 real measurement slots/interval).

Decision pattern (forward/causal, not production Oracle's own same-
interval hindsight use): at interval k's close, Oracle's coordinate ascent
(DynamicTiltController.update, exactly as production calls it) decides a
JOINT per-sector tilt assignment a*[k] from interval k's own realized,
pooled tilt-sweep power table (IntervalMeasurementPool, unmodified) -- the
same "decide from available data" pattern production's own Dynamic Local
Causal already uses, just applied here to the Oracle's search algorithm.
That decision is then held for the ENTIRE next interval (k+1), and its
outcome (reward, network coverage, per-sector KPIs, next DRL state) is
measured over that interval:

    s[k] -> a*[k] -> r[k+1], C_network[k+1]

async_schedule is forced to "sync" for this run only (a config override,
not a production change) so every sector's DRL observation window closes
at the SAME 300-slot boundary as Oracle's own joint decision -- otherwise
"the state at the decision boundary" wouldn't be a well-defined single
snapshot for a per-sector-staggered scheduler.

s[k] is captured via RLTiltController._prev_observations right after
drl_process_slot (unmodified production code) closes every sector's
window this interval -- the exact observation compute_region_state() built.
DRL's own policy is set to "random" and its chosen action is discarded
every interval (RLTiltController.tilt_idx is overwritten with Oracle's
joint assignment right after capture) -- DRL's OWN action never drives
the network here, only its state/reward machinery is exercised.

Run: python tests/drl/test_oracle_state_action_reward_diagnostic.py \\
    [--num-control-intervals 100]

Saves to results/tests/oracle_state_action_reward_diagnostic/:
  oracle_state_action_reward_diagnostics.csv -- one row per
    (decision_index, sector): current_tilt_deg, state_0..N,
    oracle_action_idx, oracle_tilt_deg, reward, network_coverage,
    sector_coverage, num_served, mean_sinr_db, neighbor_reward_mean,
    next_state_0..N.
"""

import argparse
import csv
import os
import sys

import numpy as np
import scipy.stats
import sionna
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from drl.interface import tilt_idx_to_deg
from helpers.simulation_engine import SimulationEngine
from helpers.utils import RepoLogging, get_logger, load_config

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests",
                       "oracle_state_action_reward_diagnostic")


def run_driving_loop(engine, num_control_intervals, logger):
    """Returns completed_rows: one dict per (decision_index, sector),
    with state_*/oracle_action_idx/oracle_tilt_deg/current_tilt_deg filled
    at decision time, and reward/network_coverage/sector_coverage/
    num_served/mean_sinr_db/neighbor_reward_mean/next_state_* filled in
    once the FOLLOWING interval's outcome is known."""
    num_bs = engine.num_bs
    downtilt_sweep_deg = engine.downtilt_sweep_deg
    neighbor_ids = engine.topology.neighbor_ids

    tilt_prev_deg = np.zeros(num_bs)
    engine.drl_controller.tilt_idx[:] = 0
    global_slot = 0
    pending_rows = None
    completed_rows = []

    for k in range(num_control_intervals + 1):  # +1 extra interval to score the LAST decision
        engine.measurement_pool.start_interval()
        for _ in range(engine.measurement_slots_per_interval):
            live_tilt_deg = tilt_idx_to_deg(engine.drl_controller.tilt_idx, downtilt_sweep_deg)
            ut_loc_this_slot = engine.mobility_list[0].ut_loc.detach().cpu().numpy().copy()
            state0 = engine.step_environment(is_last_step=False)[0]
            drl_power_w_this_slot = engine.measurement_pool.pool_measurement(
                state0, adaptive_tilt_deg=live_tilt_deg, drl_tilt_deg=live_tilt_deg,
                downtilt_sweep_deg=downtilt_sweep_deg, ut_loc_r0=ut_loc_this_slot, needs_sweep=True)
            engine.drl_process_slot(drl_power_w_this_slot, ut_loc_this_slot, global_slot, live_tilt_deg)
            global_slot += 1
        pooled = engine.measurement_pool.pooled_interval()

        state_k = engine.drl_controller._prev_observations.copy()  # [num_bs, num_features]
        reward_k = engine._interval_last_reward.copy()             # [num_bs]

        kpis_k = engine.kpi_manager.compute_ue_kpis(
            pooled["drl_power_w_r0"][None, :, :], threshold_db=engine.coverage_threshold_db,
            ut_loc=pooled["ut_loc_r0"][None, :, :], distance_threshold_m=engine.distance_threshold_m,
            overlap_margin_db=engine.adaptive_legacy_overlap_margin_db, return_counts=True,
        )
        network_coverage_k = float(kpis_k["coverage"])
        sector_coverage_k = kpis_k["per_sector_coverage"]
        served_count_k = kpis_k["per_sector_served_count"]
        serving_idx_k = kpis_k["serving_idx"][0]
        sinr_db_k = kpis_k["sinr_db"][0]
        mean_sinr_k = np.full(num_bs, np.nan)
        for s in range(num_bs):
            mask = serving_idx_k == s
            if mask.any():
                mean_sinr_k[s] = sinr_db_k[mask].mean()

        if pending_rows is not None:
            for s in range(num_bs):
                row = pending_rows[s]
                row["reward"] = float(reward_k[s])
                row["network_coverage"] = network_coverage_k
                row["sector_coverage"] = float(sector_coverage_k[s])
                row["num_served"] = float(served_count_k[s])
                row["mean_sinr_db"] = float(mean_sinr_k[s])
                valid_neighbors = neighbor_ids[s][neighbor_ids[s] >= 0]
                row["neighbor_reward_mean"] = (float(reward_k[valid_neighbors].mean())
                                               if len(valid_neighbors) else float("nan"))
                for i, v in enumerate(state_k[s]):
                    row[f"next_state_{i}"] = float(v)
                completed_rows.append(row)

        if k == num_control_intervals:
            break

        oracle_assignment_k = engine.dynamic_local_oracle_controller.update(
            engine.kpi_manager, pooled["power_table"], engine.coverage_threshold_db)
        oracle_tilt_deg_k = downtilt_sweep_deg[oracle_assignment_k]

        pending_rows = []
        for s in range(num_bs):
            row = {"decision_index": k, "sector": s, "current_tilt_deg": float(tilt_prev_deg[s]),
                  "oracle_action_idx": int(oracle_assignment_k[s]), "oracle_tilt_deg": float(oracle_tilt_deg_k[s])}
            for i, v in enumerate(state_k[s]):
                row[f"state_{i}"] = float(v)
            pending_rows.append(row)

        engine.drl_controller.tilt_idx[:] = oracle_assignment_k
        tilt_prev_deg = oracle_tilt_deg_k

        if k % 10 == 0:
            logger.info("decision %d/%d: network_coverage=%.4f oracle_tilt_deg=%s",
                       k, num_control_intervals, network_coverage_k, np.round(oracle_tilt_deg_k, 1).tolist())

    return completed_rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze(rows, num_bs, num_actions, downtilt_sweep_deg):
    num_features = sum(1 for k in rows[0] if k.startswith("state_"))
    by_sector = {s: sorted([r for r in rows if r["sector"] == s], key=lambda r: r["decision_index"]) for s in range(num_bs)}

    print("\n" + "=" * 70)
    print("7A. DOES THE OPTIMAL ACTION VARY?")
    print("=" * 70)
    action_idx_all = np.array([r["oracle_action_idx"] for r in rows])
    total = len(action_idx_all)
    for a in range(num_actions):
        count = int((action_idx_all == a).sum())
        print(f"  action {a} (tilt={downtilt_sweep_deg[a]:g} deg): {count} selections ({count/total:.1%})")
    modal_frac = np.bincount(action_idx_all, minlength=num_actions).max() / total
    if modal_frac > 0.8:
        print(f"  FLAG: one action accounts for {modal_frac:.1%} of all oracle selections -- "
             f"the environment may not require a strongly state-dependent policy.")

    changed_count, changed_total = 0, 0
    for s in range(num_bs):
        seq = [r["oracle_action_idx"] for r in by_sector[s]]
        for a, b in zip(seq[:-1], seq[1:]):
            changed_total += 1
            changed_count += int(a != b)
    print(f"  fraction of consecutive decisions where oracle action changes (per sector): "
         f"{changed_count/changed_total:.1%}" if changed_total else "  (not enough decisions to compare)")

    print("\n" + "=" * 70)
    print("7B. DOES THE OPTIMAL ACTION DEPEND ON STATE?")
    print("=" * 70)
    sectors_with_multiple_actions = 0
    same_action_dists, diff_action_dists = [], []
    for s in range(num_bs):
        sector_rows = by_sector[s]
        actions = np.array([r["oracle_action_idx"] for r in sector_rows])
        unique_actions = np.unique(actions)
        counts = np.bincount(actions, minlength=num_actions)
        probs = counts[counts > 0] / counts.sum()
        entropy = float(-(probs * np.log2(probs)).sum())
        if len(unique_actions) > 1:
            sectors_with_multiple_actions += 1
        print(f"  sector {s:2d}: unique_actions={len(unique_actions)}, entropy={entropy:.3f} bits, "
             f"action_dist={counts.tolist()}")

        states = np.array([[r[f"state_{i}"] for i in range(num_features)] for r in sector_rows])
        n = len(sector_rows)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(states[i] - states[j])
                (same_action_dists if actions[i] == actions[j] else diff_action_dists).append(d)
    print(f"  fraction of sectors with >1 oracle action used: "
         f"{sectors_with_multiple_actions}/{num_bs} ({sectors_with_multiple_actions/num_bs:.1%})")
    if same_action_dists and diff_action_dists:
        print(f"  mean state distance, SAME action pairs: {np.mean(same_action_dists):.4f}")
        print(f"  mean state distance, DIFFERENT action pairs: {np.mean(diff_action_dists):.4f}")

    print("\n" + "=" * 70)
    print("8. IS THE REWARD ALIGNED WITH NETWORK COVERAGE?")
    print("=" * 70)
    reward_all = np.array([r["reward"] for r in rows])
    network_coverage_all = np.array([r["network_coverage"] for r in rows])
    pearson_r = float(np.corrcoef(reward_all, network_coverage_all)[0, 1])
    spearman_r = float(scipy.stats.spearmanr(reward_all, network_coverage_all).correlation)
    print(f"  Pearson(reward, network_coverage)  = {pearson_r:.4f}")
    print(f"  Spearman(reward, network_coverage) = {spearman_r:.4f}")

    delta_reward, delta_network = [], []
    for s in range(num_bs):
        sector_rows = by_sector[s]
        rewards = np.array([r["reward"] for r in sector_rows])
        net_cov = np.array([r["network_coverage"] for r in sector_rows])
        delta_reward.extend((rewards[1:] - rewards[:-1]).tolist())
        delta_network.extend((net_cov[1:] - net_cov[:-1]).tolist())
    delta_reward = np.array(delta_reward)
    delta_network = np.array(delta_network)
    if len(delta_reward) > 1:
        pearson_d = float(np.corrcoef(delta_reward, delta_network)[0, 1])
        spearman_d = float(scipy.stats.spearmanr(delta_reward, delta_network).correlation)
        print(f"  Pearson(delta_reward, delta_network_coverage)  = {pearson_d:.4f}")
        print(f"  Spearman(delta_reward, delta_network_coverage) = {spearman_d:.4f}")
        reward_up_network_down = int(((delta_reward > 0) & (delta_network < 0)).sum())
        reward_down_network_up = int(((delta_reward < 0) & (delta_network > 0)).sum())
        same_sign = int((np.sign(delta_reward) == np.sign(delta_network)).sum())
        print(f"  reward improves while network coverage worsens: {reward_up_network_down}/{len(delta_reward)}")
        print(f"  reward worsens while network coverage improves: {reward_down_network_up}/{len(delta_reward)}")
        print(f"  fraction sign(delta_reward) == sign(delta_network_coverage): {same_sign/len(delta_reward):.1%}")

    print("\n" + "=" * 70)
    print("9. TRANSITION STATISTICS")
    print("=" * 70)
    state_change_norms = np.array([
        np.linalg.norm(np.array([r[f"next_state_{i}"] for i in range(num_features)])
                       - np.array([r[f"state_{i}"] for i in range(num_features)]))
        for r in rows
    ])
    print(f"  ||next_state - state||: mean={state_change_norms.mean():.4f} std={state_change_norms.std():.4f}")
    print(f"  reward: mean={reward_all.mean():.4f} std={reward_all.std():.4f}")
    print(f"  network_coverage: mean={network_coverage_all.mean():.4f} std={network_coverage_all.std():.4f}")
    print("  conditioned on oracle action:")
    for a in range(num_actions):
        mask = action_idx_all == a
        if mask.any():
            print(f"    action {a}: reward mean={reward_all[mask].mean():.4f}, "
                 f"||s'-s|| mean={state_change_norms[mask].mean():.4f}, "
                 f"network_coverage mean={network_coverage_all[mask].mean():.4f}")

    print("\n" + "=" * 70)
    print("10. FINAL DIAGNOSIS")
    print("=" * 70)
    print("STATE:")
    print(f"  {'Yes' if sectors_with_multiple_actions/num_bs > 0.3 and (not same_action_dists or not diff_action_dists or np.mean(diff_action_dists) > np.mean(same_action_dists)) else 'Weak/no evidence that'} "
         f"the optimal action meaningfully depends on the observed state "
         f"({sectors_with_multiple_actions}/{num_bs} sectors used >1 action).")
    print("ACTION:")
    print(f"  Changing tilt {'does' if state_change_norms.mean() > 0.1 else 'does NOT clearly'} produce "
         f"meaningful state/network-performance changes (mean ||s'-s||={state_change_norms.mean():.4f}, "
         f"network_coverage std={network_coverage_all.std():.4f}).")
    print("REWARD:")
    if len(delta_reward) > 1:
        print(f"  Reward moves with the true objective {same_sign/len(delta_reward):.1%} of the time "
             f"(Pearson r={pearson_r:.3f} on levels, {pearson_d:.3f} on deltas).")
    else:
        print("  Not enough decisions to assess reward/objective alignment.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-control-intervals", type=int, default=100)
    RepoLogging.add_argument(parser)
    args = parser.parse_args()
    RepoLogging.configure(args.log_level, overrides=RepoLogging.parse_overrides(args.log_level_override))
    logger = get_logger(__name__)

    sionna.phy.config.precision = "single"
    sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = load_config()
    cfg["simulation"]["tilt_control_interval_s"] = 300.0
    cfg["simulation"]["measurement_interval_s"] = 1.0
    cfg["algorithms"]["optimization"]["enabled"] = True    # Oracle needed as ground truth here
    cfg["algorithms"]["optimization"]["run_causal"] = False
    cfg["algorithms"]["drl"]["policy_name"] = "random"     # DRL's own action is discarded every interval
    cfg["algorithms"]["drl"]["async_schedule"] = "sync"    # align DRL state capture with Oracle's joint decision boundary
    sionna.phy.config.seed = cfg["simulation"]["environment_seed"]

    os.makedirs(OUT_DIR, exist_ok=True)
    engine = SimulationEngine(cfg)
    logger.info("num_bs=%d num_ut=%d measurement_slots_per_interval=%d num_control_intervals=%d",
               engine.num_bs, engine.num_ut, engine.measurement_slots_per_interval, args.num_control_intervals)

    rows = run_driving_loop(engine, args.num_control_intervals, logger)

    path = os.path.join(OUT_DIR, "oracle_state_action_reward_diagnostics.csv")
    save_csv(rows, path)
    print(f"Saved {len(rows)} rows ({args.num_control_intervals} decisions x {engine.num_bs} sectors) to: {path}")

    analyze(rows, engine.num_bs, len(engine.downtilt_sweep_deg), engine.downtilt_sweep_deg)


if __name__ == "__main__":
    main()
