"""Empirical sector-to-sector tilt influence matrix: for every target
sector i and every candidate affected sector j (ALL 21 sectors, no
predefined geometric neighbor list), measures how much changing ONLY
sector i's tilt moves sector j's coverage/reward-factor over a fixed
60-second window. Not an RL/control experiment -- no DRL state/reward,
no oracle optimization, no training. Answers a single question: which
sectors are empirically coupled by tilt choice, independent of any
geometric adjacency assumption.

Method (deterministic replay, same technique validated in this project's
earlier counterfactual/oracle diagnostics): initialize the simulator,
warm it up so mobility reaches a naturally-occurring state (not the raw
synthetic drop), then cache ONE 60-second sequence of (ut_loc,
LargeScaleState) pairs. Those artifacts are tilt-independent (mobility
and the underlying channel realization don't depend on any sector's
tilt; KpiManager.resolve_power_at_tilt/compute_ue_kpis are pure functions
of (state, tilt)), so replaying the SAME cached 60 slots under every
(target sector, candidate tilt) combination gives every branch the
IDENTICAL stochastic realization -- exact, not "as close as possible",
and needs no RNG snapshot/restore. All 21 x 5 = 105 branches reuse this
one cached trajectory; only KPI math is repeated, not channel simulation.

For target sector i and tested tilt a_i (every other sector held at the
shared initial tilt, 0 deg for all sectors), records for every candidate
sector j:
    reward_factor R_j(i, a_i) = covered UE samples served by j
                                / total network UE samples over the window
so that sum_j R_j == network_coverage exactly (verified numerically per
branch) -- this is each sector's own slice of the global coverage
objective. Sector-to-sector influence is then the max-min spread of R_j
(or of sector_coverage C_j) across the 5 candidate tilts for i.

Run: python tests/drl/test_sector_influence_matrix.py \\
    [--experiment-duration-s 60] [--warmup-seconds 300]

Saves to results/tests/sector_influence_matrix/:
  sector_influence_raw.csv            -- one row per (target, action, candidate).
  sector_influence_matrix.csv         -- [I_ij] reward-factor influence, rows=target i, cols=candidate j.
  sector_coverage_influence_matrix.csv -- [I^C_ij] sector-coverage influence, same layout.
  sector_influence_heatmap.png        -- [I_ij] as an image (row=tilt-changed sector, col=responding sector).
"""

import argparse
import csv
import os
import sys

import numpy as np
import sionna
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from helpers.simulation_engine import SimulationEngine
from helpers.utils import RepoLogging, get_logger, load_config

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "sector_influence_matrix")
NONZERO_THRESHOLD = 1e-6  # "nonzero within numerical precision" for the printed summary only


def collect_cached_trajectory(engine, num_slots):
    """Advances mobility/channel normally for num_slots measurement slots,
    caching each slot's (ut_loc, LargeScaleState) -- both tilt-independent,
    so this ONE cached trajectory is reused, unmodified, for every branch."""
    cached_ut_loc, cached_states = [], []
    for _ in range(num_slots):
        ut_loc_this_slot = engine.mobility_list[0].ut_loc.detach().cpu().numpy().copy()
        state0 = engine.step_environment(is_last_step=False)[0]
        cached_ut_loc.append(ut_loc_this_slot)
        cached_states.append(state0)
    return cached_ut_loc, cached_states


def run_branch(engine, cached_ut_loc, cached_states, tilt_deg_per_sector):
    """Replays the cached trajectory under one fixed tilt configuration --
    returns per-sector (reward_factor, sector_coverage, avg_num_served,
    covered_ue_samples, total_served_ue_samples, mean_sinr_db),
    network_coverage, and total_network_ue_samples."""
    num_bs = engine.num_bs
    served_count_sum = np.zeros(num_bs)
    covered_count_sum = np.zeros(num_bs)
    sinr_sum = np.zeros(num_bs)
    sinr_n = np.zeros(num_bs)
    total_ue_samples = 0

    for ut_loc, state in zip(cached_ut_loc, cached_states):
        power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_deg_per_sector)[0]  # [num_bs, num_ut]
        kpis = engine.kpi_manager.compute_ue_kpis(
            power_w[None, :, :], threshold_db=engine.coverage_threshold_db,
            ut_loc=ut_loc[None, :, :], return_counts=True,
        )
        sinr_db = kpis["sinr_db"][0]
        serving_idx = kpis["serving_idx"][0]
        served_count_sum += kpis["per_sector_served_count"]
        covered_count_sum += kpis["per_sector_covered_count"]
        for j in range(num_bs):
            mask = serving_idx == j
            if mask.any():
                sinr_sum[j] += sinr_db[mask].sum()
                sinr_n[j] += mask.sum()
        total_ue_samples += ut_loc.shape[0]

    network_coverage = float(covered_count_sum.sum() / total_ue_samples)
    reward_factor = covered_count_sum / total_ue_samples  # [num_bs], sums to network_coverage by construction
    with np.errstate(invalid="ignore"):
        sector_coverage = np.where(served_count_sum > 0, covered_count_sum / np.where(served_count_sum > 0, served_count_sum, 1), np.nan)
        mean_sinr_db = np.where(sinr_n > 0, sinr_sum / np.where(sinr_n > 0, sinr_n, 1), np.nan)
    avg_num_served = served_count_sum / len(cached_ut_loc)

    assert abs(reward_factor.sum() - network_coverage) < 1e-9, \
        f"sum(R_j)={reward_factor.sum():.10f} != network_coverage={network_coverage:.10f}"

    return {
        "reward_factor": reward_factor, "sector_coverage": sector_coverage,
        "avg_num_served": avg_num_served, "covered_ue_samples": covered_count_sum,
        "total_served_ue_samples": served_count_sum, "mean_sinr_db": mean_sinr_db,
        "network_coverage": network_coverage, "total_network_ue_samples": total_ue_samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-duration-s", type=int, default=60)
    parser.add_argument("--warmup-seconds", type=int, default=300,
                       help="Real slots advanced before snapshotting, so the starting state is "
                            "naturally-occurring rather than the raw synthetic UE drop.")
    RepoLogging.add_argument(parser)
    args = parser.parse_args()
    RepoLogging.configure(args.log_level, overrides=RepoLogging.parse_overrides(args.log_level_override))
    logger = get_logger(__name__)

    sionna.phy.config.precision = "single"
    sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = load_config()
    cfg["simulation"]["measurement_interval_s"] = 1.0
    cfg["algorithms"]["optimization"]["enabled"] = False  # no oracle optimization
    sionna.phy.config.seed = cfg["simulation"]["environment_seed"]

    os.makedirs(OUT_DIR, exist_ok=True)
    engine = SimulationEngine(cfg)
    num_bs = engine.num_bs
    downtilt_sweep_deg = engine.downtilt_sweep_deg
    num_actions = len(downtilt_sweep_deg)
    logger.info("num_bs=%d num_ut=%d num_actions=%d experiment_duration_s=%d warmup_seconds=%d",
               num_bs, engine.num_ut, num_actions, args.experiment_duration_s, args.warmup_seconds)

    # 1-2. Warm up to a naturally-occurring state
    collect_cached_trajectory(engine, args.warmup_seconds)

    # 3-4. Snapshot: cache the SINGLE 60s trajectory reused by every branch; record initial tilt.
    initial_tilt_deg = np.zeros(num_bs)  # shared "No Tilt" baseline, held fixed for every non-target sector
    cached_ut_loc, cached_states = collect_cached_trajectory(engine, args.experiment_duration_s)
    logger.info("Cached %d slots for replay", len(cached_ut_loc))

    raw_rows = []
    reward_factor_grid = np.zeros((num_bs, num_actions, num_bs))    # [target, action, candidate]
    sector_coverage_grid = np.full((num_bs, num_actions, num_bs), np.nan)

    for i in range(num_bs):
        for action_idx, a_i in enumerate(downtilt_sweep_deg):
            tilt_deg_per_sector = initial_tilt_deg.copy()
            tilt_deg_per_sector[i] = a_i
            branch = run_branch(engine, cached_ut_loc, cached_states, tilt_deg_per_sector)
            reward_factor_grid[i, action_idx] = branch["reward_factor"]
            sector_coverage_grid[i, action_idx] = branch["sector_coverage"]

            for j in range(num_bs):
                raw_rows.append({
                    "target_sector": i, "candidate_sector": j,
                    "original_tilt_deg": float(initial_tilt_deg[i]), "tested_tilt_deg": float(a_i),
                    "avg_num_served": float(branch["avg_num_served"][j]),
                    "covered_ue_samples": float(branch["covered_ue_samples"][j]),
                    "total_served_ue_samples": float(branch["total_served_ue_samples"][j]),
                    "sector_coverage": float(branch["sector_coverage"][j]),
                    "mean_sinr_db": float(branch["mean_sinr_db"][j]),
                    "reward_factor": float(branch["reward_factor"][j]),
                    "network_coverage": branch["network_coverage"],
                    "total_network_ue_samples": branch["total_network_ue_samples"],
                })
        logger.info("target sector %d/%d done", i + 1, num_bs)

    # --- Influence matrices ---
    I = reward_factor_grid.max(axis=1) - reward_factor_grid.min(axis=1)               # [target, candidate]
    I_C = np.nanmax(sector_coverage_grid, axis=1) - np.nanmin(sector_coverage_grid, axis=1)

    # --- Save raw + matrices ---
    raw_path = os.path.join(OUT_DIR, "sector_influence_raw.csv")
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    print(f"Saved {len(raw_rows)} rows to: {raw_path}")

    def save_matrix(matrix, path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["target_sector"] + [f"candidate_{j}" for j in range(num_bs)])
            for i in range(num_bs):
                writer.writerow([i] + matrix[i].tolist())

    matrix_path = os.path.join(OUT_DIR, "sector_influence_matrix.csv")
    coverage_matrix_path = os.path.join(OUT_DIR, "sector_coverage_influence_matrix.csv")
    save_matrix(I, matrix_path)
    save_matrix(I_C, coverage_matrix_path)
    print(f"Saved reward-factor influence matrix to: {matrix_path}")
    print(f"Saved sector-coverage influence matrix to: {coverage_matrix_path}")

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(I, cmap="viridis", aspect="equal")
    ax.set_xlabel("responding sector j")
    ax.set_ylabel("tilt-changed sector i")
    ax.set_title("Sector-to-sector reward-factor influence I_ij")
    fig.colorbar(im, ax=ax, label="I_ij (max-min reward factor)")
    heatmap_path = os.path.join(OUT_DIR, "sector_influence_heatmap.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved heatmap to: {heatmap_path}")

    # --- Per-target-sector ranked summary ---
    print("\n" + "=" * 70)
    print("PER-TARGET-SECTOR INFLUENCE RANKINGS")
    print("=" * 70)
    for i in range(num_bs):
        order = np.argsort(-I[i])
        print(f"\nTarget sector {i}:")
        print(f"{'candidate':>10}  {'reward-factor influence':>24}  {'coverage influence':>20}")
        for j in order:
            print(f"{j:>10}  {I[i, j]:>24.6f}  {I_C[i, j]:>20.6f}")

        self_influence = I[i, i]
        cross = np.delete(I[i], i)
        cross_candidates = np.delete(np.arange(num_bs), i)
        largest_cross_idx = cross_candidates[np.argmax(cross)]
        nonzero_count = int((I[i] > NONZERO_THRESHOLD).sum())
        print(f"  self-influence I_{i}{i} = {self_influence:.6f}")
        print(f"  largest cross-sector influence: sector {largest_cross_idx} "
             f"(I={I[i, largest_cross_idx]:.6f})")
        print(f"  sectors with influence > {NONZERO_THRESHOLD:g}: {nonzero_count}/{num_bs}")
        if self_influence > 0:
            ratios = I[i] / self_influence
            print(f"  cross/self influence ratio: min={ratios.min():.4f} max={np.delete(ratios, i).max():.4f} "
                 f"mean={np.delete(ratios, i).mean():.4f}")


if __name__ == "__main__":
    main()
