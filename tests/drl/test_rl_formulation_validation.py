"""Validates whether the proposed 57-D observation o_i[k] (26 occupancy +
26 coverage + 5 tilts: own + 4 side-sharing neighbors) preserves the
information needed to choose sector i's correct UNILATERAL tilt action --
NOT whether it matches some engineered "global state", NOT WESN/sequence
history, and NOT actual RL training. Ground truth is the counterfactual
action-value vector obtained directly from the simulator,
J_i(x) = [J_i(0deg),...,J_i(8deg)], holding every other sector's tilt
fixed at theta[k] -- never an engineered global feature. a_i^BR =
argmax_a J_i(a) is the globally-EVALUATED unilateral best response for
sector i, not a claim of joint-global optimality.

Main question: given o_i[k], can sector i choose a near-optimal
unilateral action with low global-coverage regret? Judged PRIMARILY by
decision regret from a supervised o_i -> [A(a_0),...,A(a_4)] diagnostic
(regret = J(a_i^BR) - J(argmax_a predicted A(a))); nearest-state transfer
regret (in standardized 57-D space) is secondary/supporting evidence for
the same question, not an independent pass/fail axis.

============================================================
GEOMETRY: X_s = sector s + its 4 side-sharing neighbors, pooled into 26
regions via a 144.3m-tile overlay, offset-optimized per target.
Verified at runtime (find_consistent_target_sectors), not assumed --
whether every one of the 21 sectors can even produce this exact 57-D
definition is itself part of what this test establishes. Historically
only 6 of the 9 full-4-neighbor sectors achieve exactly 26 regions --
{0,2,5,8,15,18}, a real geometric fact from the hex grid's rotational
symmetry classes -- so unless the runtime check finds all 21 consistent,
validation is scoped to whichever sectors it actually finds, with an
explicit STATE GEOMETRY ISSUE notice (see main()).
============================================================

Joint tilt diversity: joint tilts are driven by the PRODUCTION async
scheduler + RandomPolicy (RLTiltController, policy_name=random,
training=False so nothing learns) continuously throughout the run, so
every sampled snapshot has a realistically diverse, staggered joint tilt
vector -- never all-zero. The async scheduler keeps moving tilts WHILE
window k is being observed, so o_i[k]'s own+neighbor-tilt features and
its per-slot power/coverage resolution both use theta[k] -- the tilt
vector actually live at each slot, not a snapshot taken before window k
started (see compute_observation).

Counterfactual branches reuse the same deterministic-replay technique
validated in this project's earlier diagnostics: mobility and the
channel realization are tilt-independent (KpiManager.resolve_power_at_tilt/
compute_ue_kpis are pure functions of (state, tilt)), so caching one
window's (ut_loc, LargeScaleState) sequence and replaying it under every
candidate tilt for the target sector gives every branch an IDENTICAL
stochastic realization -- actions are always compared on matched
randomness. This measures CONDITIONAL performance for one matched replay
per state, not an expected action value averaged over independent futures.

Run: python tests/drl/test_rl_formulation_validation.py \\
    [--num-network-states 100] [--k-neighbors 5] [--warmup-intervals 5]

Saves to results/tests/rl_formulation_validation/:
  rl_state_sufficiency_extended.csv
  rl_state_neighbor_consistency.csv
  rl_state_transfer_regret_summary.csv
  rl_state_value_prediction_summary.csv
  plot_distance_vs_action_value_distance.png
  plot_distance_vs_transfer_regret.png
  plot_transfer_regret_distribution.png
  plot_best_response_action_distribution.png
  plot_best_second_gap_distribution.png
  plot_predicted_vs_true_advantage.png
"""

import argparse
import csv
import os
import sys

import numpy as np
import scipy.stats
import sionna
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from drl.interface import tilt_idx_to_deg
from helpers.simulation_engine import SimulationEngine
from helpers.utils import RepoLogging, get_logger, load_config

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "rl_formulation_validation")
NUM_REGIONS = 26  # design parameter of the proposed X_s = sector + 4 neighbors state, not derived
GEOMETRY_ISSUE_MESSAGE = (
    "STATE GEOMETRY ISSUE:\n"
    "The proposed fixed 57-D state is currently not defined consistently\n"
    "for all 21 sector agents."
)
OCCUPANCY_EMA_ALPHA = 0.8
EPSILONS = [0.0001, 0.0005, 0.001]


# ============================================================
# Tile geometry (unchanged from the earlier version of this script)
# ============================================================

def compute_tile_geometry(topo, grid_points, target, tile_size, search_n=10):
    neighbors = topo.neighbor_ids[target]
    neighbors = neighbors[neighbors >= 0]
    sector_set = [int(target)] + neighbors.tolist()
    assert len(sector_set) == 5, f"target {target} does not have all 4 side-sharing neighbors"

    relevant_bs = topo.bs_loc[0, sector_set, :2].cpu().numpy()
    center = relevant_bs.mean(axis=0)
    R = float(topo.grid.cell_radius.item())
    max_reach = np.linalg.norm(relevant_bs - center[None, :], axis=1).max() + R
    half_extent = max_reach + tile_size
    n_tiles_per_side = int(np.ceil(2 * half_extent / tile_size)) + 3
    sub = np.linspace(-0.5, 0.5, 5) * tile_size

    def tiles_touched(ox, oy):
        offset = (n_tiles_per_side * tile_size) / 2
        xs = center[0] - offset + ox + tile_size * (np.arange(n_tiles_per_side) + 0.5)
        ys = center[1] - offset + oy + tile_size * (np.arange(n_tiles_per_side) + 0.5)
        keys = set()
        for cx in xs:
            for cy in ys:
                gx, gy = np.meshgrid(cx + sub, cy + sub)
                pts = np.stack([gx.ravel(), gy.ravel()], axis=-1)
                sector_idx, _ = topo.assign_to_sector_rhombus_grid(pts)
                if np.isin(sector_idx, sector_set).any():
                    row = int(np.floor((cx - center[0] + offset) / tile_size))
                    col = int(np.floor((cy - center[1] + offset) / tile_size))
                    keys.add((row, col))
        return keys

    best_keys, best_origin = None, None
    for ox in np.linspace(0, tile_size, search_n, endpoint=False):
        for oy in np.linspace(0, tile_size, search_n, endpoint=False):
            keys = tiles_touched(ox, oy)
            if best_keys is None or len(keys) < len(best_keys):
                best_keys, best_origin = keys, (ox, oy)

    offset = (n_tiles_per_side * tile_size) / 2
    origin = np.array([center[0] - offset + best_origin[0], center[1] - offset + best_origin[1]])
    tile_index_map = {key: i for i, key in enumerate(sorted(best_keys))}
    return origin, sector_set, tile_index_map


def find_consistent_target_sectors(topo, grid_points, tile_size, num_bs):
    """Which of the num_bs sectors can actually produce this exact 57-D
    definition (X_s = sector + its 4 side-sharing neighbors, tiled into
    exactly NUM_REGIONS regions) -- computed here, not assumed, since
    whether the same fixed-size state is even DEFINABLE for every sector
    is itself part of what this test must establish, not a given. A sector
    without all 4 side-sharing neighbors (network edge) raises inside
    compute_tile_geometry; one that has them may still tile to a different
    region count depending on its rotational symmetry class.

    :output: (consistent_sectors, all_bs_consistent) -- consistent_sectors
        is every sector where this state definition is well-formed with
        exactly NUM_REGIONS regions; all_bs_consistent is whether that's
        every one of the num_bs sectors (i.e. whether the proposed state
        could be used network-wide, not just for a subset).
    """
    consistent_sectors = []
    for s in range(num_bs):
        try:
            _, _, tile_index_map = compute_tile_geometry(topo, grid_points, s, tile_size)
        except AssertionError:
            continue
        if len(tile_index_map) == NUM_REGIONS:
            consistent_sectors.append(s)
    return consistent_sectors, len(consistent_sectors) == num_bs


def assign_ue_to_tile(ue_xy, sector_idx, sector_set, origin, tile_size, tile_index_map):
    in_xs = np.isin(sector_idx, sector_set)
    tile_idx = np.full(len(ue_xy), -1, dtype=int)
    if not in_xs.any():
        return tile_idx
    rel = (ue_xy[in_xs] - origin) / tile_size
    rc = np.floor(rel).astype(int)
    keys = [tuple(k) for k in rc]
    tile_idx[in_xs] = [tile_index_map.get(k, -1) for k in keys]
    return tile_idx


class RegionTracker:
    def __init__(self, num_regions):
        self.num_regions = num_regions
        self.visit_count = np.zeros(num_regions)
        self.covered_count = np.zeros(num_regions)
        self.occupancy_ema = np.zeros(num_regions)
        self._initialized = False

    def set_window_sums(self, visit_sum, covered_sum):
        self.visit_count[:] = visit_sum
        self.covered_count[:] = covered_sum

    def compute_and_reset(self, total_network_visits):
        occ_tilde = self.visit_count / total_network_visits if total_network_visits > 0 else np.zeros(self.num_regions)
        if not self._initialized:
            self.occupancy_ema = occ_tilde.copy()
            self._initialized = True
        else:
            self.occupancy_ema = OCCUPANCY_EMA_ALPHA * occ_tilde + (1 - OCCUPANCY_EMA_ALPHA) * self.occupancy_ema
        visited = self.visit_count > 0
        with np.errstate(invalid="ignore"):
            coverage = np.where(visited, self.covered_count / np.where(visited, self.visit_count, 1), 0.0)
        occupancy_out = self.occupancy_ema.copy()
        self.visit_count[:] = 0.0
        self.covered_count[:] = 0.0
        return occupancy_out, coverage


def compute_observation(engine, tracker, tile_geom, target, window_tilt_deg, window_ut_loc, window_states,
                        tilt_deg_min, tilt_deg_max, neighbor_ids_of_target, tile_size):
    """The proposed 57-D o_i(x): 26 occupancy + 26 coverage (this window) +
    5 tilts (own + 4 neighbors). Each slot's power/coverage is resolved at
    THAT slot's own live tilt vector (window_tilt_deg[t], one entry per
    slot) -- the async scheduler keeps moving tilts underneath the window,
    so reusing one constant tilt vector across every slot would silently
    reconstruct the window under conditions that never actually applied.
    The tilt features (own + neighbors) describe theta[k], the joint tilt
    AT the window's decision boundary (its last slot) -- window_tilt_deg[-1].
    """
    origin, sector_set, tile_index_map = tile_geom
    tile_visit_sum = np.zeros(NUM_REGIONS)
    tile_covered_sum = np.zeros(NUM_REGIONS)
    network_total_samples = 0

    for ut_loc, state, tilt_deg_this_slot in zip(window_ut_loc, window_states, window_tilt_deg):
        power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_deg_this_slot)[0]
        kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :], threshold_db=engine.coverage_threshold_db)
        sinr_db = kpis["sinr_db"][0]
        covered = sinr_db > engine.coverage_threshold_db

        sector_idx_geo, _ = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])
        tile_idx = assign_ue_to_tile(ut_loc[:, :2], sector_idx_geo, sector_set, origin, tile_size, tile_index_map)
        valid_tile = tile_idx >= 0
        np.add.at(tile_visit_sum, tile_idx[valid_tile], 1.0)
        np.add.at(tile_covered_sum, tile_idx[valid_tile], covered[valid_tile].astype(float))
        network_total_samples += ut_loc.shape[0]

    tracker.set_window_sums(tile_visit_sum, tile_covered_sum)
    occupancy, coverage = tracker.compute_and_reset(network_total_samples)

    theta_k = window_tilt_deg[-1]
    own_tilt_norm = (theta_k[target] - tilt_deg_min) / (tilt_deg_max - tilt_deg_min)
    neighbor_tilts_norm = [(theta_k[n] - tilt_deg_min) / (tilt_deg_max - tilt_deg_min)
                           for n in neighbor_ids_of_target]
    observation = np.concatenate([occupancy, coverage, [own_tilt_norm], neighbor_tilts_norm])
    assert observation.shape[0] == 2 * NUM_REGIONS + 5
    return observation


def network_coverage_for_branch(engine, cached_ut_loc, cached_states, tilt_deg_per_sector):
    """J_x(a) = C_network -- direct network-wide coverage, no tile/region
    machinery needed (that's only for the OBSERVATION, not the ground
    truth used to evaluate it)."""
    covered_sum, total_sum = 0, 0
    for ut_loc, state in zip(cached_ut_loc, cached_states):
        power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_deg_per_sector)[0]
        kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :], threshold_db=engine.coverage_threshold_db)
        covered_sum += int((kpis["sinr_db"][0] > engine.coverage_threshold_db).sum())
        total_sum += ut_loc.shape[0]
    return covered_sum / total_sum if total_sum > 0 else float("nan")


def advance_tilts_one_slot(engine, num_bs, global_slot):
    """Drives the production async scheduler + RandomPolicy for one slot
    (training=False -- nothing learns; RandomPolicy.act() ignores its
    observations argument entirely, so dummy zeros are fine) -- the only
    purpose is realistic, staggered, non-zero joint tilt diversity."""
    closes = engine.scheduler.closes(global_slot)
    engine.drl_controller.update(np.zeros((num_bs, 1)), np.zeros(num_bs), training=False, schedule=closes)


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def get_vec(row, n, prefix):
    return np.array([row[f"{prefix}{i}"] for i in range(n)])


# ============================================================
# Main driving loop + dataset construction
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-network-states", type=int, default=100)
    parser.add_argument("--k-neighbors", type=int, default=5)
    parser.add_argument("--warmup-intervals", type=int, default=5)
    RepoLogging.add_argument(parser)
    args = parser.parse_args()
    RepoLogging.configure(args.log_level, overrides=RepoLogging.parse_overrides(args.log_level_override))
    logger = get_logger(__name__)

    sionna.phy.config.precision = "single"
    sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = load_config()
    cfg["simulation"]["tilt_control_interval_s"] = 300.0
    cfg["simulation"]["measurement_interval_s"] = 1.0
    cfg["algorithms"]["optimization"]["enabled"] = False
    cfg["algorithms"]["drl"]["policy_name"] = "random"  # diverse joint tilts via the real async scheduler + RandomPolicy
    sionna.phy.config.seed = cfg["simulation"]["environment_seed"]

    os.makedirs(OUT_DIR, exist_ok=True)
    engine = SimulationEngine(cfg)
    num_bs = engine.num_bs
    downtilt_sweep_deg = engine.downtilt_sweep_deg
    num_actions = len(downtilt_sweep_deg)
    tilt_deg_min, tilt_deg_max = float(downtilt_sweep_deg.min()), float(downtilt_sweep_deg.max())

    grid_points, gn = engine.topology.build_sector_rhombus_grid(cfg["algorithms"]["drl"]["spatial_grid_cell_size_m"])
    R = float(engine.topology.grid.cell_radius.item())
    tile_size = 3 * (R / gn)

    VALID_TARGET_SECTORS, geometry_consistent_for_all = find_consistent_target_sectors(
        engine.topology, grid_points, tile_size, num_bs)
    if geometry_consistent_for_all:
        print(f"Geometry check: all {num_bs} sectors produce exactly {NUM_REGIONS} regions -- "
             f"validating all {num_bs} sectors.")
    else:
        print(GEOMETRY_ISSUE_MESSAGE)
        print(f"Sectors where this state IS well-formed (exactly {NUM_REGIONS} regions): {VALID_TARGET_SECTORS}")
        print(f"Validation below is scoped to those {len(VALID_TARGET_SECTORS)} sectors only.")

    tile_geoms, trackers, neighbor_lists = {}, {}, {}
    for t in VALID_TARGET_SECTORS:
        tile_geoms[t] = compute_tile_geometry(engine.topology, grid_points, t, tile_size)
        assert len(tile_geoms[t][2]) == NUM_REGIONS
        trackers[t] = RegionTracker(NUM_REGIONS)
        neighbor_lists[t] = tile_geoms[t][1][1:]
        logger.info("target %d: sector_set=%s verified", t, tile_geoms[t][1])

    print("NOTE: this run measures CONDITIONAL performance for one matched replay per state, "
         "not an expected action value averaged over futures.")

    global_slot = 0
    logger.info("warming up %d intervals for joint-tilt diversity", args.warmup_intervals)
    for _ in range(args.warmup_intervals * engine.measurement_slots_per_interval):
        engine.step_environment(is_last_step=False)
        advance_tilts_one_slot(engine, num_bs, global_slot)
        global_slot += 1

    rows = []
    for state_id in range(args.num_network_states):
        # Window k: real trajectory under whatever joint tilt is ACTUALLY
        # live each slot -- captured BEFORE that slot's own
        # advance_tilts_one_slot call, since that's the tilt that governed
        # this slot's real UE conditions. The async scheduler keeps moving
        # tilts underneath the window as it runs, so a single snapshot
        # taken once before the loop would silently go stale mid-window.
        window_ut_loc, window_states, window_tilt_deg = [], [], []
        for _ in range(engine.measurement_slots_per_interval):
            live_tilt_deg = tilt_idx_to_deg(engine.drl_controller.tilt_idx, downtilt_sweep_deg)
            ut_loc = engine.mobility_list[0].ut_loc.detach().cpu().numpy().copy()
            state0 = engine.step_environment(is_last_step=False)[0]
            advance_tilts_one_slot(engine, num_bs, global_slot)
            global_slot += 1
            window_ut_loc.append(ut_loc)
            window_states.append(state0)
            window_tilt_deg.append(live_tilt_deg)

        # theta[k]: the joint tilt AT window k's decision boundary (its
        # last slot) -- what o_i[k]'s own+neighbor-tilt features describe,
        # and what the counterfactual sweep below holds fixed for every
        # sector except the target (the decision-point configuration, not
        # a snapshot from before window k even started).
        theta_k = window_tilt_deg[-1]

        obs_per_target = {}
        for target in VALID_TARGET_SECTORS:
            obs_per_target[target] = compute_observation(
                engine, trackers[target], tile_geoms[target], target, window_tilt_deg,
                window_ut_loc, window_states, tilt_deg_min, tilt_deg_max, neighbor_lists[target], tile_size)

        # Future window: the continuing real trajectory (also advances tilts).
        future_windows = []
        r0_ut_loc, r0_states = [], []
        for _ in range(engine.measurement_slots_per_interval):
            ut_loc = engine.mobility_list[0].ut_loc.detach().cpu().numpy().copy()
            state0 = engine.step_environment(is_last_step=False)[0]
            advance_tilts_one_slot(engine, num_bs, global_slot)
            global_slot += 1
            r0_ut_loc.append(ut_loc)
            r0_states.append(state0)
        future_windows.append((r0_ut_loc, r0_states))

        for target in VALID_TARGET_SECTORS:
            # Counterfactual sweep: ONLY the target sector's tilt varies
            # across the 5 candidates; the other 20 stay fixed at theta[k]
            # (the decision-point configuration o_i[k] itself describes),
            # replayed against the SAME future realization(s) for every
            # candidate. a_i^BR below is the globally-EVALUATED unilateral
            # best response for sector i specifically -- not a claim that
            # this is the globally optimal JOINT action across all 21
            # sectors (that would require jointly varying every sector).
            J_per_action = np.zeros(num_actions)
            for action_idx, a in enumerate(downtilt_sweep_deg):
                tilt_deg_per_sector = theta_k.copy()
                tilt_deg_per_sector[target] = a
                J_r = [network_coverage_for_branch(engine, ut_loc_list, state_list, tilt_deg_per_sector)
                      for ut_loc_list, state_list in future_windows]
                J_per_action[action_idx] = np.mean(J_r)

            best_idx = int(np.argmax(J_per_action))  # a_i^BR: unilateral best response, not joint-optimal
            advantage = J_per_action - J_per_action[best_idx]
            sorted_J = np.sort(J_per_action)[::-1]

            row = {"network_state_id": state_id, "target_sector": target}
            for s in range(num_bs):
                row[f"joint_tilt_{s}"] = float(theta_k[s])
            for i, v in enumerate(obs_per_target[target]):
                row[f"local_state_{i}"] = float(v)
            for a in range(num_actions):
                row[f"J_action_{a}"] = float(J_per_action[a])
            for a in range(num_actions):
                row[f"advantage_action_{a}"] = float(advantage[a])
            row["best_response_action_idx"] = best_idx
            row["best_response_tilt_deg"] = float(downtilt_sweep_deg[best_idx])
            row["best_J"] = float(sorted_J[0])
            row["second_best_J"] = float(sorted_J[1])
            row["best_second_gap"] = float(sorted_J[0] - sorted_J[1])
            rows.append(row)

        if state_id % 10 == 0:
            logger.info("network state %d/%d done", state_id, args.num_network_states)

    save_csv(rows, os.path.join(OUT_DIR, "rl_state_sufficiency_extended.csv"))
    print(f"Saved {len(rows)} rows ({args.num_network_states} states x {len(VALID_TARGET_SECTORS)} sectors) "
         f"to rl_state_sufficiency_extended.csv")

    num_features = 2 * NUM_REGIONS + 5
    analyze(rows, num_features, num_actions, downtilt_sweep_deg, args.k_neighbors,
           len(VALID_TARGET_SECTORS), geometry_consistent_for_all)


# ============================================================
# Analysis: standardized nearest-neighbor consistency + direct transfer
# regret (Sec 5/7), epsilon-optimal-set overlap, supervised decision-regret
# diagnostic (Sec 6, primary), final verdict (Sec 9).
# ============================================================

def analyze(rows, num_features, num_actions, downtilt_sweep_deg, k_neighbors,
           num_target_sectors, geometry_consistent_for_all):
    by_sector = {}
    for r in rows:
        by_sector.setdefault(r["target_sector"], []).append(r)

    neighbor_rows, transfer_rows = [], []
    all_d_o, all_d_A = [], []
    all_d_o_for_regret, all_transfer_regret = [], []  # 2 regret entries (both directions) per (d_o, pair)
    eps_consistency = {eps: [] for eps in EPSILONS}
    exact_agree_count, exact_agree_total = 0, 0

    for sector, sector_rows in by_sector.items():
        X = np.array([get_vec(r, num_features, "local_state_") for r in sector_rows])
        A = np.array([[r[f"advantage_action_{a}"] for a in range(num_actions)] for r in sector_rows])
        J = np.array([[r[f"J_action_{a}"] for a in range(num_actions)] for r in sector_rows])
        n = len(sector_rows)
        if n < 2:
            continue
        # Standardize (z-score) each of the 57 features across this
        # sector's own samples before computing distances -- diagnostic
        # only, does not touch the production observation. Otherwise
        # features with a naturally larger numeric range would dominate
        # raw Euclidean distance regardless of how decision-relevant they
        # actually are.
        mu, sigma = X.mean(axis=0), X.std(axis=0)
        X_std = (X - mu) / (sigma + 1e-8)
        dmat = np.linalg.norm(X_std[:, None, :] - X_std[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)

        for m in range(n):
            k = min(k_neighbors, n - 1)
            neighbor_idx = np.argsort(dmat[m])[:k]
            for nb in neighbor_idx:
                d_o = dmat[m, nb]
                d_A = np.linalg.norm(A[m] - A[nb])
                all_d_o.append(d_o)
                all_d_A.append(d_A)

                a_m = int(np.argmax(J[m]))
                a_nb = int(np.argmax(J[nb]))
                exact_agree_total += 1
                exact_agree_count += int(a_m == a_nb)

                # direct state-aliasing test: transferring m's best action
                # to n (and vice versa), in standardized-observation space (Sec 7)
                r_m_to_nb = J[nb, a_nb] - J[nb, a_m]
                r_nb_to_m = J[m, a_m] - J[m, a_nb]
                all_transfer_regret.extend([r_m_to_nb, r_nb_to_m])
                all_d_o_for_regret.extend([d_o, d_o])

                for eps in EPSILONS:
                    set_m = set(np.where(J[m] >= J[m].max() - eps)[0].tolist())
                    set_nb = set(np.where(J[nb] >= J[nb].max() - eps)[0].tolist())
                    overlap = len(set_m & set_nb) / len(set_m | set_nb) if (set_m | set_nb) else 1.0
                    eps_consistency[eps].append(overlap)

                neighbor_rows.append({
                    "target_sector": sector, "state_m": sector_rows[m]["network_state_id"],
                    "state_n": sector_rows[nb]["network_state_id"], "d_o": d_o, "d_A": d_A,
                    "a_star_m": a_m, "a_star_n": a_nb, "transfer_regret_m_to_n": r_m_to_nb,
                    "transfer_regret_n_to_m": r_nb_to_m,
                })

    save_csv(neighbor_rows, os.path.join(OUT_DIR, "rl_state_neighbor_consistency.csv"))

    all_d_o, all_d_A = np.array(all_d_o), np.array(all_d_A)
    all_d_o_for_regret, all_transfer_regret = np.array(all_d_o_for_regret), np.array(all_transfer_regret)
    pearson_do_dA = float(np.corrcoef(all_d_o, all_d_A)[0, 1]) if len(all_d_o) > 1 else float("nan")

    transfer_summary = {
        "mean_nn_state_distance": float(all_d_o.mean()), "mean_nn_action_value_distance": float(all_d_A.mean()),
        "pearson_do_dA": pearson_do_dA, "mean_transfer_regret": float(all_transfer_regret.mean()),
        "p95_transfer_regret": float(np.percentile(all_transfer_regret, 95)),
        "max_transfer_regret": float(all_transfer_regret.max()),
        "exact_best_action_agreement": exact_agree_count / exact_agree_total if exact_agree_total else float("nan"),
    }
    for eps in EPSILONS:
        transfer_summary[f"eps_optimal_overlap_{eps}"] = float(np.mean(eps_consistency[eps])) if eps_consistency[eps] else float("nan")
    save_csv([transfer_summary], os.path.join(OUT_DIR, "rl_state_transfer_regret_summary.csv"))

    # --- Supervised decision-regret diagnostic: o_i -> [A(a_0),...,A(a_4)] ---
    # This is the PRIMARY evidence for state sufficiency (exact-action
    # accuracy is secondary -- several actions can be practically
    # equivalent, so predicting a near-tied alternative isn't a real error).
    from sklearn.ensemble import RandomForestRegressor

    state_ids = sorted(set(r["network_state_id"] for r in rows))
    split = len(state_ids) // 2
    train_ids, test_ids = set(state_ids[:split]), set(state_ids[split:])
    train_rows = [r for r in rows if r["network_state_id"] in train_ids]
    test_rows = [r for r in rows if r["network_state_id"] in test_ids]

    pred_summary = {}
    plot_pred, plot_true = None, None
    if len(train_rows) >= 4 and len(test_rows) >= 1:
        X_train = np.array([get_vec(r, num_features, "local_state_") for r in train_rows])
        Y_train = np.array([[r[f"advantage_action_{a}"] for a in range(num_actions)] for r in train_rows])
        X_test = np.array([get_vec(r, num_features, "local_state_") for r in test_rows])
        Y_test = np.array([[r[f"advantage_action_{a}"] for a in range(num_actions)] for r in test_rows])
        J_test = np.array([[r[f"J_action_{a}"] for a in range(num_actions)] for r in test_rows])

        reg = RandomForestRegressor(n_estimators=200, random_state=0)
        reg.fit(X_train, Y_train)
        Y_pred = reg.predict(X_test)

        mse = float(np.mean((Y_pred - Y_test) ** 2))
        a_hat = np.argmax(Y_pred, axis=1)
        a_true = np.argmax(J_test, axis=1)
        regret = np.array([J_test[i].max() - J_test[i, a_hat[i]] for i in range(len(test_rows))])
        exact_acc = float((a_hat == a_true).mean())
        eps_acc = {eps: float(np.mean([J_test[i, a_hat[i]] >= J_test[i].max() - eps for i in range(len(test_rows))]))
                  for eps in EPSILONS}

        pred_summary = {
            "n_train_states": split, "n_test_states": len(state_ids) - split,
            "value_prediction_mse": mse, "mean_decision_regret": float(regret.mean()),
            "median_decision_regret": float(np.median(regret)), "p95_decision_regret": float(np.percentile(regret, 95)),
            "max_decision_regret": float(regret.max()),
            "exact_action_accuracy": exact_acc,
        }
        for eps in EPSILONS:
            pred_summary[f"eps_optimal_decision_accuracy_{eps}"] = eps_acc[eps]
        plot_pred, plot_true = Y_pred, Y_test
    else:
        pred_summary = {"note": "not enough states for a train/test split"}
    save_csv([pred_summary], os.path.join(OUT_DIR, "rl_state_value_prediction_summary.csv"))

    # --- Plots ---
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_d_o, all_d_A, alpha=0.4, s=12)
    ax.set_xlabel("local-observation distance d_o")
    ax.set_ylabel("action-value-vector distance d_A")
    ax.set_title(f"State distance vs action-value distance (Pearson r={pearson_do_dA:.3f})")
    fig.savefig(os.path.join(OUT_DIR, "plot_distance_vs_action_value_distance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(all_d_o_for_regret, all_transfer_regret, alpha=0.4, s=12, color="tab:red")
    ax.set_xlabel("local-observation distance d_o")
    ax.set_ylabel("cross-state transfer regret")
    ax.set_title("State distance vs action-transfer regret")
    fig.savefig(os.path.join(OUT_DIR, "plot_distance_vs_transfer_regret.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(all_transfer_regret, bins=30, color="tab:orange")
    ax.set_xlabel("cross-state transfer regret (nearest-neighbor pairs)")
    ax.set_ylabel("count")
    ax.set_title("Transfer regret distribution")
    fig.savefig(os.path.join(OUT_DIR, "plot_transfer_regret_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    best_actions_all = [r["best_response_action_idx"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(range(num_actions), np.bincount(best_actions_all, minlength=num_actions), color="tab:green")
    ax.set_xlabel("action index")
    ax.set_ylabel("# (state, sector) samples where this is the best response")
    ax.set_title("Best-response action distribution")
    fig.savefig(os.path.join(OUT_DIR, "plot_best_response_action_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    gaps = [r["best_second_gap"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(gaps, bins=30, color="tab:purple")
    ax.set_xlabel("best-vs-second-best global coverage gap")
    ax.set_ylabel("count")
    ax.set_title("Best-vs-second-best gap distribution")
    fig.savefig(os.path.join(OUT_DIR, "plot_best_second_gap_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    if plot_pred is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(plot_true.ravel(), plot_pred.ravel(), alpha=0.4, s=12)
        lims = [min(plot_true.min(), plot_pred.min()), max(plot_true.max(), plot_pred.max())]
        ax.plot(lims, lims, "k--", alpha=0.5)
        ax.set_xlabel("true advantage")
        ax.set_ylabel("predicted advantage")
        ax.set_title("Supervised diagnostic: predicted vs true action-relative value")
        fig.savefig(os.path.join(OUT_DIR, "plot_predicted_vs_true_advantage.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- Final verdict: decision regret (supervised) is PRIMARY evidence;
    # nearest-state transfer regret is secondary/supporting (Sec 7's direct
    # aliasing test). Exact-action agreement is not a pass/fail input --
    # several actions can be practically equivalent, so a label mismatch
    # alone isn't evidence of aliasing; only real regret is.
    has_supervised = "mean_decision_regret" in pred_summary
    mean_regret = pred_summary.get("mean_decision_regret", float("nan"))
    median_regret = pred_summary.get("median_decision_regret", float("nan"))
    p95_regret = pred_summary.get("p95_decision_regret", float("nan"))
    max_regret = pred_summary.get("max_decision_regret", float("nan"))

    mean_transfer_regret = float(all_transfer_regret.mean())
    p95_transfer_regret = float(np.percentile(all_transfer_regret, 95))
    frac_close = float((all_d_o < np.percentile(all_d_o, 25)).mean())
    most_actions_near_tied = np.mean([r["best_second_gap"] for r in rows]) < 0.005

    print("\n" + "=" * 60)
    print("STATE SUFFICIENCY")
    print("=" * 60)
    print(f"\nNumber of network states: {len(set(r['network_state_id'] for r in rows))}")
    print(f"Number of target sectors tested: {num_target_sectors}")
    print(f"Number of state-sector samples: {len(rows)}")

    print(f"\nMean decision regret: {mean_regret:.5f}" if has_supervised else "\nMean decision regret: n/a")
    print(f"Median decision regret: {median_regret:.5f}" if has_supervised else "Median decision regret: n/a")
    print(f"95th-percentile decision regret: {p95_regret:.5f}" if has_supervised else "95th-percentile decision regret: n/a")
    print(f"Maximum decision regret: {max_regret:.5f}" if has_supervised else "Maximum decision regret: n/a")

    print("\nepsilon-optimal accuracy:")
    for eps in EPSILONS:
        acc = pred_summary.get(f"eps_optimal_decision_accuracy_{eps}")
        print(f"  eps={eps}: {acc:.1%}" if acc is not None else f"  eps={eps}: n/a")

    print(f"\nNearest-state mean transfer regret: {mean_transfer_regret:.5f}")
    print(f"Nearest-state 95th-percentile transfer regret: {p95_transfer_regret:.5f}")

    if not geometry_consistent_for_all:
        print(f"\n{GEOMETRY_ISSUE_MESSAGE}")
    print("\nNOTE: regret figures above are CONDITIONAL on one matched replay per state, "
         "not an expected-value estimate.")

    if not has_supervised:
        conclusion = "INCONCLUSIVE"
        reason = "not enough states for a supervised train/test split"
    elif not geometry_consistent_for_all:
        conclusion = "INCONCLUSIVE"
        reason = "the 57-D geometry is not consistently defined for all 21 sector agents (see message above)"
    elif most_actions_near_tied:
        conclusion = "INCONCLUSIVE"
        reason = "most actions are nearly equivalent in global coverage (best-second gap is tiny)"
    elif frac_close < 0.05:
        conclusion = "INCONCLUSIVE"
        reason = "too few genuinely similar (standardized) local-state pairs in this sample"
    elif (mean_regret < 0.005 and p95_regret < 0.02
         and mean_transfer_regret < 0.005 and p95_transfer_regret < 0.02):
        conclusion = "PASS"
        reason = "the 57-D observation supports consistently low-regret unilateral decisions"
    elif (mean_regret > 0.02 or p95_regret > 0.05
         or mean_transfer_regret > 0.02 or p95_transfer_regret > 0.05):
        conclusion = "FAIL"
        reason = "similar 57-D observations repeatedly require materially different actions, causing real regret"
    else:
        conclusion = "INCONCLUSIVE"
        reason = "mixed evidence -- see the raw regret numbers above"

    print(f"\nConclusion:\n{conclusion}  ({reason})")


if __name__ == "__main__":
    main()
