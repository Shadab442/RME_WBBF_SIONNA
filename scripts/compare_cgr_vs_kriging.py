"""Head-to-head comparison of CGRRadioMapEstimator (algorithm/cgr.py)
against the existing OrdinaryKriging baseline (algorithm/kriging.py), over
`--num-intervals` (default 1000) intervals of an INDEPENDENT UNIFORM-RANDOM
per-sector electrical tilt each interval, scored against
scripts/debug_pipeline_audit.py's own compute_ground_truth_grid (reused
directly, not reimplemented) -- see that function's own docstring for the
"statistically representative, not sample-exact" caveat.

Fairness: both estimators run inside their OWN SpatialGridEstimator
instance, fed the exact SAME accumulate() inputs each interval from ONE
SimulationEngine/one real measurement stream -- not two separate engine
runs. This avoids any risk of the comparison being confounded by the two
estimators consuming randomness differently (CGR draws Gaussian-init
jitter from the global torch RNG every interval; Kriging draws none),
which could otherwise make two independently-run engines' own UE
mobility/channel realizations quietly diverge from each other.

Scoring: match rate (predicted.round() == ground_truth) at UNVISITED
cells only -- that is what actually distinguishes the two estimators
(visited cells are the real measurement, identical for both by
construction -- reported too, as a sanity check that should read ~equal).

Run: python scripts/compare_cgr_vs_kriging.py [--num-intervals 1000] [--seed 7]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from debug_pipeline_audit import compute_ground_truth_grid
from helpers.simulation_engine import SimulationEngine
from helpers.spatial_grid_estimator import SpatialGridEstimator
from helpers.utils import RepoLogging, get_logger, load_config

BLOCK_SIZE = 200


def build_engine(num_intervals):
    cfg = load_config()
    cfg["simulation"]["num_tilt_control_intervals"] = num_intervals
    return SimulationEngine(cfg), cfg


def build_estimators(engine, cfg):
    drl_cfg = cfg["algorithms"]["drl"]
    spatial_grid_cell_size_m = cfg["topology"]["spatial_grid_cell_size_m"]
    cgr_cfg = drl_cfg.get("cgr", {})
    common = dict(num_bs=engine.num_bs, num_grid_cells=engine.num_grid_cells, grid_points=engine.grid_points,
                 length_scale=spatial_grid_cell_size_m, coverage_threshold_db=engine.coverage_threshold_db,
                 radio_map_method="rsrp_per_sector", noise_power_w=engine.kpi_manager.noise_power_w,
                 # topology is needed regardless of radio_map_estimator now
                 # -- it's how SpatialGridEstimator builds estimable_cell_mask
                 # (cells inside min_bs_ut_dist of every site, where a real
                 # UE can never be dropped, are never Kriged/CGR-predicted).
                 topology=engine.topology)
    kriging_estimator = SpatialGridEstimator(**common, radio_map_estimator="kriging")
    cgr_estimator = SpatialGridEstimator(
        **common, radio_map_estimator="cgr",
        cgr_num_gaussian=cgr_cfg.get("num_gaussian", 12), cgr_ridge_lambda=cgr_cfg.get("ridge_lambda", 1e-4),
        cgr_baseline_method=cgr_cfg.get("baseline_method", "global_mean"),
        cgr_baseline_length_scale_m=cgr_cfg.get("baseline_length_scale_m", spatial_grid_cell_size_m),
        cgr_gaussian_config={k: v for k, v in cgr_cfg.items()
                             if k not in ("num_gaussian", "ridge_lambda", "baseline_method",
                                         "baseline_length_scale_m")})
    return kriging_estimator, cgr_estimator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-intervals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    RepoLogging.add_argument(parser)
    args = parser.parse_args()
    RepoLogging.configure(args.log_level, overrides=RepoLogging.parse_overrides(args.log_level_override))
    logger = get_logger(__name__)

    engine, cfg = build_engine(args.num_intervals)
    # A SEPARATE engine dedicated to compute_ground_truth_grid: Sionna's
    # own channel-model scenario refuses to change num_ut between
    # set_topology() calls without an explicit reset_topology() ("Cannot
    # change shape of '_ut_loc'... Call reset_topology() before changing
    # batch_size/num_ut/num_bs") -- `engine`'s own channel_model sees the
    # REAL 300-UE topology every interval, so calling compute_ground_truth_grid
    # (756 virtual "UEs", one per grid cell) on THAT SAME engine would hit
    # exactly that error on interval 2 onward. This engine's channel_model
    # instead NEVER sees a real-UE set_topology() call -- only ever the
    # SAME constant 756-virtual-point shape, every interval, which Sionna
    # is fine with repeating. Built from the identical cfg, so its
    # topology/grid_points/bs_loc are byte-identical to `engine`'s own
    # (CellularTopology construction is deterministic, no mobility/UE
    # randomness involved) -- the ground truth is for the SAME physical grid.
    ground_truth_engine, _ = build_engine(args.num_intervals)
    kriging_estimator, cgr_estimator = build_estimators(engine, cfg)
    rng = np.random.default_rng(args.seed)
    tilt_min, tilt_max = engine.tilt_deg_min, engine.tilt_deg_max
    num_bs = engine.num_bs
    closes = np.ones(num_bs, dtype=bool)

    match_rates = {"kriging": {"observed": [], "unobserved": []}, "cgr": {"observed": [], "unobserved": []}}

    for interval in range(args.num_intervals):
        live_tilt_deg = rng.uniform(tilt_min, tilt_max, size=num_bs)
        is_last = interval == args.num_intervals - 1
        ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
        state = engine.step_environment(is_last)
        sector_idx, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])

        drl_power_w = engine.kpi_manager.resolve_power_at_tilt(state, live_tilt_deg)[0]
        kpis = engine.kpi_manager.compute_ue_kpis(drl_power_w[None, :, :])
        covered_ue = kpis["sinr_db"][0] > engine.coverage_threshold_db
        power_db_all_sectors = 10.0 * np.log10(drl_power_w)

        ground_truth = compute_ground_truth_grid(ground_truth_engine, live_tilt_deg).reshape(-1)

        for name, estimator in (("kriging", kriging_estimator), ("cgr", cgr_estimator)):
            estimator.accumulate(sector_idx, cell_idx, covered_ue, power_db_all_sectors=power_db_all_sectors)
            observed = (estimator.visit_count > 0).reshape(-1).copy()
            _, coverage_per_cell, _ = estimator.compute(closes, current_tilts_deg=live_tilt_deg)
            predicted = coverage_per_cell.reshape(-1)

            for label, mask in (("observed", observed), ("unobserved", ~observed & (predicted >= 0))):
                if mask.sum() == 0:
                    continue
                match = float((predicted[mask].round() == ground_truth[mask]).mean())
                match_rates[name][label].append(match)

        if (interval + 1) % BLOCK_SIZE == 0 or interval == args.num_intervals - 1:
            block_summary = {name: {label: float(np.mean(vals[-BLOCK_SIZE:])) if vals else float("nan")
                                    for label, vals in scores.items()}
                             for name, scores in match_rates.items()}
            logger.info("interval %4d: kriging unobserved=%.4f observed=%.4f | cgr unobserved=%.4f observed=%.4f",
                       interval + 1, block_summary["kriging"]["unobserved"], block_summary["kriging"]["observed"],
                       block_summary["cgr"]["unobserved"], block_summary["cgr"]["observed"])

    logger.info("=== FINAL (mean over %d intervals) ===", args.num_intervals)
    for name in ("kriging", "cgr"):
        unobs = match_rates[name]["unobserved"]
        obs = match_rates[name]["observed"]
        logger.info("%-7s: unobserved match=%.4f (n_intervals=%d) | observed match=%.4f (n_intervals=%d)",
                   name, float(np.mean(unobs)) if unobs else float("nan"), len(unobs),
                   float(np.mean(obs)) if obs else float("nan"), len(obs))


if __name__ == "__main__":
    main()
