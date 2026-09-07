"""Spatial grid resolution validation for the WBBF e-tilt RL state --
NOT an RL script: no policy, no training, no learning of any kind.

Question: does aggregating the fine (25m-target) per-cell coverage grid into
coarser 50m/100m cells discard decision-relevant information? Answers it
empirically, on the real channel model, rather than through a closed-form
proxy (an antenna-gain-only radial argument was tried and rejected earlier
in this project's design discussion for exactly the reasons this script
avoids: it ignored path loss, ignored shadow fading, and used a 1D radial
sweep that doesn't match the real 2D oblique grid).

Method
------
For ISD=500m, CellularTopology.build_sector_rhombus_grid gives
side_length = cell_radius = ISD/sqrt(3) ~= 288.68m (see the derivation this
followed from in the project discussion). Reference resolution: target 25m
-> n=12 -> 144 cells/sector (~24.1m actual side). At every one of those 144
cell centers, for ONE representative sector under test, place a single UE
and draw MANY independent channel realizations (fresh pathloss + shadow
fading each draw, same location) while sweeping that sector's own tilt
through every candidate in downtilt_sweep_deg -- ALL OTHER sectors held at
a fixed baseline tilt throughout, consistent with the async design (a
sector's own action effect shouldn't be confounded by a simultaneously
-changing neighbor). This directly reuses KpiManager.resolve_power_at_tilt
(pure function of (channel state, tilt), pathloss/shadow-fading/mobility
are never re-drawn by it) and KpiManager.compute_ue_kpis's own "coverage"
field -- with exactly one UE repeated across the batch (realization)
dimension, compute_ue_kpis's coverage output IS the Monte Carlo estimate of
p_g(theta) = P[SINR >= threshold | location g, tilt theta] directly, no
separate estimator needed.

The resulting 12x12xnum_actions probability cube is then block-averaged
(2x2 -> 6x6/"50m", 4x4 -> 3x3/"100m") -- NOT re-simulated at coarser
resolution -- so all three resolutions are compared against the exact same
underlying data, isolating the aggregation question from simulation noise
between separate runs.

Assumptions (see also each console section)
---------------------------------------------
- Serving-sector/attachment criterion: whatever KpiManager currently
  implements (max-SINR, as of this script's writing) -- this diagnostic is
  about spatial aggregation, not attachment rule, and the conclusion
  doesn't depend on which attachment rule is active.
- Tested on ONE representative sector (SECTOR_UNDER_TEST below), not all
  21 -- the underlying UMa channel/array physics is identical up to
  position/rotation for every sector, so one sector is representative;
  re-run with a different index to spot-check another.
- Each p_g(theta) is a Monte Carlo estimate from NUM_REALIZATIONS
  independent draws, so it carries its own sampling error
  (~sqrt(p(1-p)/N)) -- the console summary reports this alongside the
  cross-resolution RMSE so the two aren't confused.
- Neighboring sectors' tilts are held fixed at BASELINE_TILT_DEG for every
  candidate action tested on the sector under test.

Run: python tests/drl/test_spatial_grid_resolution.py [--num-realizations 300]
    [--sector 0] [--seed 0]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sionna
import torch

from helpers.simulation_engine import SimulationEngine
from helpers.utils import RepoLogging, get_logger, load_config

RepoLogging.configure("INFO")
logger = get_logger(__name__)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "spatial_grid_resolution")
FINE_CELL_SIZE_M = 25.0
BASELINE_TILT_DEG = None  # set to downtilt_sweep_deg[0] once engine is built


def build_engine(cfg):
    """Reuses SimulationEngine's own construction (topology, UMa channel
    model, antenna arrays, KpiManager) unmodified. policy_name forced to
    "random" purely to skip constructing an unused DQN network -- this
    script never calls act()/observe()."""
    cfg["algorithms"]["drl"]["policy_name"] = "random"
    return SimulationEngine(cfg)


def generate_realizations(engine, ut_xy, ut_height, num_realizations):
    """ONE test UE, fixed at ut_xy, repeated across a `num_realizations`
    batch dimension -- Sionna draws an independent pathloss+shadow-fading
    realization per batch entry despite the identical position, since it's
    a stochastic model. Reuses engine.channel_model.set_topology /
    engine.large_scale_channel.generate_state directly -- the same
    production code path, just with a synthetic single-UE batch instead of
    the real UE population.
    """
    dtype = engine.topology.bs_loc.dtype
    device = engine.topology.bs_loc.device

    ut_loc = torch.tensor([[ut_xy[0], ut_xy[1], ut_height]], dtype=dtype, device=device)
    ut_loc = ut_loc.unsqueeze(0).expand(num_realizations, 1, 3).clone()  # [batch, 1, 3]
    ut_orientations = torch.zeros(num_realizations, 1, 3, dtype=dtype, device=device)
    ut_velocities = torch.zeros_like(ut_orientations)
    in_state = torch.zeros(num_realizations, 1, dtype=torch.bool, device=device)

    # BS geometry doesn't vary per realization -- broadcast from batch=1.
    bs_loc = engine.topology.bs_loc[:1].expand(num_realizations, -1, -1)
    bs_orientations = engine.topology.bs_orientations[:1].expand(num_realizations, -1, -1)
    bs_virtual_loc = engine.topology.mirror_bs_loc(ut_loc)

    engine.channel_model.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                                      ut_velocities, in_state, None, bs_virtual_loc)
    return engine.large_scale_channel.generate_state()


def block_average(grid, factor):
    """[n, n, num_actions] -> [n/factor, n/factor, num_actions], averaging
    non-overlapping factor x factor blocks -- the aggregation being
    validated (NOT a re-simulation at coarser resolution)."""
    n = grid.shape[0]
    m = n // factor
    return grid.reshape(m, factor, m, factor, -1).mean(axis=(1, 3))


def broadcast_up(coarse_grid, factor):
    """Inverse of block_average's shape -- repeats each coarse cell back
    over the factor x factor fine cells it was built from, for a fine-vs-
    aggregate elementwise comparison."""
    return np.repeat(np.repeat(coarse_grid, factor, axis=0), factor, axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-realizations", type=int, default=300,
                        help="Independent channel draws per (location, tilt) -- sets the "
                             "Monte Carlo standard error of each p_g(theta) estimate.")
    parser.add_argument("--sector", type=int, default=0, help="Sector under test.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    sionna.phy.config.precision = "single"
    sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = load_config()
    sionna.phy.config.seed = args.seed
    torch.manual_seed(args.seed)
    engine = build_engine(cfg)

    downtilt_sweep_deg = engine.downtilt_sweep_deg
    num_actions = len(downtilt_sweep_deg)
    baseline_tilt = np.full(engine.num_bs, downtilt_sweep_deg[0])
    sector = args.sector
    ut_height = cfg["topology"]["ut_height"]

    logger.info("spatial_grid_resolution: sector=%d num_realizations=%d num_actions=%d",
               sector, args.num_realizations, num_actions)

    # Reference (fine) grid: target 25m.
    grid_points_fine, n_fine = engine.topology.build_sector_rhombus_grid(FINE_CELL_SIZE_M)
    actual_side_fine = float(engine.topology.grid.cell_radius.item()) / n_fine
    logger.info("Fine grid: n=%d (%d cells), actual side=%.2fm", n_fine, n_fine * n_fine, actual_side_fine)

    fine_locations = grid_points_fine[sector]  # [n_fine*n_fine, 2]
    p_flat = np.zeros((n_fine * n_fine, num_actions))

    rows = []
    for loc_idx in range(n_fine * n_fine):
        state = generate_realizations(engine, fine_locations[loc_idx], ut_height, args.num_realizations)
        for action_idx, candidate in enumerate(downtilt_sweep_deg):
            tilt_deg_per_sector = baseline_tilt.copy()
            tilt_deg_per_sector[sector] = candidate
            power = engine.kpi_manager.resolve_power_at_tilt(state, tilt_deg_per_sector)
            kpis = engine.kpi_manager.compute_ue_kpis(power, threshold_db=engine.coverage_threshold_db)
            p_flat[loc_idx, action_idx] = kpis["coverage"]
            rows.append({
                "loc_idx": loc_idx, "x": float(fine_locations[loc_idx, 0]), "y": float(fine_locations[loc_idx, 1]),
                "action": action_idx, "tilt_deg": candidate, "p_coverage": kpis["coverage"],
            })
        if loc_idx % 20 == 0:
            logger.info("location %3d/%d done", loc_idx, n_fine * n_fine)

    p_fine = p_flat.reshape(n_fine, n_fine, num_actions)  # row-major (alpha, beta), matches grid construction

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "spatial_grid_resolution.csv")
    df.to_csv(csv_path, index=False)

    # Aggregate: same probability cube, block-averaged -- not re-simulated.
    resolutions = {
        "25m (reference)": (p_fine, n_fine, 1),
        "50m": (block_average(p_fine, 2), n_fine // 2, 2),
        "100m": (block_average(p_fine, 4), n_fine // 4, 4),
    }

    summary_rows = []
    best_tilt_fine = p_fine.argmax(axis=-1)
    mean_p = p_fine.mean()
    mc_se = float(np.sqrt(mean_p * (1 - mean_p) / args.num_realizations))

    for label, (p_grid, n, factor) in resolutions.items():
        if factor == 1:
            rmse, max_err, agreement = 0.0, 0.0, 100.0
        else:
            p_broadcast = broadcast_up(p_grid, factor)
            diff = p_fine - p_broadcast
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            max_err = float(np.max(np.abs(diff)))
            best_tilt_agg = p_grid.argmax(axis=-1)
            best_tilt_agg_broadcast = broadcast_up(best_tilt_agg[:, :, None], factor)[:, :, 0]
            agreement = float(np.mean(best_tilt_fine == best_tilt_agg_broadcast)) * 100.0
        actual_side = actual_side_fine * factor
        summary_rows.append({
            "grid": label, "n_per_side": n, "num_cells": n * n, "actual_side_m": round(actual_side, 1),
            "coverage_rmse": round(rmse, 4), "coverage_max_abs_err": round(max_err, 4),
            "best_tilt_agreement_pct": round(agreement, 1),
        })

    best_tilt_50m = resolutions["50m"][0].argmax(axis=-1)
    best_tilt_100m = resolutions["100m"][0].argmax(axis=-1)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUT_DIR, "spatial_grid_resolution_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    np.savez(os.path.join(OUT_DIR, "spatial_grid_resolution.npz"),
            p_fine=p_fine, p_50m=resolutions["50m"][0], p_100m=resolutions["100m"][0],
            best_tilt_25m=best_tilt_fine, best_tilt_50m=best_tilt_50m, best_tilt_100m=best_tilt_100m,
            downtilt_sweep_deg=downtilt_sweep_deg, sector=sector, num_realizations=args.num_realizations,
            mc_standard_error=mc_se)

    make_heatmaps(p_fine, resolutions, downtilt_sweep_deg)
    make_best_tilt_heatmaps(best_tilt_fine, best_tilt_50m, best_tilt_100m, downtilt_sweep_deg)

    print("\n" + "=" * 78)
    print("SPATIAL GRID RESOLUTION VALIDATION -- SUMMARY")
    print("=" * 78)
    print(f"Sector under test: {sector}   Realizations/point: {args.num_realizations}   "
         f"Neighbor/other-sector tilt held at: {downtilt_sweep_deg[0]:g} deg")
    print(f"Monte Carlo standard error of each p_g(theta) estimate (at mean p={mean_p:.3f}): "
         f"~{mc_se:.4f}  -- cross-resolution RMSE below this is not distinguishable from sampling noise.")
    print()
    print(summary_df.to_string(index=False))
    print("=" * 78)
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_csv_path}")
    print(f"Saved: {os.path.join(OUT_DIR, 'spatial_grid_resolution.npz')}")
    print(f"Heatmaps saved under: {OUT_DIR}")


def make_heatmaps(p_fine, resolutions, downtilt_sweep_deg):
    num_actions = p_fine.shape[-1]
    for action_idx, tilt in enumerate(downtilt_sweep_deg):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, (label, (p_grid, n, factor)) in zip(axes, resolutions.items()):
            im = ax.imshow(p_grid[:, :, action_idx], vmin=0, vmax=1, cmap="viridis")
            ax.set_title(f"{label} ({n}x{n})")
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"p_g(theta={tilt:g} deg) across resolutions")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"heatmap_tilt_{tilt:g}deg.png"), dpi=120)
        plt.close(fig)
    logger.info("Saved %d per-tilt heatmap figures", num_actions)


def make_best_tilt_heatmaps(best_tilt_25m, best_tilt_50m, best_tilt_100m, downtilt_sweep_deg):
    """Which tilt WINS at each location, per resolution -- distinct from
    make_heatmaps' per-tilt coverage maps. A discrete colormap with one
    color per candidate tilt (labeled in degrees, not action index), so
    where fine-vs-aggregate best-tilt disagreements concentrate is visible
    directly, not just their overall percentage.
    """
    num_actions = len(downtilt_sweep_deg)
    cmap = plt.get_cmap("tab10", num_actions)
    maps = {"25m (reference)": best_tilt_25m, "50m": best_tilt_50m, "100m": best_tilt_100m}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (label, best_tilt) in zip(axes, maps.items()):
        im = ax.imshow(best_tilt, cmap=cmap, vmin=-0.5, vmax=num_actions - 0.5)
        ax.set_title(f"{label} ({best_tilt.shape[0]}x{best_tilt.shape[1]})")
    cbar = fig.colorbar(im, ax=axes, ticks=range(num_actions), fraction=0.025, pad=0.02)
    cbar.ax.set_yticklabels([f"{t:g} deg" for t in downtilt_sweep_deg])
    fig.suptitle("Best tilt (argmax_theta p_g(theta)) per location, by resolution")
    fig.savefig(os.path.join(OUT_DIR, "heatmap_best_tilt.png"), dpi=120)
    plt.close(fig)
    logger.info("Saved best-tilt heatmap figure")


if __name__ == "__main__":
    main()
