"""Pipeline debug audit: does each stage of the environment (raw UE ->
grid -> RME -> region -> state/reward) produce physically meaningful,
sufficiently variable, decision-relevant information before it reaches the
RL agent? Built in 8 independent phases (see phaseN_* functions below),
run and verified ONE AT A TIME -- no changes to any existing production
code; this script only drives SimulationEngine's own public
attributes/methods (topology, engine.grid_points, engine.num_own_regions,
engine.regional_info_provider, engine.step_environment/drl_process_slot,
...), the same way tests/drl/test_rl_formulation_validation.py already does.

Debug scenario (shared by every phase): tilt_control_interval_s=30,
num_tilt_control_intervals=50 -- intervals 0-19 held at 0 deg (no tilt),
intervals 20-49 driven by RandomPolicy, so later phases can compare a
static baseline against actual tilt perturbation.

Visual convention (shared by every figure in this script):
  green solid  -- site (hex) boundary
  green dashed -- sector division within a site
  red          -- region_aligned region boundary + region ID -- each
                  region is a PARALLELOGRAM in the sector's own oblique
                  rhombus basis (a block_size x block_size pool of fine
                  grid cells), never a Cartesian square, and the same
                  region ID (0..num_own_regions-1) means the same relative
                  position in every sector's own grid -- comparable ACROSS
                  sectors, unlike the earlier per-sector-local design this
                  replaces. Regions from different sectors never overlap:
                  the 3 sectors' own rhombi exactly partition their site's
                  hexagon.
  dark blue dots -- rhombus fine-grid cell centers
  sector ID labels are the GLOBAL index (0..num_bs-1), never the
  per-site local 1/2/3 sionna's own HexGrid.show() would draw.

Run: python scripts/debug_pipeline_audit.py --phase 1

Saves each phase's figures to results/debug/phase<N>_<name>/.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "results", "debug")

SITE_SECTOR_COLOR = "green"
REGION_COLOR = "red"
GRID_COLOR = "darkblue"


def build_debug_engine() -> SimulationEngine:
    """The shared 50-interval/30s debug scenario -- policy_name is left at
    config.yaml's own value here; each phase's own driving loop decides
    interval-by-interval whether to hold 0 deg or call the policy (see
    module docstring), not this construction step."""
    cfg = load_config()
    cfg["simulation"]["tilt_control_interval_s"] = 30.0
    cfg["simulation"]["num_tilt_control_intervals"] = 50
    return SimulationEngine(cfg)


def draw_sites_and_sectors(ax, topo, sector_ids=None) -> None:
    """Green solid hex (site) boundaries + green dashed sector-division
    lines, drawn directly from HexGrid's own cell corners (NOT via its
    show(show_sectors=True), which also stamps a confusing per-site-local
    "1/2/3" label we don't want). Labels every sector in `sector_ids`
    (default: all) with its GLOBAL index, placed the same way sionna's own
    show() places its own labels (inside the wedge, near its outer edge) --
    just matched to the correct global sector by nearest boresight angle,
    not sionna's own local numbering.
    """
    if sector_ids is None:
        sector_ids = range(topo.num_bs)
    boresight = topo.bs_orientations[0, :, 0].cpu().numpy()
    n_per_site = topo.num_sectors_per_site
    sector_ids = set(sector_ids)

    for cell_idx, cell in topo.grid._grid.items():
        corners = cell.corners().cpu().numpy()
        center = cell.coord_euclid.cpu().numpy()

        # Site boundary: solid.
        ax.plot([corners[-1][0]] + [c[0] for c in corners],
               [corners[-1][1]] + [c[1] for c in corners],
               color=SITE_SECTOR_COLOR, linewidth=1.2, zorder=2)

        # Sector division lines: dashed, one per wedge corner (0, 2, 4).
        site_sectors = [s for s in range(cell_idx * n_per_site, (cell_idx + 1) * n_per_site)]
        for ii in (0, 2, 4):
            ax.plot([center[0], corners[ii][0]], [center[1], corners[ii][1]],
                   linestyle="--", color=SITE_SECTOR_COLOR, linewidth=1.0, zorder=2)
            # Match this wedge to whichever of the site's sectors points
            # closest to it (angle from center to the wedge's own far corner).
            wedge_angle = np.arctan2(corners[ii + 1][1] - center[1], corners[ii + 1][0] - center[0])
            angle_diff = np.abs(np.angle(np.exp(1j * (boresight[site_sectors] - wedge_angle))))
            sector = site_sectors[int(np.argmin(angle_diff))]
            if sector in sector_ids:
                label_xy = (center + corners[ii + 1]) / 2
                ax.annotate(str(sector), label_xy, fontsize=7, ha="center", va="center",
                           color="#0b0b0b", fontweight="bold", zorder=4)


def draw_regions(ax, engine, sector_ids, label: bool = True) -> None:
    """Red region-block boundaries (+ region ID) for each sector in
    `sector_ids` -- the SAME region ID means the same relative block in
    every sector's own oblique (alpha, beta) grid, so IDs are comparable
    ACROSS sectors, unlike the earlier per-sector-local tiling. Each
    region is a parallelogram spanning block_size x block_size fine cells,
    built directly from the rhombus grid's own basis vectors (private but
    stable CellularTopology attributes cached by build_sector_rhombus_grid,
    same access pattern as draw_sites_and_sectors's own topo.grid._grid).
    """
    topo = engine.topology
    bs_xy, e1, e2, n = (topo._sector_grid_bs_xy, topo._sector_grid_e1,
                        topo._sector_grid_e2, topo._sector_grid_n)
    block_size = engine.region_block_size
    blocks_per_side = n // block_size
    for s in sector_ids:
        for region_idx in range(engine.num_own_regions):
            block_row, block_col = divmod(region_idx, blocks_per_side)
            a_lo, a_hi = block_row * block_size / n, (block_row + 1) * block_size / n
            b_lo, b_hi = block_col * block_size / n, (block_col + 1) * block_size / n
            corners = np.array([
                bs_xy[s] + a_lo * e1[s] + b_lo * e2[s],
                bs_xy[s] + a_hi * e1[s] + b_lo * e2[s],
                bs_xy[s] + a_hi * e1[s] + b_hi * e2[s],
                bs_xy[s] + a_lo * e1[s] + b_hi * e2[s],
            ])
            ax.add_patch(mpatches.Polygon(corners, closed=True, facecolor="none",
                                          edgecolor=REGION_COLOR, linewidth=0.5, zorder=3))
            if label:
                ax.annotate(str(region_idx), corners.mean(axis=0), fontsize=4.5, ha="center",
                           va="center", color=REGION_COLOR, zorder=4)


def draw_grid_points(ax, engine, sector_ids=None) -> None:
    """Dark blue dots at every rhombus fine-grid cell center."""
    pts = engine.grid_points if sector_ids is None else engine.grid_points[list(sector_ids)]
    xy = pts.reshape(-1, 2)
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c=GRID_COLOR, alpha=0.5, linewidths=0, zorder=1)


def phase1_static_geometry(engine: SimulationEngine, out_dir: str) -> None:
    """One overview figure (all sites/sectors/regions/grid points, whole
    network) plus one small-multiples figure (one zoomed panel per sector:
    just that sector + its own side-sharing neighbors, same visual language)."""
    os.makedirs(out_dir, exist_ok=True)
    topo = engine.topology
    num_bs = topo.num_bs

    # Overview: every sector's own sites/sectors/regions/grid points at once.
    fig, ax = plt.subplots(figsize=(10, 9))
    draw_sites_and_sectors(ax, topo)
    draw_regions(ax, engine, range(num_bs))
    draw_grid_points(ax, engine)
    ax.set_title(f"Network overview: {num_bs} sectors -- green=site(solid)/sector(dashed), "
                f"red=region_aligned block+ID, blue=rhombus grid point", fontsize=10)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(out_dir, "overview_sites_sectors_regions_grid.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Small multiples: one zoomed panel per sector, own X_s neighborhood only.
    ncols = 7
    nrows = int(np.ceil(num_bs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    bs_xy = topo.bs_loc[0, :, :2].cpu().numpy()
    cell_radius = float(topo.grid.cell_radius.item())
    for s in range(num_bs):
        ax = axes[s // ncols, s % ncols]
        sector_set = [s] + engine.regional_info_provider.neighbor_sets[s]
        draw_sites_and_sectors(ax, topo, sector_ids=sector_set)
        draw_regions(ax, engine, sector_set)
        draw_grid_points(ax, engine, sector_ids=sector_set)
        # Own sector: filled red star, on top of everything else.
        ax.plot(*bs_xy[s], marker="*", markersize=16, color="#e34948", zorder=5)
        # Zoom to this sector's own X_s footprint (own + neighbors) with margin.
        member_xy = bs_xy[sector_set]
        pad = 1.3 * cell_radius
        ax.set_xlim(member_xy[:, 0].min() - pad, member_xy[:, 0].max() + pad)
        ax.set_ylim(member_xy[:, 1].min() - pad, member_xy[:, 1].max() + pad)
        ax.set_title(f"sector {s} + {len(sector_set) - 1} neighbors", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
    for s in range(num_bs, nrows * ncols):
        axes[s // ncols, s % ncols].axis("off")
    fig.suptitle("Per-sector neighborhood (* = own sector) -- same green/red/blue convention as overview",
                fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "per_sector_neighborhoods.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Phase 1 done -- saved to {out_dir}/")


PHASES = {
    1: ("static_geometry", phase1_static_geometry),
}


def main():
    # Parse which phase to run -- one at a time, per the debug protocol.
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True, choices=sorted(PHASES.keys()))
    args = parser.parse_args()

    # Build the shared debug scenario and dispatch to the requested phase.
    name, fn = PHASES[args.phase]
    out_dir = os.path.join(OUT_ROOT, f"phase{args.phase}_{name}")
    engine = build_debug_engine()
    fn(engine, out_dir)


if __name__ == "__main__":
    main()
