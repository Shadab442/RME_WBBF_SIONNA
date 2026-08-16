"""Shared I/O-adjacent helpers

Config loading: config.yaml is grouped by topic (topology, antenna, channel,
kpi, mobility, simulation, algorithms), plus one top-level section per
script that needs its own extra parameters (e.g. test_tilt_window_calibration).
load_config() returns the whole nested dict as-is -- callers index into the
topic they need (cfg["topology"]["scenario"]) or their own script section
(cfg["test_tilt_window_calibration"]["candidate_window_slots"]).

Visualization: Sionna's own sionna.sys.topology.HexGrid.show() already draws
the hexagon and sector geometry -- but it doesn't plot UE positions.

:param grid: A :class:`~sionna.sys.topology.HexGrid` -- either
    ``CellularTopology.grid`` or one from Sionna's own
    ``gen_hexgrid_topology(..., return_grid=True)``; both work identically here
    since only ``grid.show()`` is used.
:param ut_loc: UE positions, shape ``[num_ut, 2 or 3]`` (no batch dimension --
    index into a batch yourself before calling, e.g. ``ut_loc[0]``).
:param colors: optional per-UE color values (e.g. a group/cluster index),
    passed straight to ``scatter(..., c=colors, cmap="tab10")``.
"""

import math
import os

import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# Enough visually-distinct colors for any realistic num_site_colors * 3
# (sector-within-site) combinations -- a hex grid's site-adjacency graph
# rarely needs more than 3-4 colors even for many rings, so 4*3=12 covers it.
_SECTOR_COLOR_PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
    "navy", "darkgreen",
]


def compute_site_coloring(site_loc) -> np.ndarray:
    """Greedy graph coloring of sites by adjacency (nearest-neighbor
    spacing), so any two adjacent sites always get different color indices
    -- a proper map coloring, not an attempt at NUM_SITES globally-unique
    colors (which stops scaling once there are more sites than a palette
    has distinct entries; a coloring only ever needs as many colors as the
    adjacency graph's structure actually requires -- 3 for a single ring's
    wheel-shaped site graph, regardless of how many total sites there are
    further out).

    :param site_loc: [num_sites, 2] (x, y) site positions.
    :output: [num_sites] int array of 0-based color indices.
    """
    site_loc_np = site_loc.detach().cpu().numpy() if hasattr(site_loc, "detach") else np.asarray(site_loc)
    num_sites = site_loc_np.shape[0]
    dist = np.linalg.norm(site_loc_np[:, None, :] - site_loc_np[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    min_dist = dist.min()
    adjacent = dist <= min_dist * 1.05  # small tolerance around the true nearest-neighbor spacing

    colors = -np.ones(num_sites, dtype=int)
    # Welsh-Powell: color highest-degree sites first, tends to need fewer
    # total colors than a naive index-order pass.
    order = np.argsort(-adjacent.sum(axis=1))
    for site in order:
        used_by_neighbors = set(colors[adjacent[site]].tolist()) - {-1}
        c = 0
        while c in used_by_neighbors:
            c += 1
        colors[site] = c
    return colors


def compute_sector_index(xy, site_xy, num_sectors_per_site: int = 3) -> np.ndarray:
    """Which of a site's num_sectors_per_site angular wedges each point in
    xy falls into, purely by direction from site_xy -- matches
    CellularTopology's sector_yaws convention (boresights at 60/180/300 deg
    for the standard 3-sector case, so wedge 0 spans [0, 120) centered on
    60 deg, etc.), for a geometric "which sector is this near" grouping
    when there's no actual RF attachment computed (e.g. verify_mobility.py,
    which is pure geometry).

    :param xy: [N, 2] points.
    :param site_xy: [N, 2] each point's own site position (already matched
        to its nearest site by the caller).
    :output: [N] int array in [0, num_sectors_per_site).
    """
    xy_np = np.asarray(xy)
    site_xy_np = np.asarray(site_xy)
    delta = xy_np - site_xy_np
    angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2 * math.pi)
    wedge = 2 * math.pi / num_sectors_per_site
    return np.floor(angle / wedge).astype(int) % num_sectors_per_site


def compute_cell_colors(xy, site_loc, num_sectors_per_site: int = 3):
    """Per-point color, combining site coloring (adjacent sites always
    differ) with sector-within-site index (the num_sectors_per_site
    co-located sectors always differ from each other) -- so any two
    geometrically adjacent cells, whether they're different sectors of the
    SAME site or sectors of two neighboring sites, always get different
    colors, without needing to compute which specific sectors face each
    other across a site boundary.

    :param xy: [N, 2] points to color (e.g. cluster reference points).
    :param site_loc: [num_sites, 2] site positions.
    :output: (colors, nearest_site_idx) -- colors is a length-N list of
        matplotlib color strings; nearest_site_idx is [N] int array, e.g.
        for reuse in computing sector indices.
    """
    xy_np = np.asarray(xy)
    site_loc_np = site_loc.detach().cpu().numpy() if hasattr(site_loc, "detach") else np.asarray(site_loc)
    dist = np.linalg.norm(xy_np[:, None, :] - site_loc_np[None, :, :], axis=-1)
    nearest_site_idx = dist.argmin(axis=-1)

    site_color_idx = compute_site_coloring(site_loc_np)
    sector_idx = compute_sector_index(xy_np, site_loc_np[nearest_site_idx], num_sectors_per_site)
    combined_idx = site_color_idx[nearest_site_idx] * num_sectors_per_site + sector_idx
    colors = [_SECTOR_COLOR_PALETTE[i % len(_SECTOR_COLOR_PALETTE)] for i in combined_idx]
    return colors, nearest_site_idx


def add_cluster_ellipses(ax, cluster_centers, cluster_radius, **ellipse_kwargs):
    """Draws one circle (an Ellipse with equal width/height) per cluster
    center, radius ``cluster_radius`` -- the exact deviation_radius a
    ReferencePointGroupMobility cluster was built with, not a statistical
    fit to its (often just 2-3) member positions, which would be noisy and
    could misleadingly exclude/include members by chance. Deliberately
    plain black outlines regardless of site/sector -- see
    ``compute_cell_colors`` for coloring the UEs themselves instead.

    :param cluster_centers: [num_clusters, 2] (x, y) -- e.g. a
        ReferencePointGroupMobility's ``ref_xy``.
    :param cluster_radius: shared radius [m] for every cluster's circle.
    :output: list of the created Ellipse patches (e.g. to update their
        ``.center`` per animation frame).
    """
    style = dict(fill=False, edgecolor="black", linewidth=1.0, alpha=0.6)
    style.update(ellipse_kwargs)
    patches = []
    for cx, cy in cluster_centers:
        patch = Ellipse((float(cx), float(cy)), width=2 * cluster_radius, height=2 * cluster_radius, **style)
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def plot_scenario(grid, ut_loc=None, colors=None, show_sectors: bool = True,
                  cluster_centers=None, cluster_radius=None, **grid_kwargs):
    """Draws the hex grid (via HexGrid.show()), with UEs scattered on top if
    given, and one circle per cluster if ``cluster_centers``/
    ``cluster_radius`` are given (see ``add_cluster_ellipses``)."""
    fig = grid.show(show_sectors=show_sectors, **grid_kwargs)
    ax = fig.gca()
    if cluster_centers is not None:
        add_cluster_ellipses(ax, cluster_centers, cluster_radius)
    if ut_loc is not None:
        xy = ut_loc[:, :2].detach().cpu().numpy()
        # colors, if given, is a list of literal color specs (e.g. from
        # compute_cell_colors) -- no cmap involved.
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, marker="x", label=None if colors is not None else "UE")
        if colors is None:
            ax.legend()
    return fig


def save_scenario(path: str, grid, ut_loc=None, colors=None, dpi: int = 150, **kwargs) -> None:
    """Draws (via `plot_scenario`) and saves the figure to `path`, closing it
    afterward -- the save/close boilerplate every caller was repeating."""
    fig = plot_scenario(grid, ut_loc, colors, **kwargs)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


class _MobilityAnimationUpdater:
    """FuncAnimation callback (frame -> updated artists) for
    save_mobility_animation -- a class instead of a closure so per-frame
    state (scatter/ellipses/ax) lives as attributes, not captured locals."""

    def __init__(self, ax, scatter, ellipses, position_history, ref_xy_history,
                mobility_model, tilt_control_interval_s):
        self.ax = ax
        self.scatter = scatter
        self.ellipses = ellipses
        self.position_history = position_history
        self.ref_xy_history = ref_xy_history
        self.mobility_model = mobility_model
        self.tilt_control_interval_s = tilt_control_interval_s

    def __call__(self, frame):
        self.scatter.set_offsets(self.position_history[frame, :, :2])
        artists = [self.scatter]
        if self.ellipses is not None:
            for patch, (cx, cy) in zip(self.ellipses, self.ref_xy_history[frame]):
                patch.set_center((float(cx), float(cy)))
            artists.extend(self.ellipses)
        elapsed_s = frame * self.tilt_control_interval_s
        self.ax.set_title(f"{self.mobility_model}: tilt control interval {frame}, elapsed time {elapsed_s:g} s")
        return artists


def save_mobility_animation(cfg, out_dir, topology, position_history, ref_xy_history=None,
                            deviation_radius=None, member_group_idx=None):
    """Save the same style of position animation as verify_mobility.py.
    One frame per TILT CONTROL INTERVAL (each interval's first measurement
    draw), not per measurement draw -- 150,000 individual frames would make
    for an unusable animation; this keeps the frame count to
    num_tilt_control_intervals, each frame spanning tilt_control_interval_s.

    One circle per cluster (its exact deviation_radius, not a statistical
    fit to its few member positions) is drawn and updated from
    ref_xy_history if given (RPGM only). UEs are colored by their cluster's
    (site, sector-within-site) combination -- a proper map coloring, so any
    two geometrically adjacent cells (same site/different sector, or
    different but neighboring sites) always get different colors -- computed
    once from the initial cluster positions and held fixed for the whole
    animation, rather than per-point group coloring (NUM_GROUPS=60 is far
    more clusters than a colormap can distinguish) or recomputing every
    frame (which would make colors flicker as clusters cross boundaries).
    """
    mobility_model = cfg["mobility"]["mobility_model"]
    tilt_control_interval_s = cfg["simulation"]["tilt_control_interval_s"]
    num_tilt_control_intervals = cfg["simulation"]["num_tilt_control_intervals"]
    animation_fps = cfg["simulation"].get("animation_fps", 10)

    fig = topology.grid.show(show_sectors=True)
    ax = fig.gca()
    # grid.show()'s own layout leaves no room for a title added afterward;
    # FuncAnimation.save() bakes in a FIXED canvas (animations can't use
    # bbox_inches="tight" the way a static savefig can, since every frame
    # must share one frame size), so an unreserved title just gets clipped
    # by the top of that fixed canvas -- reserve the room explicitly.
    fig.subplots_adjust(top=0.90)
    xy0 = position_history[0, :, :2]

    colors = None
    if ref_xy_history is not None and member_group_idx is not None:
        cluster_colors, _ = compute_cell_colors(ref_xy_history[0], topology.site_loc, topology.num_sectors_per_site)
        colors = [cluster_colors[g] for g in member_group_idx.tolist()]
    scatter = ax.scatter(xy0[:, 0], xy0[:, 1], marker="x", color=colors if colors is not None else "tab:red")

    ellipses = None
    if ref_xy_history is not None:
        ellipses = add_cluster_ellipses(ax, ref_xy_history[0], deviation_radius)

    update = _MobilityAnimationUpdater(ax, scatter, ellipses, position_history, ref_xy_history,
                                       mobility_model, tilt_control_interval_s)
    animation = FuncAnimation(fig, update, frames=num_tilt_control_intervals, blit=False)
    path = os.path.join(out_dir, "mobility_animation.gif")
    animation.save(path, writer=PillowWriter(fps=animation_fps))
    plt.close(fig)
    print(f"Saved: {path}")
