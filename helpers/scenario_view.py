"""Combined topology + UE visualization.

Sionna's own sionna.sys.topology.HexGrid.show() already draws the hexagon and
sector geometry -- the only thing it doesn't do is plot UE positions. This adds
that one missing piece on top, instead of redrawing the grid/sector geometry.

Deliberately plain functions, not a class living on CellularTopology or a UE-drop
type: this is the one place that legitimately depends on both a grid (topology)
and UE positions (drop/mobility) at once, so it's kept separate from both rather
than either absorbing a piece of the other's job.

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
