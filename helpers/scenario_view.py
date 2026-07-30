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

import matplotlib.pyplot as plt


def plot_scenario(grid, ut_loc=None, colors=None, show_sectors: bool = True, **grid_kwargs):
    """Draws the hex grid (via HexGrid.show()), with UEs scattered on top if given."""
    fig = grid.show(show_sectors=show_sectors, **grid_kwargs)
    if ut_loc is not None:
        xy = ut_loc[:, :2].detach().cpu().numpy()
        ax = fig.gca()
        ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap="tab10" if colors is not None else None,
                  marker="x", label=None if colors is not None else "UE")
        if colors is None:
            ax.legend()
    return fig


def save_scenario(path: str, grid, ut_loc=None, colors=None, dpi: int = 150, **kwargs) -> None:
    """Draws (via `plot_scenario`) and saves the figure to `path`, closing it
    afterward -- the save/close boilerplate every caller was repeating."""
    fig = plot_scenario(grid, ut_loc, colors, **kwargs)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
