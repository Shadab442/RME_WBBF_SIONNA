"""Hex-grid visualization with UE overlay.

Sionna's own sionna.sys.topology.HexGrid.show() already draws the hexagon and
sector geometry -- the only thing it doesn't do is plot UE positions. This
wraps an existing HexGrid (as returned by
sionna.sys.topology.gen_hexgrid_topology(..., return_grid=True)) and adds
that one missing piece, instead of redrawing the grid/sector geometry again.
"""

import matplotlib.pyplot as plt
import torch


class HexGridView:
    """A Sionna HexGrid plus the UE positions from the same topology draw.

    :param grid: A :class:`~sionna.sys.topology.HexGrid`, e.g. the one
        returned by ``gen_hexgrid_topology(..., return_grid=True)``.
    :param ut_loc: UE positions, shape ``[batch, num_ut, 3]``.
    """

    def __init__(self, grid, ut_loc: torch.Tensor):
        self.grid = grid
        self.ut_loc = ut_loc

    def show(self, show_sectors: bool = True, **kwargs):
        """Draws the hex grid (via HexGrid.show()) with UEs scattered on top."""
        fig = self.grid.show(show_sectors=show_sectors, **kwargs)
        ut_loc_np = self.ut_loc[0].detach().cpu().numpy()
        ax = fig.gca()
        ax.scatter(ut_loc_np[:, 0], ut_loc_np[:, 1], color="k", marker="x", label="UE")
        ax.legend()
        return fig

    def save(self, path: str, dpi: int = 150, **show_kwargs) -> None:
        """Draws (via `show`) and saves the figure to `path`, closing it
        afterward -- the save/close boilerplate every caller was repeating."""
        fig = self.show(**show_kwargs)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
