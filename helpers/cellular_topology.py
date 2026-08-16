# Hex-grid BS site/sector layout (a thing wrapper around sionna.sys.topology.HexGrid),
import math

import torch

from sionna.phy import PI
from sionna.sys.topology import HexGrid

NUM_SECTORS_PER_SITE = 3  # same 120-degree, 3-sector-per-site convention used throughout


class CellularTopology:
    """BS site/sector placement

    :ivar grid: the underlying :class:`~sionna.sys.topology.HexGrid`.
    :ivar bs_loc: [batch_size, num_bs, 3]. BS/sector positions [m].
    :ivar bs_orientations: [batch_size, num_bs, 3]. BS orientations
        [radian]; index 0 is yaw (sector boresight).
    :ivar site_loc: [num_cells, 2]. BS site (x, y) positions [m].
    :ivar num_cells: number of BS sites.
    :ivar num_bs: total sectors (``num_cells * num_sectors_per_site``).
    """

    def __init__(
        self,
        scenario_params,
        num_rings: int,
        num_sectors_per_site: int = NUM_SECTORS_PER_SITE,
        batch_size: int = 1,
        precision=None,
        device=None,
    ):
        """
        :param scenario_params: "isd", "bs_height", "min_bs_ut_dist" 
        :param num_rings: HexGrid rings (1 ring = 7 sites).
        :param num_sectors_per_site: sectors co-located at each site.
        :param batch_size: batch dimension for bs_loc/bs_orientations.
        """
        # Assertion check for 3 sector site
        assert num_sectors_per_site == 3, (
            "sector_yaws below hardcodes the standard 3-sector, 120-degree "
            "boresight convention (60/180/300 deg); a different "
            "num_sectors_per_site would need a different yaw formula."
        )

        # Read input parameters
        self.num_sectors_per_site = num_sectors_per_site
        self.batch_size = batch_size
        self.min_bs_ut_dist = scenario_params["min_bs_ut_dist"]
        isd = scenario_params["isd"]
        bs_height = scenario_params["bs_height"]

        # Create hexagonal cellular grid
        self.grid = HexGrid(
            isd=isd.item(),
            cell_height=bs_height.item(),
            num_rings=num_rings,
            precision=precision,
            device=device,
        )
        num_cells = self.grid.num_cells

        # num_sectors_per_site co-located sectors per site
        bs_loc = self.grid.cell_loc.repeat_interleave(num_sectors_per_site, dim=0)
        bs_loc = bs_loc.unsqueeze(0).expand(batch_size, -1, -1).clone()
        dtype = bs_loc.dtype

        # Standard 3-sector boresight yaws (60, 180, 300 deg)
        sector_yaws = torch.tensor([PI / 3.0, PI, 5.0 * PI / 3.0], dtype=dtype, device=bs_loc.device)
        bs_yaw = sector_yaws.repeat(num_cells)  # [num_bs]
        bs_yaw = bs_yaw.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)  # [batch, num_bs, 1]

        # Mechanical downtilt toward a nominal sector-center distance.
        # Vertical drop is to typical UE height, not all the way to ground.
        ut_height = scenario_params.get("min_ut_height", torch.zeros_like(bs_height))
        sector_center = (self.min_bs_ut_dist + 0.5 * isd) * 0.5
        bs_downtilt = torch.atan2(bs_height - ut_height, sector_center)
        num_bs = num_cells * num_sectors_per_site
        bs_pitch = torch.full((batch_size, num_bs, 1), bs_downtilt.item(), dtype=dtype, device=bs_loc.device)
        bs_roll = torch.zeros(batch_size, num_bs, 1, dtype=dtype, device=bs_loc.device)

        self.bs_loc = bs_loc
        self.bs_orientations = torch.cat([bs_yaw, bs_pitch, bs_roll], dim=-1)  # [batch, num_bs, 3]
        self.site_loc = bs_loc[0, ::num_sectors_per_site, :2]  # [num_cells, 2]
        self.num_cells = num_cells
        self.num_bs = num_bs

    @property
    def default_drop_radius(self) -> float:
        """Smallest disk radius that fully covers the hex-grid footprint
        (farthest site center + one hex radius)."""
        max_site_dist = torch.linalg.norm(self.site_loc, dim=-1).max()
        return float(max_site_dist.item() + self.grid.cell_radius.item())

    def is_within_coverage(self, xy: torch.Tensor) -> torch.Tensor:
        """True for points that lie within their nearest site's actual
        hexagon -- not just within the `default_drop_radius` disk,

        :param xy: [N, 2] candidate (x, y) positions [m].
        :output: [N] bool tensor.
        """
        site_loc = self.site_loc.to(dtype=xy.dtype, device=xy.device)
        cell_radius = self.grid.cell_radius.to(dtype=xy.dtype, device=xy.device)

        dist_to_sites = torch.linalg.norm(xy[:, None, :] - site_loc[None, :, :], dim=-1)
        nearest_dist, nearest_idx = dist_to_sites.min(dim=-1)

        # Hex boundary radius varies with angle: 
        # largest at vertices, smallest at edge midpoints.
        delta = xy - site_loc[nearest_idx]
        angle = torch.atan2(delta[:, 1], delta[:, 0])
        offset = torch.remainder(angle, math.pi / 3.0) - math.pi / 6.0
        hex_boundary = cell_radius * math.cos(math.pi / 6.0) / torch.cos(offset)
        return nearest_dist <= hex_boundary

    def mirror_bs_loc(self, ut_loc: torch.Tensor) -> torch.Tensor:
        """Wraparound topology: 
        use the nearest mirror of each BS site for every UE to avoid edge effects.

        :param ut_loc: [batch_size, num_ut, 3].
        :output bs_virtual_loc: [batch_size, num_bs, num_ut, 3]. Feeds
            directly into `channel_model.set_topology`'s
            ``bs_virtual_loc`` argument.
        """
        batch_size, num_ut, _ = ut_loc.shape
        # [num_cells, 7, 3]: each site's own position + its 6 mirror images.
        mirror_cell_loc = self.grid.mirror_cell_loc.to(dtype=ut_loc.dtype, device=ut_loc.device)

        # [batch, num_ut, num_cells, 7]
        dist = torch.norm(
            ut_loc[:, :, None, None, :] - mirror_cell_loc[None, None, :, :, :], dim=-1
        )
        closest_idx = dist.argmin(dim=-1, keepdim=True).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
        mirror_expand = mirror_cell_loc[None, None].expand(batch_size, num_ut, -1, -1, -1)
        # [batch, num_ut, num_cells, 3]
        virtual_loc = torch.gather(mirror_expand, dim=3, index=closest_idx).squeeze(3)

        # Co-located sectors at a site share the same virtual site position.
        virtual_loc = virtual_loc.repeat_interleave(self.num_sectors_per_site, dim=2)
        return virtual_loc.permute(0, 2, 1, 3)  # [batch, num_bs, num_ut, 3]
