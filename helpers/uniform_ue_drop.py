# Hex-grid BS layout (site placement via Sionna's own sionna.sys.topology.HexGrid)
# with UEs dropped uniformly at random over the coverage area, instead of
# HexGrid.call's per-sector wedge (a fixed count per sector is a hard
# constraint of that method, not something its parameters can turn off).
#
# Copied from sionna-sls/functions/uniform_ue_drop.py (a live cross-repo
# import was tried first and rejected: both repos have a top-level package
# named `functions`, and sionna-sls's functions/__init__.py does an absolute
# self-import that breaks as soon as any other `functions` package is loaded
# in the same process -- not worth patching around for one file). One
# adaptation from the original: uses Sionna's own HexGrid directly
# (sionna.sys.topology) instead of this repo's now-deleted local wrapper --
# confirmed the properties this file needs (cell_loc, num_cells, cell_radius,
# mirror_cell_loc) all exist on it, so this also avoids chaining into any of
# sionna-sls's other files.

import math

import torch

from sionna.phy import PI
from sionna.sys.topology import HexGrid

NUM_SECTORS_PER_SITE = 3  # same 120-degree, 3-sector-per-site convention used throughout


class UniformDropTopology:
    """BS site/sector placement (in `__init__`) plus on-demand
    uniform-random UE dropping via `sample_ut_loc()`.

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
        :param scenario_params: dict with "isd", "bs_height", and
            "min_bs_ut_dist" torch scalars (subset of
            set_3gpp_scenario_parameters's output).
        :param num_rings: HexGrid rings (1 ring = 7 sites).
        :param num_sectors_per_site: sectors co-located at each site.
        :param batch_size: batch dimension for bs_loc/bs_orientations.
        """
        assert num_sectors_per_site == 3, (
            "sector_yaws below hardcodes the standard 3-sector, 120-degree "
            "boresight convention (60/180/300 deg); a different "
            "num_sectors_per_site would need a different yaw formula."
        )
        self.num_sectors_per_site = num_sectors_per_site
        self.batch_size = batch_size
        self.min_bs_ut_dist = scenario_params["min_bs_ut_dist"]

        isd = scenario_params["isd"]
        bs_height = scenario_params["bs_height"]

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

    def sample_ut_loc(
        self,
        num_ut: int,
        ut_height: float,
        dtype,
        device,
        disk_radius: float = None,
        generator=None,
        batch_size: int = None,
    ) -> torch.Tensor:
        """Uniformly-at-random (x, y) UE drop, at a fixed height, over the
        hex-grid's actual coverage footprint -- the union of every site's
        hexagon, not just a circumscribing disk (which bulges past the
        hexagon edges in the gaps between cells).

        Candidates are drawn uniformly over a disk (``r = R*sqrt(U)``,
        ``theta = 2*pi*U``, so area density is uniform), then rejected and
        resampled unless both:
        - at least ``min_bs_ut_dist`` from every BS site (3GPP's minimum
          BS-UT distance), and
        - inside their nearest site's hexagon (so no UE lands in the disk's
          overshoot beyond actual coverage).

        :param disk_radius: circumscribing disk to draw candidates from;
            defaults to `self.default_drop_radius`.
        :param batch_size: if given, draws this many independent drops and
            stacks them into ``[batch_size, num_ut, 3]`` -- e.g. for a
            common-random-numbers study where several independent
            (topology, channel) realizations are evaluated as one batch.
            If `None` (default), returns a single drop ``[num_ut, 3]``.
        :output ut_loc: ``[num_ut, 3]``, or ``[batch_size, num_ut, 3]`` if
            ``batch_size`` is given.
        """
        if batch_size is not None:
            return torch.stack(
                [
                    self.sample_ut_loc(num_ut, ut_height, dtype, device, disk_radius, generator)
                    for _ in range(batch_size)
                ],
                dim=0,
            )
        if disk_radius is None:
            disk_radius = self.default_drop_radius

        site_loc = self.site_loc.to(dtype=dtype, device=device)
        min_bs_ut_dist = self.min_bs_ut_dist
        cell_radius = self.grid.cell_radius.to(dtype=dtype, device=device)

        accepted = []
        n_accepted = 0
        max_rounds = 200
        for _ in range(max_rounds):
            if n_accepted >= num_ut:
                break
            n_try = int((num_ut - n_accepted) * 1.5) + 8
            r = disk_radius * torch.sqrt(torch.rand(n_try, dtype=dtype, device=device, generator=generator))
            theta = 2.0 * math.pi * torch.rand(n_try, dtype=dtype, device=device, generator=generator)
            candidates = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)  # [n_try, 2]

            dist_to_sites = torch.linalg.norm(
                candidates[:, None, :] - site_loc[None, :, :], dim=-1
            )  # [n_try, num_cells]
            nearest_dist, nearest_idx = dist_to_sites.min(dim=-1)
            far_enough = nearest_dist >= min_bs_ut_dist

            # Inside the nearest site's hexagon: a hexagon's boundary
            # distance varies with angle (closer at flat edges, farther at
            # corners), unlike a circle. Sionna's own Hexagon.corners() places
            # vertices at angle = k*60deg (0, 60, 120, ...), so the boundary
            # is at its maximum (= cell_radius, the circumradius) exactly at
            # those angles and at its minimum (apothem) at the midpoints
            # (30, 90, 150, ...) -- offset=0 must land on the midpoints for
            # the cos(offset) formula below to peak at the vertices.
            delta = candidates - site_loc[nearest_idx]
            angle = torch.atan2(delta[:, 1], delta[:, 0])
            offset = torch.remainder(angle, math.pi / 3.0) - math.pi / 6.0
            hex_boundary = cell_radius * math.cos(math.pi / 6.0) / torch.cos(offset)
            within_coverage = nearest_dist <= hex_boundary

            kept = candidates[far_enough & within_coverage]
            accepted.append(kept)
            n_accepted += kept.shape[0]

        ut_loc_xy = torch.cat(accepted, dim=0)
        if ut_loc_xy.shape[0] < num_ut:
            raise RuntimeError(
                f"Could not sample {num_ut} valid UE positions after {max_rounds} "
                "rejection rounds -- disk_radius may be too small relative to "
                "min_bs_ut_dist."
            )
        ut_loc_xy = ut_loc_xy[:num_ut]
        z = torch.full((num_ut, 1), float(ut_height), dtype=dtype, device=device)
        return torch.cat([ut_loc_xy, z], dim=-1)  # [num_ut, 3]

    def mirror_bs_loc(self, ut_loc: torch.Tensor) -> torch.Tensor:
        """Wraparound-corrected BS positions per UE:
        for each site, picks whichever of its 6 mirror images
        or itself is closest to a given UE, so edge UEs see interference
        as if the grid tiled infinitely instead of stopping at the drop
        boundary.

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
