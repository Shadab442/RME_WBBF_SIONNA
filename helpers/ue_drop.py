"""UE-placement strategies, built on top of a CellularTopology.
"""

import math

import torch

from helpers.utils import get_logger

logger = get_logger(__name__)


class UeDropper:
    """simply stores the fixed context required for every UE draw.

    :ivar topology: a :class:`~helpers.cellular_topology.CellularTopology`.
    """

    def __init__(self, topology, dtype=None, device=None, generator=None):
        self.topology = topology
        self.dtype = dtype if dtype is not None else topology.bs_loc.dtype
        self.device = device if device is not None else topology.bs_loc.device
        self.generator = generator

    def disk_offset(self, radius: float, num_points: int, angle_center: float = None,
                    angle_half_width: float = None) -> torch.Tensor:
        """Uniform-random (x, y) offsets within a disk of the given radius --
        r=R*sqrt(U), so area density is uniform. theta is drawn over the full
        circle by default, or restricted to the wedge
        [angle_center - angle_half_width, angle_center + angle_half_width]
        [rad] if both are given.
        """

        logger.function("disk_offset start: radius=%.2f num_points=%d", radius, num_points)

        # Radius that makes the points uniformly distributed over the area of the disk
        r = radius * torch.sqrt(torch.rand(num_points, dtype=self.dtype, device=self.device, generator=self.generator))

        # Uniformly distributed angle -- independent draw from r, not reused,
        # or (r, theta) collapses onto a spiral instead of filling the disk
        u = torch.rand(num_points, dtype=self.dtype, device=self.device, generator=self.generator)
        if angle_center is None:  # over the entire circle
            theta = 2.0 * math.pi * u
        else:  # over the sector edge
            theta = angle_center + angle_half_width * (2.0 * u - 1.0)
        logger.debug("disk_offset: wedge_restricted=%s", angle_center is not None)

        # return disk offsets
        offsets = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
        logger.function("disk_offset end: shape=%s", tuple(offsets.shape))
        return offsets

    def valid_points(self, centers: torch.Tensor, radius: float, max_rounds: int = 100,
                     min_dist_from_site: float = None, angle_center: float = None,
                     angle_half_width: float = None) -> torch.Tensor:
        """For each row in centers [N, 2], samples a random disk offset such
        that centers[i] + offset lies within the topology's real hex-grid
        footprint, rejecting and resampling per-point as needed.
        """
        logger.function("valid_points start: num_points=%d radius=%.2f", centers.shape[0], radius)

        # Initialization
        xy = centers.clone()
        valid = torch.zeros(centers.shape[0], dtype=torch.bool, device=self.device)
        if min_dist_from_site is not None:
            site_loc = self.topology.site_loc.to(dtype=self.dtype, device=self.device)

        # Fill up with valid points
        round_idx = 0
        for round_idx in range(max_rounds):
            # Check for pending valid points
            if valid.all():
                break
            pending = (~valid).nonzero(as_tuple=True)[0]

            # Generate pending valid points
            offsets = self.disk_offset(radius, pending.shape[0], angle_center=angle_center,
                                       angle_half_width=angle_half_width)
            candidates = centers[pending] + offsets

            # Reject points outside network coverage footprint
            ok = self.topology.is_within_coverage(candidates)

            # Enforce min BS-UE distance
            if min_dist_from_site is not None:
                nearest_dist = torch.linalg.norm(
                    candidates[:, None, :] - site_loc[None, :, :], dim=-1
                ).min(dim=-1).values
                ok = ok & (nearest_dist >= min_dist_from_site)

            # Update the valid points
            xy[pending[ok]] = candidates[ok]
            valid[pending[ok]] = True

        logger.debug("valid_points: rounds_used=%d valid=%d/%d", round_idx + 1, int(valid.sum()), valid.numel())
        if not valid.all():
            logger.warning("valid_points: %d/%d points still invalid after max_rounds=%d",
                          int((~valid).sum()), valid.numel(), max_rounds)
        logger.function("valid_points end")
        return xy

    def uniform(self, num_ut: int, ut_height: float, disk_radius: float = None,
               batch_size: int = None) -> torch.Tensor:
        """Uniformly-at-random (x, y) UE drop, at a fixed height, over the
        hex-grid's coverage footprint.

        Candidates are drawn uniformly over a disk (``r = R*sqrt(U)``,
        ``theta = 2*pi*U``, so area density is uniform), then rejected and
        resampled unless both:
        - at least ``min_bs_ut_dist`` from every BS site (3GPP's minimum
          BS-UT distance), and
        - inside their nearest site's hexagon (so no UE lands in the disk's
          overshoot beyond actual coverage).

        :param disk_radius: circumscribing disk to draw candidates from;
            defaults to ``topology.default_drop_radius``.
        :param batch_size: if given, draws this many independent drops and
            stacks them into ``[batch_size, num_ut, 3]``
        :output ut_loc: ``[num_ut, 3]``, or ``[batch_size, num_ut, 3]`` if
            ``batch_size`` is given.
        """
        if batch_size is not None:
            return torch.stack([self.uniform(num_ut, ut_height, disk_radius) for _ in range(batch_size)], dim=0)

        logger.function("uniform start: num_ut=%d", num_ut)
        if disk_radius is None:
            disk_radius = self.topology.default_drop_radius

        site_loc = self.topology.site_loc.to(dtype=self.dtype, device=self.device)
        min_bs_ut_dist = self.topology.min_bs_ut_dist

        accepted = []
        n_accepted = 0
        max_rounds = 200
        round_idx = 0
        for round_idx in range(max_rounds):
            if n_accepted >= num_ut:
                break
            n_try = int((num_ut - n_accepted) * 1.5) + 8

            # Candidate generation
            r = disk_radius * torch.sqrt(torch.rand(n_try, dtype=self.dtype, device=self.device, generator=self.generator))
            theta = 2.0 * math.pi * torch.rand(n_try, dtype=self.dtype, device=self.device, generator=self.generator)
            candidates = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)  # [n_try, 2]

            # UE rejection condition
            nearest_dist = torch.linalg.norm(
                candidates[:, None, :] - site_loc[None, :, :], dim=-1
            ).min(dim=-1).values
            far_enough = nearest_dist >= min_bs_ut_dist
            within_coverage = self.topology.is_within_coverage(candidates)

            kept = candidates[far_enough & within_coverage]
            accepted.append(kept)
            n_accepted += kept.shape[0]
        logger.debug("uniform: rounds_used=%d n_accepted=%d/%d", round_idx + 1, n_accepted, num_ut)
        if round_idx + 1 >= 0.75 * max_rounds and n_accepted >= num_ut:
            logger.warning("uniform: took %d/%d rejection rounds to fill %d UEs -- disk_radius/min_bs_ut_dist "
                          "margin is getting tight", round_idx + 1, max_rounds, num_ut)

        # Repeat until enough UEs
        ut_loc_xy = torch.cat(accepted, dim=0)
        if ut_loc_xy.shape[0] < num_ut:
            logger.warning("uniform: only %d/%d valid UE positions after %d rejection rounds",
                          ut_loc_xy.shape[0], num_ut, max_rounds)
            raise RuntimeError(
                f"Could not sample {num_ut} valid UE positions after {max_rounds} "
                "rejection rounds -- disk_radius may be too small relative to "
                "min_bs_ut_dist."
            )
        ut_loc_xy = ut_loc_xy[:num_ut]
        z = torch.full((num_ut, 1), float(ut_height), dtype=self.dtype, device=self.device)
        result = torch.cat([ut_loc_xy, z], dim=-1)  # [num_ut, 3]
        logger.function("uniform end: result shape=%s", tuple(result.shape))
        return result

    def cluster_centers(self, num_groups: int, min_dist_from_site: float = None):
        """``num_groups`` cluster-center (x, y) positions, split as evenly as
        possible across the topology's sites, then across each site's sectors.

        :param min_dist_from_site: defaults to topology.min_bs_ut_dist --
            pass a larger, buffered value (e.g. min_bs_ut_dist + a cluster
            radius cap) to keep the WHOLE cluster disk centered here outside
            the exclusion zone, not just this one center point.
        """
        logger.function("cluster_centers start: num_groups=%d", num_groups)

        # Initialization
        topology = self.topology
        num_cells = topology.num_cells
        num_sectors_per_site = topology.num_sectors_per_site
        cell_radius = float(topology.grid.cell_radius.item())
        if min_dist_from_site is None:
            min_dist_from_site = topology.min_bs_ut_dist

        # Distribute clusters across sites
        base, extra = divmod(num_groups, num_cells)
        clusters_per_site = [base + 1 if site_idx < extra else base for site_idx in range(num_cells)]

        # Sector angular region
        wedge_half_width = math.pi / num_sectors_per_site

        # Cluster center generation
        start_xy_list = []
        for site_idx, n_clusters in enumerate(clusters_per_site):
            if n_clusters == 0:
                continue

            # Distribute a site's clusters across sectors
            sector_base, sector_extra = divmod(n_clusters, num_sectors_per_site)
            clusters_per_sector = [sector_base + 1 if k < sector_extra else sector_base
                                   for k in range(num_sectors_per_site)]

            # Obtain sector boresights
            sector_yaws = topology.bs_orientations[0, site_idx * num_sectors_per_site:
                                                   (site_idx + 1) * num_sectors_per_site, 0]
            site_center = topology.site_loc[site_idx]

            # Generate cluster centers
            for k, n_sector_clusters in enumerate(clusters_per_sector):
                if n_sector_clusters == 0:
                    continue
                centers = site_center.unsqueeze(0).expand(n_sector_clusters, -1)
                cluster_xy = self.valid_points(
                    centers, cell_radius,
                    angle_center=float(sector_yaws[k]), angle_half_width=wedge_half_width,
                    min_dist_from_site=min_dist_from_site,
                )
                start_xy_list.extend(tuple(xy) for xy in cluster_xy.tolist())
        logger.debug("cluster_centers: generated %d cluster centers", len(start_xy_list))
        logger.function("cluster_centers end")
        return start_xy_list

    def clustered(self, cluster_centers, members_per_cluster, deviation_radius: float, ut_height: float):
        """generates the actual UEs around the cluster centers.

        :param cluster_centers: [num_clusters, 2] (x, y) cluster center points.
        :param members_per_cluster: int (same for every cluster) or a
            length-``num_clusters`` list of per-cluster member counts.
        :param deviation_radius: max per-member offset from its cluster's
            center [m].
        :output: (ut_loc [num_ut, 3], member_cluster_idx [num_ut] long tensor
            giving which cluster each member belongs to, in the same order as
            ``cluster_centers``).
        """
        logger.function("clustered start: num_clusters=%d", len(cluster_centers))

        # Initialization
        centers = torch.as_tensor(cluster_centers, dtype=self.dtype, device=self.device)
        num_clusters = centers.shape[0]
        if isinstance(members_per_cluster, int):
            members_per_cluster = [members_per_cluster] * num_clusters

        # Cluster membership indices
        member_cluster_idx = torch.repeat_interleave(
            torch.arange(num_clusters, device=self.device),
            torch.tensor(members_per_cluster, device=self.device),
        )

        # Duplicate each cluster center for its members
        center_per_member = centers[member_cluster_idx]  # [num_ut, 2]

        # Generate valid UE locations
        xy = self.valid_points(center_per_member, deviation_radius, min_dist_from_site=self.topology.min_bs_ut_dist)
        z = torch.full((xy.shape[0], 1), float(ut_height), dtype=self.dtype, device=self.device)

        # Update UE locations
        ut_loc = torch.cat([xy, z], dim=-1)

        logger.debug("clustered: ut_loc shape=%s", tuple(ut_loc.shape))
        logger.function("clustered end")
        return ut_loc, member_cluster_idx
