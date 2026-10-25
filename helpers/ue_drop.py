"""UE-placement strategies, built on top of a CellularTopology.
"""

import math

import numpy as np
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

    def ellipse_offset(self, radial_radius: float, tangential_radius: float, num_points: int,
                       radial_dir: torch.Tensor) -> torch.Tensor:
        """Uniform-random (x, y) offsets within an ellipse whose semi-axes
        are radial_radius (along radial_dir, PER-POINT) and
        tangential_radius (perpendicular to it) -- same area-uniform
        r=sqrt(U) trick as disk_offset, just affinely stretched into two
        independent axes instead of one shared radius. Exists so a
        cluster's member scatter can be compressed along the radial
        (site-distance) direction specifically -- the only direction
        hotspot_region_membership's near/mid/far cut actually cares about
        -- while staying as wide as before tangentially, rather than
        shrinking the scatter isotropically and losing area in a direction
        that was never the problem.

        :param radial_dir: [num_points, 2] unit vectors, one per point --
            typically (ref_point - its own home site) normalized, since
            near/mid/far membership is purely radial distance from the
            site, not a fixed sector-wide direction.
        """
        logger.function("ellipse_offset start: radial_radius=%.2f tangential_radius=%.2f num_points=%d",
                        radial_radius, tangential_radius, num_points)
        r = torch.sqrt(torch.rand(num_points, dtype=self.dtype, device=self.device, generator=self.generator))
        theta = 2.0 * math.pi * torch.rand(num_points, dtype=self.dtype, device=self.device, generator=self.generator)
        radial_component = radial_radius * r * torch.cos(theta)
        tangential_component = tangential_radius * r * torch.sin(theta)
        tangential_dir = torch.stack([-radial_dir[:, 1], radial_dir[:, 0]], dim=-1)
        offsets = radial_component[:, None] * radial_dir + tangential_component[:, None] * tangential_dir
        logger.function("ellipse_offset end: shape=%s", tuple(offsets.shape))
        return offsets

    def valid_points(self, centers: torch.Tensor, radius: float, max_rounds: int = 100,
                     min_dist_from_site: float = None, angle_center: float = None,
                     angle_half_width: float = None, radial_dir: torch.Tensor = None,
                     tangential_radius: float = None) -> torch.Tensor:
        """For each row in centers [N, 2], samples a random disk offset such
        that centers[i] + offset lies within the topology's real hex-grid
        footprint, rejecting and resampling per-point as needed.

        :param radial_dir: [N, 2] unit vectors, one per row of centers --
            when given, samples an ellipse_offset (radius=radial semi-axis,
            tangential_radius=the other) instead of an isotropic
            disk_offset; mutually exclusive with angle_center/
            angle_half_width (unused when radial_dir is given). None
            (default): the original isotropic disk_offset behavior,
            unchanged for every other caller.
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
            if radial_dir is not None:
                offsets = self.ellipse_offset(radius, tangential_radius, pending.shape[0], radial_dir[pending])
            else:
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

    def walk_points(self, current_xy: torch.Tensor, ref_centers: torch.Tensor, deviation_radius: float,
                    step_radius: float, max_rounds: int = 100, min_dist_from_site: float = None,
                    radial_dir: torch.Tensor = None, tangential_radius: float = None) -> torch.Tensor:
        """Persistent per-point random walk, one step: perturbs current_xy[i]
        by a small disk_offset(step_radius) draw, rejecting and resampling
        just the STEP (not the whole position, unlike valid_points, which
        always starts fresh from centers) if the result leaves the
        deviation_radius disk around ref_centers[i], the network coverage
        footprint, or violates min_dist_from_site. A point that still has no
        valid step after max_rounds simply keeps its previous position for
        this step, rather than teleporting -- the previous position was
        already valid, so standing still is always a safe fallback.

        :param radial_dir: [N, 2] unit vectors, one per point -- when
            given, the CONTAINMENT check (not the small step itself, which
            stays an isotropic disk_offset(step_radius) -- too small a
            perturbation for its own shape to matter) becomes elliptical:
            deviation_radius is then the radial semi-axis, tangential_radius
            the other. None (default): the original circular containment
            check, unchanged for every other caller.
        """
        logger.function("walk_points start: num_points=%d step_radius=%.2f", current_xy.shape[0], step_radius)

        # Initialization
        xy = current_xy.clone()
        valid = torch.zeros(current_xy.shape[0], dtype=torch.bool, device=self.device)
        if min_dist_from_site is not None:
            site_loc = self.topology.site_loc.to(dtype=self.dtype, device=self.device)

        # Fill up with valid steps
        round_idx = 0
        for round_idx in range(max_rounds):
            if valid.all():
                break
            pending = (~valid).nonzero(as_tuple=True)[0]

            # Step from the ORIGINAL current position each round (not the
            # previous round's rejected candidate), so a resample never
            # compounds drift across rounds.
            offsets = self.disk_offset(step_radius, pending.shape[0])
            candidates = current_xy[pending] + offsets

            # Reject points that left the cluster's own deviation_radius
            # disk (or ellipse, if radial_dir is given) around ref_centers.
            if radial_dir is not None:
                delta = candidates - ref_centers[pending]
                rdir = radial_dir[pending]
                tdir = torch.stack([-rdir[:, 1], rdir[:, 0]], dim=-1)
                radial_comp = (delta * rdir).sum(-1)
                tangential_comp = (delta * tdir).sum(-1)
                ok = (radial_comp / deviation_radius) ** 2 + (tangential_comp / tangential_radius) ** 2 <= 1.0
            else:
                ok = torch.linalg.norm(candidates - ref_centers[pending], dim=-1) <= deviation_radius

            # Reject points outside network coverage footprint
            ok = ok & self.topology.is_within_coverage(candidates)

            # Enforce min BS-UE distance
            if min_dist_from_site is not None:
                nearest_dist = torch.linalg.norm(
                    candidates[:, None, :] - site_loc[None, :, :], dim=-1
                ).min(dim=-1).values
                ok = ok & (nearest_dist >= min_dist_from_site)

            # Update the valid points
            xy[pending[ok]] = candidates[ok]
            valid[pending[ok]] = True

        logger.debug("walk_points: rounds_used=%d valid=%d/%d", round_idx + 1, int(valid.sum()), valid.numel())
        if not valid.all():
            logger.debug("walk_points: %d/%d points kept their previous position after max_rounds=%d",
                         int((~valid).sum()), valid.numel(), max_rounds)
        logger.function("walk_points end")
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

    def hotspot_region_points(self, sector_idx: int, region_type: str, num_points: int,
                              deviation_radius: float, min_dist_from_site: float = None,
                              max_rounds: int = 300, tangential_radius: float = None) -> torch.Tensor:
        """num_points FRESH (x, y) draws inside sector_idx's own
        `region_type` ("near" | "mid" | "far") hotspot region (see
        CellularTopology.hotspot_region_membership) -- never a fixed/reused
        waypoint, a genuinely new point every call ("the waypoints can be
        anywhere within the hotspot area").

        Samples uniform (alpha, beta) in [0,1) directly instead of adapting
        the disk-based samplers above -- the affine map to world
        coordinates is area-preserving up to a constant Jacobian, so this
        is uniform-over-rhombus-area, the same principle
        build_sector_rhombus_grid's own grid layout already relies on.

        Two-pass rejection, matching valid_points/walk_points' own idiom:
        pass 1 (preferred, first half of max_rounds) additionally requires
        a whole ellipse around the candidate -- semi-axes deviation_radius
        (RADIAL, along the site->candidate direction) and tangential_radius
        (perpendicular to it; defaults to deviation_radius, i.e. a plain
        circle, when not given) -- probed at NUM_CLEARANCE_SAMPLES points
        on that ellipse's boundary, to ALSO classify as the same
        region_type. near/mid/far is purely a RADIAL cutoff (see
        hotspot_region_membership), so only the radial clearance actually
        needs to be tight; the tangential clearance can stay as wide as
        deviation_radius used to be without risking a member crossing into
        a neighboring near/mid/far band. "Clearance from the boundary" is
        checked empirically (does the whole ellipse stay in-region) rather
        than analytically, same as before.
        Pass 2 (fallback, second half, logged once) drops that clearance
        check. Raises RuntimeError if even pass 2 can't fill every point
        within max_rounds -- an explicit "this hotspot region has ~no
        real area" detection, never a silent bad result.

        :param region_type: "near" | "mid" | "far" | "near_mid" (the last
            is CellularTopology.hotspot_region_membership's own derived
            near|mid union, for callers that only need a coarser 2-way
            near_mid-vs-far split).
        """
        assert region_type in ("near", "mid", "far", "near_mid"), f"Unknown region_type: {region_type!r}"
        logger.function("hotspot_region_points start: sector=%d region=%s num_points=%d",
                        sector_idx, region_type, num_points)
        topology = self.topology
        if min_dist_from_site is None:
            min_dist_from_site = float(topology.min_bs_ut_dist)
        if tangential_radius is None:
            tangential_radius = deviation_radius
        bs_xy_np, e1_np, e2_np = topology.sector_basis(sector_idx)
        bs_xy = torch.as_tensor(bs_xy_np, dtype=self.dtype, device=self.device)
        e1 = torch.as_tensor(e1_np, dtype=self.dtype, device=self.device)
        e2 = torch.as_tensor(e2_np, dtype=self.dtype, device=self.device)

        def draw_world(n):
            alpha = torch.rand(n, dtype=self.dtype, device=self.device, generator=self.generator)
            beta = torch.rand(n, dtype=self.dtype, device=self.device, generator=self.generator)
            return bs_xy[None, :] + alpha[:, None] * e1[None, :] + beta[:, None] * e2[None, :]

        num_clearance_samples = 8
        clearance_angles = np.linspace(0.0, 2 * np.pi, num_clearance_samples, endpoint=False)

        def region_mask(xy_np, radial_clearance, tang_clearance):
            membership = topology.hotspot_region_membership(sector_idx, xy_np)
            ok = membership[region_type]
            if region_type in ("near", "near_mid"):
                dist_to_site = np.linalg.norm(xy_np - bs_xy_np[None, :], axis=-1)
                ok = ok & (dist_to_site >= min_dist_from_site + radial_clearance)
            if radial_clearance > 0 or tang_clearance > 0:
                # Per-candidate radial direction (site -> candidate) -- the
                # near/mid/far cutoff only cares about this axis, so the
                # probe ellipse is oriented individually per point, not a
                # single sector-wide direction.
                radial_vec = xy_np - bs_xy_np[None, :]
                norm = np.linalg.norm(radial_vec, axis=-1, keepdims=True)
                norm = np.where(norm < 1e-9, 1.0, norm)
                radial_dir_np = radial_vec / norm
                tangential_dir_np = np.stack([-radial_dir_np[:, 1], radial_dir_np[:, 0]], axis=-1)
                for angle in clearance_angles:
                    probe = (xy_np + radial_clearance * np.cos(angle) * radial_dir_np
                            + tang_clearance * np.sin(angle) * tangential_dir_np)
                    offset_membership = topology.hotspot_region_membership(sector_idx, probe)
                    ok = ok & offset_membership[region_type]
            return ok

        xy = torch.zeros(num_points, 2, dtype=self.dtype, device=self.device)
        valid = torch.zeros(num_points, dtype=torch.bool, device=self.device)
        relaxed_logged = False
        round_idx = 0
        for round_idx in range(max_rounds):
            if valid.all():
                break
            pending = (~valid).nonzero(as_tuple=True)[0]
            candidates = draw_world(pending.shape[0])
            candidates_np = candidates.detach().cpu().numpy()
            radial_clearance = deviation_radius if round_idx < max_rounds // 2 else 0.0
            tang_clearance = tangential_radius if round_idx < max_rounds // 2 else 0.0
            if radial_clearance == 0.0 and not relaxed_logged:
                logger.warning("hotspot_region_points: sector=%d region=%s falling back to unbuffered "
                              "region membership after %d rounds -- hotspot region may be narrow "
                              "relative to deviation_radius=%.2f/tangential_radius=%.2f",
                              sector_idx, region_type, round_idx, deviation_radius, tangential_radius)
                relaxed_logged = True
            ok_np = region_mask(candidates_np, radial_clearance, tang_clearance)
            ok = torch.as_tensor(ok_np, dtype=torch.bool, device=self.device)
            ok = ok & topology.is_within_coverage(candidates)

            xy[pending[ok]] = candidates[ok]
            valid[pending[ok]] = True

        logger.debug("hotspot_region_points: sector=%d region=%s rounds_used=%d valid=%d/%d",
                    sector_idx, region_type, round_idx + 1, int(valid.sum()), valid.numel())
        if not valid.all():
            raise RuntimeError(
                f"hotspot_region_points: could not sample {int((~valid).sum())}/{num_points} valid "
                f"points for sector={sector_idx} region={region_type!r} after {max_rounds} rounds, "
                "even without the deviation_radius clearance margin -- this hotspot region is "
                "likely too narrow for this sector's real geometry given the current fine-grid "
                "resolution. Enlarge the region (coarser grid) or shrink deviation_radius rather "
                "than silently using an invalid point."
            )
        logger.function("hotspot_region_points end")
        return xy

    def cluster_centers(self, num_clusters_per_sector: int, min_dist_from_site: float = None):
        """``num_clusters_per_sector`` cluster-center (x, y) positions for
        EVERY sector of every site (num_bs * num_clusters_per_sector total)
        -- exact, not an approximate/rounded split (the old num_groups-based
        version this replaces only approximated even distribution via
        divmod, since a network-wide total doesn't always divide evenly
        across sites then sectors).

        :param min_dist_from_site: defaults to topology.min_bs_ut_dist --
            pass a larger, buffered value (e.g. min_bs_ut_dist + a cluster
            radius cap) to keep the WHOLE cluster disk centered here outside
            the exclusion zone, not just this one center point.
        """
        logger.function("cluster_centers start: num_clusters_per_sector=%d", num_clusters_per_sector)

        # Initialization
        topology = self.topology
        num_cells = topology.num_cells
        num_sectors_per_site = topology.num_sectors_per_site
        cell_radius = float(topology.grid.cell_radius.item())
        if min_dist_from_site is None:
            min_dist_from_site = topology.min_bs_ut_dist

        # Sector angular region
        wedge_half_width = math.pi / num_sectors_per_site

        # Cluster center generation -- num_clusters_per_sector for EVERY
        # sector of EVERY site.
        start_xy_list = []
        for site_idx in range(num_cells):
            sector_yaws = topology.bs_orientations[0, site_idx * num_sectors_per_site:
                                                   (site_idx + 1) * num_sectors_per_site, 0]
            site_center = topology.site_loc[site_idx]
            for k in range(num_sectors_per_site):
                centers = site_center.unsqueeze(0).expand(num_clusters_per_sector, -1)
                cluster_xy = self.valid_points(
                    centers, cell_radius,
                    angle_center=float(sector_yaws[k]), angle_half_width=wedge_half_width,
                    min_dist_from_site=min_dist_from_site,
                )
                start_xy_list.extend(tuple(xy) for xy in cluster_xy.tolist())
        logger.debug("cluster_centers: generated %d cluster centers", len(start_xy_list))
        logger.function("cluster_centers end")
        return start_xy_list

    def clustered(self, cluster_centers, members_per_cluster, deviation_radius: float, ut_height: float,
                 cluster_sector: np.ndarray = None, tangential_radius: float = None):
        """generates the actual UEs around the cluster centers.

        :param cluster_centers: [num_clusters, 2] (x, y) cluster center points.
        :param members_per_cluster: int (same for every cluster) or a
            length-``num_clusters`` list of per-cluster member counts.
        :param deviation_radius: max per-member offset from its cluster's
            center [m] -- the RADIAL semi-axis when cluster_sector is given
            (an ellipse oriented per-cluster, see valid_points), else the
            plain isotropic disk radius.
        :param cluster_sector: [num_clusters] int, each cluster's home
            sector -- when given (together with tangential_radius), member
            offsets are sampled from an ellipse compressed along the
            site->center direction instead of an isotropic disk. None
            (default): original isotropic behavior.
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

        radial_dir = None
        if cluster_sector is not None:
            site_xy = self.topology.bs_loc[0, :, :2].to(dtype=self.dtype, device=self.device)  # [num_bs, 2]
            sector_t = torch.as_tensor(cluster_sector, device=self.device)[member_cluster_idx]
            site_per_member = site_xy[sector_t]
            radial_vec = center_per_member - site_per_member
            radial_dir = radial_vec / radial_vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        # Generate valid UE locations
        xy = self.valid_points(center_per_member, deviation_radius, min_dist_from_site=self.topology.min_bs_ut_dist,
                               radial_dir=radial_dir, tangential_radius=tangential_radius)
        z = torch.full((xy.shape[0], 1), float(ut_height), dtype=self.dtype, device=self.device)

        # Update UE locations
        ut_loc = torch.cat([xy, z], dim=-1)

        logger.debug("clustered: ut_loc shape=%s", tuple(ut_loc.shape))
        logger.function("clustered end")
        return ut_loc, member_cluster_idx
