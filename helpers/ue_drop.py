"""Initial UE-placement strategies, built on top of a CellularTopology.
"""

import math

import torch


def sample_disk_offset(radius: float, num_points: int, dtype, device, generator=None,
                       angle_center: float = None, angle_half_width: float = None) -> torch.Tensor:
    """Uniform-random (x, y) offsets within a disk of the given radius --
    r=R*sqrt(U), so area density is uniform. theta is drawn over the full
    circle by default, or restricted to the wedge
    [angle_center - angle_half_width, angle_center + angle_half_width]
    [rad] if both are given.
    """
    r = radius * torch.sqrt(torch.rand(num_points, dtype=dtype, device=device, generator=generator))
    u = torch.rand(num_points, dtype=dtype, device=device, generator=generator)
    if angle_center is None:
        theta = 2.0 * math.pi * u
    else:
        theta = angle_center + angle_half_width * (2.0 * u - 1.0)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


def sample_valid_offset(centers: torch.Tensor, radius: float, topology, dtype, device,
                        generator=None, max_rounds: int = 100, min_dist_from_site: float = None,
                        angle_center: float = None, angle_half_width: float = None) -> torch.Tensor:
    """For each row in centers [N, 2], samples a random disk offset (radius,
    optionally restricted to an angular wedge -- see sample_disk_offset)
    such that centers[i] + offset lies within topology's real hex-grid
    footprint, rejecting and resampling per-point as needed.
    """
    xy = centers.clone()
    valid = torch.zeros(centers.shape[0], dtype=torch.bool, device=device)
    if min_dist_from_site is not None:
        site_loc = topology.site_loc.to(dtype=dtype, device=device)
    for _ in range(max_rounds):
        if valid.all():
            break
        pending = (~valid).nonzero(as_tuple=True)[0]
        offsets = sample_disk_offset(radius, pending.shape[0], dtype, device, generator,
                                     angle_center=angle_center, angle_half_width=angle_half_width)
        candidates = centers[pending] + offsets
        ok = topology.is_within_coverage(candidates)
        if min_dist_from_site is not None:
            nearest_dist = torch.linalg.norm(
                candidates[:, None, :] - site_loc[None, :, :], dim=-1
            ).min(dim=-1).values
            ok = ok & (nearest_dist >= min_dist_from_site)
        xy[pending[ok]] = candidates[ok]
        valid[pending[ok]] = True
    return xy


def sample_uniform_ut_loc(
    topology,
    num_ut: int,
    ut_height: float,
    dtype,
    device,
    disk_radius: float = None,
    generator=None,
    batch_size: int = None,
) -> torch.Tensor:
    """Uniformly-at-random (x, y) UE drop, at a fixed height, over the
    hex-grid's coverage footprint 

    Candidates are drawn uniformly over a disk (``r = R*sqrt(U)``,
    ``theta = 2*pi*U``, so area density is uniform), then rejected and
    resampled unless both:
    - at least ``min_bs_ut_dist`` from every BS site (3GPP's minimum
      BS-UT distance), and
    - inside their nearest site's hexagon (so no UE lands in the disk's
      overshoot beyond actual coverage).

    :param topology: a :class:`~helpers.cellular_topology.CellularTopology`.
    :param disk_radius: circumscribing disk to draw candidates from;
        defaults to ``topology.default_drop_radius``.
    :param batch_size: if given, draws this many independent drops and
        stacks them into ``[batch_size, num_ut, 3]`` 
    :output ut_loc: ``[num_ut, 3]``, or ``[batch_size, num_ut, 3]`` if
        ``batch_size`` is given.
    """
    if batch_size is not None:
        return torch.stack(
            [
                sample_uniform_ut_loc(topology, num_ut, ut_height, dtype, device, disk_radius, generator)
                for _ in range(batch_size)
            ],
            dim=0,
        )
    if disk_radius is None:
        disk_radius = topology.default_drop_radius

    site_loc = topology.site_loc.to(dtype=dtype, device=device)
    min_bs_ut_dist = topology.min_bs_ut_dist

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

        nearest_dist = torch.linalg.norm(
            candidates[:, None, :] - site_loc[None, :, :], dim=-1
        ).min(dim=-1).values
        far_enough = nearest_dist >= min_bs_ut_dist
        within_coverage = topology.is_within_coverage(candidates)

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


def sample_grid_points(topology, spacing: float, ut_height: float, dtype, device,
                       disk_radius: float = None) -> torch.Tensor:
    """Deterministic regular (x, y) grid spanning the hex-grid's real
    coverage footprint, at fixed height.

    :param spacing: grid spacing [m] along both x and y.
    :param disk_radius: bounding disk the lattice is generated over before
        filtering; defaults to topology.default_drop_radius.
    :output grid_loc: [num_grid_points, 3] -- how many lattice points
        survive the validity filters isn't specified directly, it falls
        out of spacing/disk_radius.
    """
    if disk_radius is None:
        disk_radius = topology.default_drop_radius
    site_loc = topology.site_loc.to(dtype=dtype, device=device)
    min_bs_ut_dist = topology.min_bs_ut_dist

    coords = torch.arange(-disk_radius, disk_radius + spacing, spacing, dtype=dtype, device=device)
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="xy")
    candidates = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # [N, 2]

    nearest_dist = torch.linalg.norm(
        candidates[:, None, :] - site_loc[None, :, :], dim=-1
    ).min(dim=-1).values
    far_enough = nearest_dist >= min_bs_ut_dist
    within_coverage = topology.is_within_coverage(candidates)
    kept = candidates[far_enough & within_coverage]

    z = torch.full((kept.shape[0], 1), float(ut_height), dtype=dtype, device=device)
    return torch.cat([kept, z], dim=-1)  # [num_grid_points, 3]


def sample_cluster_center_across_sites(topology, num_groups: int, generator=None):
    """``num_groups`` cluster-center (x, y) positions, split as evenly as
    possible across ``topology``'s sites, then -- within each site -- split
    as evenly as possible again across its ``num_sectors_per_site`` sectors,
    each cluster placed uniformly at random within its own sector's 120 deg
    wedge of the site's cell footprint (not anywhere in the site, which
    would leave sector balance uncontrolled).
    """
    num_cells = topology.num_cells
    num_sectors_per_site = topology.num_sectors_per_site
    base, extra = divmod(num_groups, num_cells)
    clusters_per_site = [base + 1 if site_idx < extra else base for site_idx in range(num_cells)]
    cell_radius = float(topology.grid.cell_radius.item())
    wedge_half_width = math.pi / num_sectors_per_site
    start_xy_list = []
    for site_idx, n_clusters in enumerate(clusters_per_site):
        if n_clusters == 0:
            continue
        sector_base, sector_extra = divmod(n_clusters, num_sectors_per_site)
        clusters_per_sector = [sector_base + 1 if k < sector_extra else sector_base
                               for k in range(num_sectors_per_site)]
        sector_yaws = topology.bs_orientations[0, site_idx * num_sectors_per_site:
                                               (site_idx + 1) * num_sectors_per_site, 0]
        site_center = topology.site_loc[site_idx]
        for k, n_sector_clusters in enumerate(clusters_per_sector):
            if n_sector_clusters == 0:
                continue
            centers = site_center.unsqueeze(0).expand(n_sector_clusters, -1)
            cluster_xy = sample_valid_offset(
                centers, cell_radius, topology,
                topology.bs_loc.dtype, topology.bs_loc.device, generator,
                angle_center=float(sector_yaws[k]), angle_half_width=wedge_half_width,
            )
            start_xy_list.extend(tuple(xy) for xy in cluster_xy.tolist())
    return start_xy_list


def sample_clustered_ut_loc(
    topology,
    cluster_centers,
    members_per_cluster,
    deviation_radius: float,
    ut_height: float,
    dtype,
    device,
    generator=None,
):
    """UEs dropped in clusters around given center points, each member a
    random offset (within deviation_radius) from its own cluster's center,
    rejection-sampled against the real hex-grid footprint.

    :param topology: a :class:`~helpers.cellular_topology.CellularTopology`.
    :param cluster_centers: [num_clusters, 2] (x, y) cluster center points.
    :param members_per_cluster: int (same for every cluster) or a
        length-``num_clusters`` list of per-cluster member counts.
    :param deviation_radius: max per-member offset from its cluster's
        center [m].
    :output: (ut_loc [num_ut, 3], member_cluster_idx [num_ut] long tensor
        giving which cluster each member belongs to, in the same order as
        ``cluster_centers``).
    """
    centers = torch.as_tensor(cluster_centers, dtype=dtype, device=device)
    num_clusters = centers.shape[0]
    if isinstance(members_per_cluster, int):
        members_per_cluster = [members_per_cluster] * num_clusters

    member_cluster_idx = torch.repeat_interleave(
        torch.arange(num_clusters, device=device),
        torch.tensor(members_per_cluster, device=device),
    )
    center_per_member = centers[member_cluster_idx]  # [num_ut, 2]

    xy = sample_valid_offset(center_per_member, deviation_radius, topology, dtype, device, generator,
                             min_dist_from_site=topology.min_bs_ut_dist)
    z = torch.full((xy.shape[0], 1), float(ut_height), dtype=dtype, device=device)
    ut_loc = torch.cat([xy, z], dim=-1)
    return ut_loc, member_cluster_idx
