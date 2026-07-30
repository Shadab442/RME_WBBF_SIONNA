"""Pluggable initial UE-placement strategies, built on top of a CellularTopology.

Both functions here only produce t=0 positions -- how those positions evolve
over time (if at all) is a separate, later concern; see helpers/mobility.py.
"""

import math

import torch


def sample_disk_offset(radius: float, num_points: int, dtype, device, generator=None) -> torch.Tensor:
    """Uniform-random (x, y) offsets within a disk of the given radius --
    r=R*sqrt(U), theta=2*pi*U, so area density is uniform."""
    r = radius * torch.sqrt(torch.rand(num_points, dtype=dtype, device=device, generator=generator))
    theta = 2.0 * math.pi * torch.rand(num_points, dtype=dtype, device=device, generator=generator)
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)


def sample_valid_offset(centers: torch.Tensor, radius: float, topology, dtype, device,
                        generator=None, max_rounds: int = 100) -> torch.Tensor:
    """For each row in centers [N, 2], samples a random disk offset (radius)
    such that centers[i] + offset lies within topology's real hex-grid
    footprint (topology.is_within_coverage), rejecting and resampling
    per-point as needed. Falls back to zero offset (stays at its center)
    for any point still invalid after max_rounds, so this can never loop
    forever regardless of how large radius is relative to the coverage area.
    """
    xy = centers.clone()
    valid = torch.zeros(centers.shape[0], dtype=torch.bool, device=device)
    for _ in range(max_rounds):
        if valid.all():
            break
        pending = (~valid).nonzero(as_tuple=True)[0]
        offsets = sample_disk_offset(radius, pending.shape[0], dtype, device, generator)
        candidates = centers[pending] + offsets
        ok = topology.is_within_coverage(candidates)
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
    hex-grid's actual coverage footprint -- the union of every site's
    hexagon, not just a circumscribing disk (which bulges past the hexagon
    edges in the gaps between cells).

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
    rejection-sampled against the real hex-grid footprint (never just a
    circular approximation of it).

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
        torch.arange(num_clusters), torch.tensor(members_per_cluster)
    )
    center_per_member = centers[member_cluster_idx]  # [num_ut, 2]

    xy = sample_valid_offset(center_per_member, deviation_radius, topology, dtype, device, generator)
    z = torch.full((xy.shape[0], 1), float(ut_height), dtype=dtype, device=device)
    ut_loc = torch.cat([xy, z], dim=-1)
    return ut_loc, member_cluster_idx
