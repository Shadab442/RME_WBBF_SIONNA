"""Mobility models for the tilt-over-time study.

These only EVOLVE positions over time -- they don't generate initial ones
(see helpers/ue_drop.py for that). RandomWalkMobility takes an initial ut_loc
directly; ReferencePointGroupMobility takes ut_loc plus which group each
member belongs to (both normally from ue_drop.sample_clustered_ut_loc).

Both classes still take the CellularTopology itself (not just a radius) for
their own ongoing boundary checks during movement (random-waypoint
destinations, random-walk reflection) -- a real mobility-time need, separate
from how the initial positions were generated.
"""

import math

import torch

from .ue_drop import sample_disk_offset, sample_valid_offset


class MobilityModel:
    """Common interface: owns current UE positions, advances them one step
    at a time.

    :ivar ut_loc: [num_ut, 3] current UE positions [m].
    """

    def __init__(self, ut_loc: torch.Tensor):
        self.ut_loc = ut_loc

    def step(self, dt: float) -> torch.Tensor:
        """Advance positions by dt [s]; updates and returns self.ut_loc."""
        raise NotImplementedError


class ReferencePointGroupMobility(MobilityModel):
    """Reference Point Group Mobility (Hong, Gerla, Pei & Chiang, 1999),
    random-waypoint variant: each group has a reference point that follows
    a random-waypoint walk (random destination within the topology's
    coverage footprint, random speed in ``[min_speed, max_speed]``,
    resampled on arrival). Every member's position is that reference point
    plus a per-member offset within ``deviation_radius`` of it. That offset
    is persistent state -- it drifts by a small random step each call to
    ``step()`` (speed ``member_jitter_speed``).
    """

    def __init__(
        self,
        ut_loc: torch.Tensor,
        member_group_idx: torch.Tensor,
        start_xy_list,
        deviation_radius: float,
        topo,
        min_speed: float,
        max_speed: float,
        member_jitter_speed: float = 0.3,
        generator=None,
    ):
        """
        :param ut_loc: [num_ut, 3] initial member positions -- e.g. from
            ``helpers.ue_drop.sample_clustered_ut_loc``. This class only
            evolves these positions; it doesn't generate them.
        :param member_group_idx: [num_ut] which group (an index into
            ``start_xy_list``) each member belongs to -- from the same drop
            call that produced ``ut_loc``.
        :param start_xy_list: one ``(x, y)`` per group -- its reference
            point's starting position, matching the cluster center used to
            generate that group's members.
        :param deviation_radius: max per-member offset from the reference
            point [m] -- see class docstring for how this controls cluster size.
        :param topo: the scenario's ``CellularTopology`` -- used for its
            real hex-footprint boundary test (``is_within_coverage``) and
            ``default_drop_radius`` (the disk random-waypoint destinations
            are drawn from, before rejecting candidates outside the real
            footprint).
        :param min_speed, max_speed: reference point speed range [m/s].
        :param member_jitter_speed: how fast each member's own offset from
            its group's reference point drifts per step [m/s]. This is a
            random walk (accumulates as sqrt(steps), not linearly) so it
            needs to be comparable to, not tiny next to, min_speed/max_speed
            to be visible at all over a run -- but not so large it swamps
            the group's own drift as the dominant, visible motion.
        """
        self.dtype, self.device, self.generator = ut_loc.dtype, ut_loc.device, generator
        self.deviation_radius = deviation_radius
        self.topo = topo
        self.min_speed, self.max_speed = min_speed, max_speed
        self.member_jitter_speed = member_jitter_speed
        self.member_group_idx = member_group_idx

        self.ref_xy = torch.tensor(start_xy_list, dtype=self.dtype, device=self.device)  # [num_groups, 2]
        num_groups = self.ref_xy.shape[0]

        # Every group gets an initial random destination + speed.
        self.dest_xy = self.ref_xy.clone()
        self.speed = torch.zeros(num_groups, dtype=self.dtype, device=self.device)
        for i in range(num_groups):
            self._pick_new_waypoint(i)

        self._z = ut_loc[:, 2:3].clone()  # persistent per-member height, carried as-is
        ref_per_member = self.ref_xy[self.member_group_idx]
        self.member_offset = ut_loc[:, :2] - ref_per_member  # each member's offset, as given

        super().__init__(ut_loc)

    def _is_valid(self, xy: torch.Tensor) -> torch.Tensor:
        """True where xy is both within the real coverage footprint and at
        least min_bs_ut_dist from every BS site -- the same two conditions
        helpers.ue_drop.sample_uniform_ut_loc enforces for a plain UE drop.
        Both matter here: a too-close position blows up pathloss into
        inf/nan in the channel model just as much as an out-of-coverage one
        breaks the topology's own assumptions.
        """
        within_coverage = self.topo.is_within_coverage(xy)
        site_loc = self.topo.site_loc.to(dtype=xy.dtype, device=xy.device)
        nearest_dist = torch.linalg.norm(xy[:, None, :] - site_loc[None, :, :], dim=-1).min(dim=-1).values
        return within_coverage & (nearest_dist >= self.topo.min_bs_ut_dist)

    def _pick_new_waypoint(self, group_idx: int) -> None:
        """Draws a new random destination (within the topology's actual
        coverage footprint) and speed for a group's reference point."""
        origin = torch.zeros(1, 2, dtype=self.dtype, device=self.device)
        dest = sample_valid_offset(origin, self.topo.default_drop_radius, self.topo,
                                   self.dtype, self.device, self.generator)
        self.dest_xy[group_idx] = dest[0]
        u = torch.rand((), dtype=self.dtype, device=self.device, generator=self.generator)
        self.speed[group_idx] = self.min_speed + (self.max_speed - self.min_speed) * u

    def step(self, dt: float) -> torch.Tensor:
        # Cluster center movement. The destination itself is always valid
        # (sample_valid_offset rejects invalid candidates when it's picked),
        # but the STRAIGHT LINE from here to there is not: the coverage
        # footprint is a union of hexagons, not a convex region, so it has
        # notches a straight line between two valid points can still cross.
        # An uncorrected reference point can then travel outside coverage
        # for as long as the current leg of transit takes -- observed
        # empirically staying out for 38+ consecutive steps on a 1-ring
        # topology. So each candidate step is itself checked: if it would
        # leave coverage, stay put this step and pick a new destination
        # instead of continuing through invalid territory.
        for i in range(self.ref_xy.shape[0]):
            to_dest = self.dest_xy[i] - self.ref_xy[i]
            dist = torch.linalg.norm(to_dest)
            step_len = self.speed[i] * dt
            if dist <= step_len:
                candidate = self.dest_xy[i]
            else:
                candidate = self.ref_xy[i] + to_dest / dist * step_len
            if self.topo.is_within_coverage(candidate.unsqueeze(0))[0]:
                self.ref_xy[i] = candidate
                if dist <= step_len:
                    self._pick_new_waypoint(i)
            else:
                self._pick_new_waypoint(i)

        # Members drift by a small random step from their PREVIOUS offset
        ref_per_member = self.ref_xy[self.member_group_idx]
        num_ut = ref_per_member.shape[0]
        jitter = sample_disk_offset(self.member_jitter_speed * dt, num_ut, self.dtype, self.device, self.generator)
        new_offset = self.member_offset + jitter
        mag = torch.linalg.norm(new_offset, dim=-1)
        too_far = mag > self.deviation_radius
        new_offset[too_far] = new_offset[too_far] / mag[too_far, None] * self.deviation_radius

        xy = ref_per_member + new_offset
        ok = self._is_valid(xy)

        # Tier 1 fallback: reuse the previous offset with the NEW reference
        # point.
        fallback_xy = ref_per_member + self.member_offset
        fallback_ok = self._is_valid(fallback_xy)
        use_fallback = (~ok) & fallback_ok
        xy[use_fallback] = fallback_xy[use_fallback]
        new_offset[use_fallback] = self.member_offset[use_fallback]

        # Tier 2 (last resort): even that combination is invalid -- e.g. the
        # reference point moved outward and this member was already near the
        # edge. Freezing at the previous absolute position (as this used to
        # do) guarantees validity but not progress: if the reference point
        # keeps moving in a direction that regenerates the same conflict
        # (e.g. it's tracking along the boundary for a few steps), the
        # member can stay frozen for many consecutive steps while the rest
        # of its group keeps moving, visibly "left behind" at the edge.
        # Resampling a fresh valid offset around the CURRENT reference point
        # instead (same rejection sampling used to build the initial drop
        # and to pick waypoints) keeps the member moving every step and
        # rejoins normal jitter-based drift from a valid state, rather than
        # waiting on a lucky jitter draw or the group wandering back away
        # from the boundary on its own.
        still_bad = (~ok) & (~fallback_ok)
        if still_bad.any():
            resampled_xy = sample_valid_offset(
                ref_per_member[still_bad], self.deviation_radius, self.topo,
                self.dtype, self.device, self.generator,
                min_dist_from_site=self.topo.min_bs_ut_dist,
            )
            xy[still_bad] = resampled_xy
            new_offset[still_bad] = resampled_xy - ref_per_member[still_bad]

        self.member_offset = new_offset
        self.ut_loc = torch.cat([xy, self._z], dim=-1)
        return self.ut_loc


class RandomWalkMobility(MobilityModel):
    """Plain, ungrouped baseline: each UE has its own constant heading and
    speed, reflecting off the topology's real hex-grid boundary -- a point
    of comparison against RPGM's clustered drift, not meant to model
    anything in particular.

    Reflection here is a simple velocity-reversal approximation (not a true
    mirror-reflection about the boundary normal), which is fine for a
    baseline but not for precise boundary physics.
    """

    def __init__(
        self,
        ut_loc: torch.Tensor,
        topo,
        min_speed: float,
        max_speed: float,
        generator=None,
    ):
        super().__init__(ut_loc)
        self.topo = topo
        dtype, device = ut_loc.dtype, ut_loc.device
        num_ut = ut_loc.shape[0]
        speed = min_speed + (max_speed - min_speed) * torch.rand(
            num_ut, dtype=dtype, device=device, generator=generator
        )
        heading = 2.0 * math.pi * torch.rand(num_ut, dtype=dtype, device=device, generator=generator)
        self.velocity_xy = torch.stack([speed * torch.cos(heading), speed * torch.sin(heading)], dim=-1)

    def step(self, dt: float) -> torch.Tensor:
        xy = self.ut_loc[:, :2] + self.velocity_xy * dt
        outside = ~self.topo.is_within_coverage(xy)
        if outside.any():
            self.velocity_xy[outside] = -self.velocity_xy[outside]
            xy[outside] = self.ut_loc[outside, :2] + self.velocity_xy[outside] * dt
            # A single reversal can still land outside near sharp hex corners
            # (unlike a circle, the boundary isn't equidistant in every
            # direction) -- fall back to just staying put for the rare UE
            # still outside after the bounce, rather than risk a second
            # escape.
            still_outside = outside.clone()
            still_outside[outside] = ~self.topo.is_within_coverage(xy[outside])
            xy[still_outside] = self.ut_loc[still_outside, :2]
        self.ut_loc = torch.cat([xy, self.ut_loc[:, 2:3]], dim=-1)
        return self.ut_loc
