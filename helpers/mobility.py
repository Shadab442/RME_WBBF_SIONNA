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
    ``step()`` (speed ``member_jitter_speed``), rather than being redrawn
    from scratch every time, so members shuffle around smoothly within
    their cluster instead of teleporting to a new random spot in the disk
    every frame (which would swamp the group's own, typically much slower,
    drift).

    (A "deterministic" scripted start->end path variant existed earlier and
    was removed -- its traversal speed was an accidental byproduct of site
    spacing and total simulation length rather than a controlled parameter,
    and its post-arrival behavior [freeze in place] wasn't well thought
    through. Revisit later if a controlled near-to-far scenario is needed
    again, with an explicit per-group speed instead.)

    ``deviation_radius`` is the knob for how much area each cluster spans --
    small values give tight, visually distinct clusters; values approaching
    ``topo.default_drop_radius`` make clusters spread across most of the
    coverage area. Member offsets are rejection-sampled against the real
    footprint at t=0, and step()'s two-tier fallback (see there) keeps every
    later position valid too, so members never end up outside it at any point
    in time -- not just for the fresh-candidate case, but also across a
    reference point's own movement between steps.

    :ivar ut_loc: [num_ut, 3] current member positions.
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
        for i in range(self.ref_xy.shape[0]):
            to_dest = self.dest_xy[i] - self.ref_xy[i]
            dist = torch.linalg.norm(to_dest)
            step_len = self.speed[i] * dt
            if dist <= step_len:
                self.ref_xy[i] = self.dest_xy[i]
                self._pick_new_waypoint(i)
            else:
                self.ref_xy[i] = self.ref_xy[i] + to_dest / dist * step_len

        # Members drift by a small random step from their PREVIOUS offset
        # (not a fresh resample) -- clipped back to deviation_radius if it
        # would wander too far from the reference point, rather than
        # rejected outright, so members glide along the cluster's edge
        # instead of freezing there.
        ref_per_member = self.ref_xy[self.member_group_idx]
        num_ut = ref_per_member.shape[0]
        jitter = sample_disk_offset(self.member_jitter_speed * dt, num_ut, self.dtype, self.device, self.generator)
        new_offset = self.member_offset + jitter
        mag = torch.linalg.norm(new_offset, dim=-1)
        too_far = mag > self.deviation_radius
        new_offset[too_far] = new_offset[too_far] / mag[too_far, None] * self.deviation_radius

        xy = ref_per_member + new_offset
        ok = self.topo.is_within_coverage(xy)

        # Tier 1 fallback: reuse the previous offset with the NEW reference
        # point. Not guaranteed valid on its own -- the reference point moved
        # this step, so "old offset + new reference point" is a combination
        # that was never actually validated (only "old reference point + old
        # offset", last step, and "new reference point + new offset", just
        # above, were ever checked).
        fallback_xy = ref_per_member + self.member_offset
        fallback_ok = self.topo.is_within_coverage(fallback_xy)
        use_fallback = (~ok) & fallback_ok
        xy[use_fallback] = fallback_xy[use_fallback]
        new_offset[use_fallback] = self.member_offset[use_fallback]

        # Tier 2 (last resort): even that combination is invalid -- e.g. the
        # reference point moved outward and this member was already near the
        # edge. Freeze this member at its previous ABSOLUTE position for this
        # one step (guaranteed valid, since it was accepted last step -- or,
        # at t=0, came from sample_clustered_ut_loc's own rejection sampling),
        # and recompute its stored offset relative to the new reference point
        # so next step's bookkeeping stays consistent.
        still_bad = (~ok) & (~fallback_ok)
        xy[still_bad] = self.ut_loc[still_bad, :2]
        new_offset[still_bad] = xy[still_bad] - ref_per_member[still_bad]

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
