"""Mobility models for the tilt-over-time study.
"""

import torch

from .ue_drop import UeDropper
from helpers.utils import get_logger

logger = get_logger(__name__)


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
    """Reference Point Group Mobility (Hong, Gerla, Pei & Chiang, 1999):
    each group has a reference point, every member's position is that
    reference point plus a per-member offset within ``deviation_radius`` of
    it. Two independently configurable axes:

    cluster_mobility_mode -- WHERE each group's reference point is headed:
      "random"   (baseline) -- a fresh random destination within the
                 topology's coverage footprint each time the group departs.
      "periodic" / "markov" -- each group has num_waypoints predefined
                 (x, y) hotspots; the NEXT one is chosen cyclically
                 (0->1->2->0->...) for "periodic", or sampled from
                 waypoint_transition_matrix's row for "markov". Waypoints
                 are auto-generated (unless explicit `waypoints` are given)
                 per waypoint_type:
                   "random"  -- within one hex cell_radius of the group's
                              own starting position (near its own site).
                   "central" -- num_waypoints chosen at random (without
                              replacement) from the centroids of the
                              group's home sector's 9 observation windows
                              (region_centers), so hotspots sit at
                              recognizable region centers instead of
                              arbitrary points.

    transition -- HOW the reference point gets from its current position to
    the next one picked above, independent of cluster_mobility_mode:
      "jump"   -- holds at the current position, then teleports instantly.
                 "periodic"/"markov" hold for a FIXED waypoint_hold_steps;
                 "random" holds for a FRESH random duration drawn uniformly
                 in [1, waypoint_hold_steps] each cycle (there's no fixed
                 hotspot to dwell at, so the dwell time itself is random
                 instead).
      "smooth" -- walks continuously toward the next position instead of
                 teleporting. "random" draws a random speed in
                 [min_speed, max_speed] each transition (its original
                 behavior) and departs again immediately on arrival (never
                 dwells -- it has no hotspot to hold at). "periodic"/
                 "markov" instead derive speed from transition_duration_s
                 (distance / transition_duration_s), so every transition
                 takes exactly that long regardless of how far the next
                 waypoint is -- not a random speed -- and dwell for
                 waypoint_hold_steps at the arrived-at waypoint first, same
                 as "jump".
    (cluster_mobility_mode="random" + transition="smooth", and "periodic"/
    "markov" + transition="jump", are this class's original two behaviors;
    the other two combinations are equally valid.)

    intra_cluster_mobility -- how a member's offset from the reference
    point evolves:
      "random_walk" (baseline) -- drifts by a small random step each call
                 to step() (speed member_jitter_speed), clamped to
                 deviation_radius.
      "static"   -- offset never changes; the member still moves WITH the
                 reference point (including across a jump), since its
                 absolute position is always reference + offset.

    All three axes (cluster_mobility_mode, transition, intra_cluster_mobility)
    are independently configurable.
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
        cluster_mobility_mode: str = "random",
        transition: str = None,
        num_waypoints: int = None,
        waypoint_hold_steps: int = None,
        waypoint_transition_matrix=None,
        waypoints=None,
        waypoint_type: str = "random",
        region_centers=None,
        region_centers_valid=None,
        cluster_home_sector=None,
        intra_cluster_mobility: str = "random_walk",
        waypoint_min_dist_from_site: float = None,
        transition_duration_s: float = None,
    ):
        """
        :param ut_loc: [num_ut, 3] initial member positions -- e.g. from
            ``helpers.ue_drop.UeDropper.clustered``. This class only
            evolves these positions; it doesn't generate them.
        :param member_group_idx: [num_ut] which group (an index into
            ``start_xy_list``) each member belongs to -- from the same drop
            call that produced ``ut_loc``.
        :param start_xy_list: one ``(x, y)`` per group -- its reference
            point's starting position, matching the cluster center used to
            generate that group's members. Also waypoint 0 when
            cluster_mobility_mode is "periodic"/"markov".
        :param deviation_radius: max per-member offset from the reference
            point [m] -- see class docstring for how this controls cluster size.
        :param topo: the scenario's ``CellularTopology`` -- used for its
            real hex-footprint boundary test (``is_within_coverage``) and
            ``default_drop_radius``
        :param min_speed, max_speed: reference point speed range [m/s] --
            transition="smooth" only (any cluster_mobility_mode).
        :param member_jitter_speed: how fast each member's own offset from
            its group's reference point drifts per step [m/s] --
            "random_walk" intra-cluster mode only.
        :param cluster_mobility_mode: "random" | "periodic" | "markov".
        :param transition: "jump" | "smooth" -- see class docstring.
            Defaults to this class's ORIGINAL behavior for
            cluster_mobility_mode's own default ("smooth" for "random",
            "jump" for "periodic"/"markov") if not given.
        :param num_waypoints: per-group hotspot count -- required for
            "periodic"/"markov".
        :param waypoint_hold_steps: number of step() calls to hold before
            transitioning -- required for "periodic"/"markov" (any
            transition), and for cluster_mobility_mode="random" with
            transition="jump" (there it bounds a PER-GROUP random dwell,
            drawn uniformly in [1, waypoint_hold_steps] each cycle, rather
            than being used as a fixed duration directly -- see class
            docstring). Caller converts to this unit; this class just
            counts its own step() calls.
        :param waypoint_transition_matrix: [num_waypoints, num_waypoints],
            rows summing to 1 -- required for "markov".
        :param waypoints: optional [num_groups, num_waypoints, 2] explicit
            per-group hotspot coordinates -- if given, overrides waypoint_type
            entirely (neither region_centers/cluster_home_sector needed).
        :param waypoint_type: "random" | "central" -- how waypoints are
            auto-generated when `waypoints` isn't given (see class docstring).
        :param region_centers: [num_bs_sectors, num_regions, 2] -- each
            sector's observation-window centroids, from
            CellularTopology.compute_observation_windows +
            CellularTopology.build_sector_rhombus_grid's grid_points.
            Required for waypoint_type="central".
        :param region_centers_valid: [num_bs_sectors, num_regions] bool --
            which of region_centers' own windows are usable as a hotspot
            candidate for that sector (far enough from every site that a
            CR_max-radius cluster centered there can't reach the
            min_bs_ut_dist exclusion zone -- see SimulationEngine, the only
            real caller). NOT every window qualifies (e.g. "corner_C" sits
            at the site itself by construction) and which ones do can vary
            per sector's real geometry, so this is a mask, not a fixed
            per-window rule. Required for waypoint_type="central".
        :param cluster_home_sector: [num_groups] int -- which sector each
            group's starting position falls in (CellularTopology.
            assign_to_sector_rhombus_grid). Required for waypoint_type="central".
        :param intra_cluster_mobility: "random_walk" | "static".
        :param waypoint_min_dist_from_site: waypoint_type="random" only --
            defaults to topo.min_bs_ut_dist. Pass a buffered value (e.g.
            min_bs_ut_dist + a cluster radius cap) so the WHOLE cluster
            disk around an auto-generated waypoint stays outside the
            exclusion zone, not just the waypoint point itself.
        :param transition_duration_s: required for cluster_mobility_mode
            "periodic"/"markov" with transition="smooth" -- the WALL-CLOCK
            duration (e.g. one tilt_control_interval_s) a transition should
            take, REGARDLESS of the distance to the next waypoint. Speed is
            derived (distance / transition_duration_s), not drawn from
            [min_speed, max_speed] -- every transition takes exactly this
            long, not a variable time depending on a random speed and how
            far the next waypoint happens to be. "random" mode's own smooth
            walk is unaffected -- it still draws speed randomly, since it
            has no natural "one interval" concept to target.
        """
        assert cluster_mobility_mode in ("random", "periodic", "markov"), \
            "cluster_mobility_mode must be 'random', 'periodic' or 'markov'"
        assert intra_cluster_mobility in ("random_walk", "static"), \
            "intra_cluster_mobility must be 'random_walk' or 'static'"
        assert waypoint_type in ("random", "central"), \
            "waypoint_type must be 'random' or 'central'"
        if transition is None:
            # This class's ORIGINAL two behaviors, preserved for any caller
            # not yet passing transition explicitly -- see class docstring.
            transition = "smooth" if cluster_mobility_mode == "random" else "jump"
        assert transition in ("jump", "smooth"), "transition must be 'jump' or 'smooth'"
        if cluster_mobility_mode == "random" and transition == "jump":
            assert waypoint_hold_steps is not None and waypoint_hold_steps >= 1, \
                "waypoint_hold_steps (>=1) is required for cluster_mobility_mode='random' " \
                "with transition='jump' (bounds its random per-group dwell time)"
        if cluster_mobility_mode in ("periodic", "markov"):
            assert num_waypoints is not None and num_waypoints >= 2, \
                "num_waypoints (>=2) is required for periodic/markov cluster_mobility_mode"
            assert waypoint_hold_steps is not None and waypoint_hold_steps >= 1, \
                "waypoint_hold_steps (>=1) is required for periodic/markov cluster_mobility_mode"
            if transition == "smooth":
                assert transition_duration_s is not None and transition_duration_s > 0, \
                    "transition_duration_s (>0) is required for periodic/markov with transition='smooth'"
            if waypoint_type == "central" and waypoints is None:
                assert (region_centers is not None and region_centers_valid is not None
                       and cluster_home_sector is not None), \
                    "waypoint_type='central' needs region_centers, region_centers_valid and cluster_home_sector"
                min_valid_regions = int(torch.as_tensor(region_centers_valid).sum(dim=1).min())
                assert num_waypoints <= min_valid_regions, \
                    (f"num_waypoints ({num_waypoints}) cannot exceed the number of valid regions "
                    f"in the WORST-case sector ({min_valid_regions}) for waypoint_type='central'")

        # Initialization
        self.dtype, self.device, self.generator = ut_loc.dtype, ut_loc.device, generator
        self.deviation_radius = deviation_radius
        self.topo = topo
        self._sampler = UeDropper(topo, self.dtype, self.device, generator)
        self.min_speed, self.max_speed = min_speed, max_speed
        self.member_jitter_speed = member_jitter_speed
        self.member_group_idx = member_group_idx
        self.cluster_mobility_mode = cluster_mobility_mode
        self.transition = transition
        self.transition_duration_s = transition_duration_s
        self.intra_cluster_mobility = intra_cluster_mobility
        self._step_count = 0

        # Reference points
        self.ref_xy = torch.tensor(start_xy_list, dtype=self.dtype, device=self.device)  # [num_groups, 2]
        num_groups = self.ref_xy.shape[0]

        if cluster_mobility_mode == "random":
            # Every group gets an initial random destination + speed --
            # used directly for transition="smooth" (interpolation target),
            # and as the "jump" target the first time a group's dwell ends.
            self.dest_xy = self.ref_xy.clone()
            self.speed = torch.zeros(num_groups, dtype=self.dtype, device=self.device)
            for i in range(num_groups):
                self._pick_new_waypoint(i)
            if transition == "jump":
                self.waypoint_hold_steps = waypoint_hold_steps
                self.hold_counter = torch.zeros(num_groups, dtype=torch.long, device=self.device)
                self.hold_duration = torch.randint(1, waypoint_hold_steps + 1, (num_groups,),
                                                   generator=self.generator, device=self.device)
        else:
            self.num_waypoints = num_waypoints
            self.waypoint_hold_steps = waypoint_hold_steps
            if cluster_mobility_mode == "markov":
                matrix = torch.as_tensor(waypoint_transition_matrix, dtype=self.dtype, device=self.device)
                assert matrix.shape == (num_waypoints, num_waypoints), \
                    f"waypoint_transition_matrix must be {num_waypoints}x{num_waypoints}, got {tuple(matrix.shape)}"
                row_sums = matrix.sum(dim=1)
                assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
                    "waypoint_transition_matrix rows must each sum to 1"
                self.waypoint_transition_matrix = matrix
            if waypoints is not None:
                # .clone() -- torch.as_tensor is a no-op (returns the SAME
                # tensor object) when waypoints is already a torch.Tensor of
                # this dtype/device, so writing self.waypoints[:, 0, :]
                # below would otherwise mutate the caller's own tensor.
                self.waypoints = torch.as_tensor(waypoints, dtype=self.dtype, device=self.device).clone()
                assert self.waypoints.shape == (num_groups, num_waypoints, 2), \
                    f"waypoints must be [{num_groups}, {num_waypoints}, 2], got {tuple(self.waypoints.shape)}"
                # start_xy_list is always waypoint 0 (class docstring) -- see
                # the "central" branch below for why this must hold even
                # when explicit waypoints are given.
                self.waypoints[:, 0, :] = self.ref_xy
            elif waypoint_type == "central":
                # num_waypoints chosen at random (no replacement) from this
                # group's home sector's VALID region centroids only --
                # region_centers_valid excludes whichever windows sit too
                # close to a site for a CR_max-radius cluster (varies per
                # sector's real geometry, not a fixed per-window rule).
                region_centers_t = torch.as_tensor(region_centers, dtype=self.dtype, device=self.device)
                valid_t = torch.as_tensor(region_centers_valid, dtype=torch.bool, device=self.device)
                home_sector_t = torch.as_tensor(cluster_home_sector, dtype=torch.long, device=self.device)
                if (home_sector_t < 0).any():
                    logger.warning("ReferencePointGroupMobility: %d group(s) fell outside every sector's "
                                  "rhombus -- falling back to sector 0's regions for those",
                                  int((home_sector_t < 0).sum()))
                    home_sector_t = torch.where(home_sector_t < 0, torch.zeros_like(home_sector_t), home_sector_t)
                self.waypoints = torch.zeros(num_groups, num_waypoints, 2, dtype=self.dtype, device=self.device)
                for g in range(num_groups):
                    sector = home_sector_t[g]
                    candidates = region_centers_t[sector][valid_t[sector]]  # [num_valid, 2]
                    perm = torch.randperm(candidates.shape[0], generator=self.generator,
                                          device=self.device)[:num_waypoints]
                    self.waypoints[g] = candidates[perm]
                # start_xy_list is always waypoint 0 (class docstring) --
                # without this, current_waypoint_idx=0 claims the group
                # starts at waypoints[0], but the actual reference/members
                # sit at start_xy_list, a different, independently-sampled
                # point; the first transition would then jump straight to
                # waypoints[1], skipping waypoints[0] until it cycles back.
                self.waypoints[:, 0, :] = self.ref_xy
            else:
                # Auto-generate, constrained near each group's OWN starting
                # position (one hex cell_radius) so hotspots stay within
                # roughly the same site rather than roaming the network.
                cell_radius = float(self.topo.grid.cell_radius.item())
                min_dist = (waypoint_min_dist_from_site if waypoint_min_dist_from_site is not None
                           else self.topo.min_bs_ut_dist)
                self.waypoints = torch.zeros(num_groups, num_waypoints, 2, dtype=self.dtype, device=self.device)
                self.waypoints[:, 0, :] = self.ref_xy
                for w in range(1, num_waypoints):
                    self.waypoints[:, w, :] = self._sampler.valid_points(
                        self.ref_xy, cell_radius, min_dist_from_site=min_dist)
            self.current_waypoint_idx = torch.zeros(num_groups, dtype=torch.long, device=self.device)
            self.hold_counter = torch.zeros(num_groups, dtype=torch.long, device=self.device)
            if transition == "smooth":
                # Walking state toward the next waypoint once a group's
                # dwell ends -- unused (stays all-False/zero) while holding.
                self.dest_xy = self.ref_xy.clone()
                self.speed = torch.zeros(num_groups, dtype=self.dtype, device=self.device)
                self._in_transit = torch.zeros(num_groups, dtype=torch.bool, device=self.device)

        # Per-member state
        self._z = ut_loc[:, 2:3].clone()  # persistent per-member height, carried as-is
        ref_per_member = self.ref_xy[self.member_group_idx]
        self.member_offset = ut_loc[:, :2] - ref_per_member  # each member's offset, as given
        # "static" only: which members are currently frozen at their last
        # valid position because ref + fixed offset is invalid -- tracked
        # to log a freeze episode once instead of once per slot.
        self._static_frozen = torch.zeros(ut_loc.shape[0], dtype=torch.bool, device=self.device)

        super().__init__(ut_loc)
        logger.info("ReferencePointGroupMobility constructed: num_groups=%d deviation_radius=%.2f "
                   "cluster_mobility_mode=%s waypoint_type=%s intra_cluster_mobility=%s speed=[%.2f, %.2f]",
                   num_groups, deviation_radius, cluster_mobility_mode,
                   waypoint_type if cluster_mobility_mode != "random" else "n/a",
                   intra_cluster_mobility, min_speed, max_speed)

    def _is_valid(self, xy: torch.Tensor) -> torch.Tensor:
        """True where xy is both within the real coverage footprint and at
        least min_bs_ut_dist from every BS site
        """
        logger.function("_is_valid start: xy shape=%s", tuple(xy.shape))
        within_coverage = self.topo.is_within_coverage(xy)
        site_loc = self.topo.site_loc.to(dtype=xy.dtype, device=xy.device)
        nearest_dist = torch.linalg.norm(xy[:, None, :] - site_loc[None, :, :], dim=-1).min(dim=-1).values
        valid = within_coverage & (nearest_dist >= self.topo.min_bs_ut_dist)
        logger.debug("_is_valid: %d/%d valid", int(valid.sum()), valid.numel())
        logger.function("_is_valid end")
        return valid

    def _pick_new_waypoint(self, group_idx: int) -> None:
        """Draws a new random destination (within the topology's actual
        coverage footprint) and speed for a group's reference point."""
        logger.function("_pick_new_waypoint start: group_idx=%d", group_idx)
        origin = torch.zeros(1, 2, dtype=self.dtype, device=self.device)
        dest = self._sampler.valid_points(origin, self.topo.default_drop_radius)
        self.dest_xy[group_idx] = dest[0]
        u = torch.rand((), dtype=self.dtype, device=self.device, generator=self.generator)
        self.speed[group_idx] = self.min_speed + (self.max_speed - self.min_speed) * u
        logger.debug("_pick_new_waypoint: group_idx=%d dest=%s speed=%.2f",
                    group_idx, dest[0].tolist(), self.speed[group_idx].item())
        logger.function("_pick_new_waypoint end")

    def _next_waypoint_idx(self, group_idx: int) -> int:
        """Which waypoint index a "periodic"/"markov" group moves to next
        -- "periodic" advances cyclically, "markov" samples from its
        transition matrix row. Does not touch position/state; callers
        (jump: instant, smooth: start walking) apply it differently."""
        old_idx = int(self.current_waypoint_idx[group_idx])
        if self.cluster_mobility_mode == "periodic":
            new_idx = (old_idx + 1) % self.num_waypoints
        else:
            probs = self.waypoint_transition_matrix[old_idx]
            new_idx = int(torch.multinomial(probs, 1, generator=self.generator).item())
        return new_idx

    def _transition_waypoint(self, group_idx: int) -> None:
        """transition="jump": instant hotspot teleport for one group."""
        old_idx = int(self.current_waypoint_idx[group_idx])
        new_idx = self._next_waypoint_idx(group_idx)
        self.current_waypoint_idx[group_idx] = new_idx
        self.ref_xy[group_idx] = self.waypoints[group_idx, new_idx]
        self.hold_counter[group_idx] = 0
        logger.info("waypoint transition: step=%d cluster_id=%d old_waypoint=%d new_waypoint=%d cluster_center=%s",
                   self._step_count, group_idx, old_idx, new_idx, self.ref_xy[group_idx].tolist())

    def _start_smooth_waypoint_transition(self, group_idx: int) -> None:
        """transition="smooth": begin walking toward the next hotspot
        (arrival happens over subsequent step() calls, see step()).
        Speed is set so arrival takes exactly transition_duration_s,
        regardless of distance -- not drawn from [min_speed, max_speed]."""
        old_idx = int(self.current_waypoint_idx[group_idx])
        new_idx = self._next_waypoint_idx(group_idx)
        self.current_waypoint_idx[group_idx] = new_idx
        self.dest_xy[group_idx] = self.waypoints[group_idx, new_idx]
        distance = torch.linalg.norm(self.dest_xy[group_idx] - self.ref_xy[group_idx])
        self.speed[group_idx] = distance / self.transition_duration_s
        self._in_transit[group_idx] = True
        logger.info("waypoint transition (smooth) started: step=%d cluster_id=%d old_waypoint=%d "
                   "new_waypoint=%d speed=%.2f", self._step_count, group_idx, old_idx, new_idx,
                   self.speed[group_idx].item())

    def _walk_toward(self, i: int, dest: torch.Tensor, speed: torch.Tensor, dt: float) -> bool:
        """Moves ref_xy[i] one step toward dest at the given speed, clamped
        to arrival; returns True once it has arrived exactly at dest."""
        to_dest = dest - self.ref_xy[i]
        dist = torch.linalg.norm(to_dest)
        step_len = speed * dt
        candidate = dest if dist <= step_len else self.ref_xy[i] + to_dest / dist * step_len
        self.ref_xy[i] = candidate
        return bool(dist <= step_len)

    def step(self, dt: float) -> torch.Tensor:
        logger.function("ReferencePointGroupMobility.step start: dt=%.2f", dt)
        self._step_count += 1

        # Cluster center movement
        if self.cluster_mobility_mode == "random" and self.transition == "smooth":
            # Unchanged original behavior: continuous walk, immediate
            # re-pick (no dwell) right on arrival -- "random" has no
            # hotspot to hold at.
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
        elif self.cluster_mobility_mode == "random":  # transition == "jump"
            # Hold at the current point for a per-group RANDOM dwell (drawn
            # in [1, waypoint_hold_steps] -- there's no fixed hotspot here,
            # so the dwell time itself is randomized instead), then
            # teleport to a fresh random point and redraw the dwell.
            self.hold_counter += 1
            for i in torch.where(self.hold_counter >= self.hold_duration)[0].tolist():
                self._pick_new_waypoint(i)
                self.ref_xy[i] = self.dest_xy[i]
                self.hold_counter[i] = 0
                self.hold_duration[i] = torch.randint(1, self.waypoint_hold_steps + 1, (1,),
                                                      generator=self.generator, device=self.device)[0]
        elif self.transition == "jump":  # "periodic"/"markov"
            # Unchanged original behavior: hold for a fixed
            # waypoint_hold_steps, then instant teleport.
            self.hold_counter += 1
            for i in torch.where(self.hold_counter >= self.waypoint_hold_steps)[0].tolist():
                self._transition_waypoint(i)
        else:  # "periodic"/"markov" + transition == "smooth"
            # Unlike "random"'s walk above, this doesn't re-check
            # is_within_coverage per step -- both endpoints (current
            # waypoint, next waypoint) are already known-valid, and hotspots
            # are near each other (same/nearby sector), so a straight-line
            # path between them leaving the footprint is a real but unlikely
            # edge case, not checked here.
            for i in range(self.ref_xy.shape[0]):
                if bool(self._in_transit[i]):
                    if self._walk_toward(i, self.dest_xy[i], self.speed[i], dt):
                        self._in_transit[i] = False
                        self.hold_counter[i] = 0
                else:
                    self.hold_counter[i] += 1
                    if self.hold_counter[i] >= self.waypoint_hold_steps:
                        self._start_smooth_waypoint_transition(i)
        logger.debug("ReferencePointGroupMobility.step: ref_xy updated for %d groups", self.ref_xy.shape[0])

        # Members drift by a small random step from their PREVIOUS offset,
        # or keep it fixed -- either way they still move WITH whatever the
        # reference point just did, since position is always reference + offset.
        ref_per_member = self.ref_xy[self.member_group_idx]
        num_ut = ref_per_member.shape[0]
        if self.intra_cluster_mobility == "static":
            new_offset = self.member_offset.clone()
        else:
            jitter = self._sampler.disk_offset(self.member_jitter_speed * dt, num_ut)
            new_offset = self.member_offset + jitter
            mag = torch.linalg.norm(new_offset, dim=-1)
            too_far = mag > self.deviation_radius
            new_offset[too_far] = new_offset[too_far] / mag[too_far, None] * self.deviation_radius
            logger.debug("ReferencePointGroupMobility.step: %d/%d members clamped to deviation_radius",
                        int(too_far.sum()), num_ut)

        # Validate proposed positions
        xy = ref_per_member + new_offset
        ok = self._is_valid(xy)

        # Reuse the previous offset with the NEW reference point.
        fallback_xy = ref_per_member + self.member_offset
        fallback_ok = self._is_valid(fallback_xy)
        use_fallback = (~ok) & fallback_ok
        xy[use_fallback] = fallback_xy[use_fallback]
        new_offset[use_fallback] = self.member_offset[use_fallback]
        logger.debug("ReferencePointGroupMobility.step: %d members fell back to previous offset",
                    int(use_fallback.sum()))

        # Persistently invalid members: "static" must never resample a new
        # offset at a random new location (that was the original bug --
        # silently changing the promised-fixed cluster geometry). Instead,
        # freeze the member's ABSOLUTE position and recompute its offset
        # from that frozen point (offset = frozen_xy - moving_ref) so the
        # class's own ut_loc == ref + offset invariant always holds -- but
        # freezing alone can drag the offset arbitrarily far outside
        # deviation_radius if the reference keeps moving while the member
        # stays put, breaking the OTHER promise (member stays within its
        # cluster disk). So pull the frozen offset back to deviation_radius
        # (same radial clamp random_walk's own jitter already uses) and
        # re-validate; only if even the clamped point is invalid does this
        # fall back to a bounded random resample around the reference (same
        # as "random_walk"'s own fallback below) -- a last resort, not the
        # default, so most freeze episodes still preserve the exact offset.
        # "random_walk" has no fixed-offset invariant to protect in the
        # first place, so it always resamples around the reference.
        still_bad = (~ok) & (~fallback_ok)
        if self.intra_cluster_mobility == "static":
            # A member can need this intervention for as long as the
            # reference sits at a hotspot that makes its fixed offset
            # invalid (up to a full waypoint_hold_steps) -- warn once per
            # episode, not once per slot, or this drowns out every other
            # log line.
            newly_frozen = still_bad & ~self._static_frozen
            if newly_frozen.any():
                logger.warning("ReferencePointGroupMobility.step: %d static member(s) newly held at "
                              "previous position, clamped to deviation_radius of the moving reference "
                              "(both proposed and fallback positions invalid); further consecutive "
                              "slots log at DEBUG", int(newly_frozen.sum()))
            continuing_frozen = still_bad & self._static_frozen
            if continuing_frozen.any():
                logger.debug("ReferencePointGroupMobility.step: %d static member(s) remain held/clamped",
                            int(continuing_frozen.sum()))
            self._static_frozen = still_bad.clone()

            clamped_offset = self.ut_loc[still_bad, :2] - ref_per_member[still_bad]
            mag = torch.linalg.norm(clamped_offset, dim=-1)
            too_far = mag > self.deviation_radius
            clamped_offset[too_far] = clamped_offset[too_far] / mag[too_far, None] * self.deviation_radius
            clamped_xy = ref_per_member[still_bad] + clamped_offset
            clamped_ok = self._is_valid(clamped_xy)
            xy[still_bad] = clamped_xy
            new_offset[still_bad] = clamped_offset

            still_invalid = still_bad.clone()
            still_invalid[still_bad] = ~clamped_ok
            if still_invalid.any():
                resampled_xy = self._sampler.valid_points(
                    ref_per_member[still_invalid], self.deviation_radius,
                    min_dist_from_site=self.topo.min_bs_ut_dist,
                )
                xy[still_invalid] = resampled_xy
                new_offset[still_invalid] = resampled_xy - ref_per_member[still_invalid]
                logger.warning("ReferencePointGroupMobility.step: %d static member(s) resampled -- even "
                              "the radius-clamped position was invalid", int(still_invalid.sum()))
        elif still_bad.any():
            resampled_xy = self._sampler.valid_points(
                ref_per_member[still_bad], self.deviation_radius,
                min_dist_from_site=self.topo.min_bs_ut_dist,
            )
            xy[still_bad] = resampled_xy
            new_offset[still_bad] = resampled_xy - ref_per_member[still_bad]
            logger.warning("ReferencePointGroupMobility.step: %d members resampled (both proposed and "
                          "fallback positions invalid)", int(still_bad.sum()))

        self.member_offset = new_offset
        self.ut_loc = torch.cat([xy, self._z], dim=-1)
        logger.function("ReferencePointGroupMobility.step end")
        return self.ut_loc

