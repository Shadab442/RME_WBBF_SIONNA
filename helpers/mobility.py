"""Mobility models for the tilt-over-time study.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

from .ue_drop import UeDropper
from helpers.utils import get_logger

logger = get_logger(__name__)

_HOTSPOT_REGION_TYPES = ("near", "mid", "far")


class MobilityModel:
    """Common interface: owns current UE positions, advances them one step
    at a time (one step == one tilt-control interval -- see
    ReferencePointGroupMobility, the only implementation).

    :ivar ut_loc: [num_ut, 3] current UE positions [m].
    """

    def __init__(self, ut_loc: torch.Tensor):
        self.ut_loc = ut_loc

    def step(self) -> torch.Tensor:
        """Advance to the next interval; updates and returns self.ut_loc."""
        raise NotImplementedError


class ReferencePointGroupMobility(MobilityModel):
    """Reference Point Group Mobility (Hong, Gerla, Pei & Chiang, 1999):
    each group has a reference point; members hold a PERSISTENT offset from
    it that random-walks by a small member_walk_step each step() (rejection-
    sampled via UeDropper.walk_points, reflecting off the deviation_radius
    disk boundary/coverage footprint/site-exclusion zone), so consecutive
    intervals are correlated instead of independently redrawn. The one
    exception: a group whose reference point ITSELF transitions this step
    (a fresh teleport -- see cluster_mobility_mode below) gets its members'
    offsets fully re-resampled within deviation_radius of the new reference
    point (UeDropper.valid_points) -- a freshly-formed cluster at a new
    hotspot has no meaningful continuity with its old member layout. There
    is no intra-interval member movement either way, since exactly one
    measurement is taken per tilt-control interval (see
    helpers/simulation_engine.py): there's nothing for continuous
    within-interval movement to be observed against.

    cluster_mobility_mode -- WHERE each group's reference point is headed:
      "random"   (baseline) -- holds for a random per-group dwell (drawn
                 uniformly in [1, waypoint_hold_steps] each cycle -- there's
                 no fixed hotspot here, so the dwell time itself is
                 randomized), then teleports to a fresh random destination
                 within the topology's coverage footprint.
      "periodic" / "markov" -- each group has num_waypoints predefined
                 (x, y) hotspots; holds for a FIXED waypoint_hold_steps,
                 then teleports to the NEXT one, chosen cyclically
                 (0->1->2->0->...) for "periodic", or sampled from
                 waypoint_transition_matrix's row for "markov". Waypoint 0
                 is always the group's own starting position (start_xy_list);
                 waypoints 1.. are auto-generated (unless explicit
                 `waypoints` are given) via the SAME near/mid/far
                 hotspot-region-driven sampling HotspotDemandMobility uses
                 for its own clusters (UeDropper.hotspot_region_points,
                 CellularTopology.hotspot_region_membership -- the ONE
                 region definition in this codebase): waypoint w's region
                 cycles through _HOTSPOT_REGION_TYPES (near, mid, far), a
                 FRESH point sampled inside the group's own home sector
                 (cluster_home_sector) every time -- never a fixed/reused
                 location.

    Every transition is instantaneous (a "jump"/teleport) -- there is no
    continuous/"smooth" walk between waypoints.
    """

    def __init__(
        self,
        ut_loc: torch.Tensor,
        member_group_idx: torch.Tensor,
        start_xy_list,
        deviation_radius: float,
        member_walk_step: float,
        topo,
        generator=None,
        cluster_mobility_mode: str = "random",
        num_waypoints: int = None,
        waypoint_hold_steps: int = None,
        waypoint_transition_matrix=None,
        waypoints=None,
        cluster_home_sector=None,
        waypoint_min_dist_from_site: float = None,
    ):
        """
        :param ut_loc: [num_ut, 3] initial member positions -- e.g. from
            ``helpers.ue_drop.UeDropper.clustered``. Only its per-member
            height (z) is retained; (x, y) is freshly resampled every step().
        :param member_group_idx: [num_ut] which group (an index into
            ``start_xy_list``) each member belongs to -- from the same drop
            call that produced ``ut_loc``.
        :param start_xy_list: one ``(x, y)`` per group -- its reference
            point's starting position, matching the cluster center used to
            generate that group's members. Also waypoint 0 when
            cluster_mobility_mode is "periodic"/"markov".
        :param deviation_radius: max per-member offset from the reference
            point [m] -- see class docstring for how this controls cluster size.
        :param member_walk_step: per-step random-walk radius [m] for a
            member's persistent offset from its (non-transitioning) group's
            reference point -- see class docstring. Small relative to
            deviation_radius (e.g. a fraction of it) so consecutive
            intervals stay correlated; 0 degenerates to a fixed offset
            (members never move relative to their reference point at all).
        :param topo: the scenario's ``CellularTopology`` -- used for its
            real hex-footprint boundary test (``is_within_coverage``) and
            ``default_drop_radius``
        :param cluster_mobility_mode: "random" | "periodic" | "markov".
        :param num_waypoints: per-group hotspot count -- required for
            "periodic"/"markov".
        :param waypoint_hold_steps: number of step() calls (== tilt-control
            intervals) to hold before transitioning -- required for
            "periodic"/"markov", and for cluster_mobility_mode="random"
            (there it bounds a PER-GROUP random dwell, drawn uniformly in
            [1, waypoint_hold_steps] each cycle, rather than being used as a
            fixed duration directly -- see class docstring).
        :param waypoint_transition_matrix: [num_waypoints, num_waypoints],
            rows summing to 1 -- required for "markov".
        :param waypoints: optional [num_groups, num_waypoints, 2] explicit
            per-group hotspot coordinates -- if given, overrides
            auto-generation entirely (cluster_home_sector not needed).
        :param cluster_home_sector: [num_groups] int -- which sector each
            group's starting position falls in (CellularTopology.
            assign_to_sector_rhombus_grid). Required for "periodic"/"markov"
            when `waypoints` isn't given -- every auto-generated waypoint is
            sampled inside the group's OWN home sector.
        :param waypoint_min_dist_from_site: defaults to topo.min_bs_ut_dist.
            Pass a buffered value (e.g. min_bs_ut_dist + a cluster radius
            cap) so the WHOLE cluster disk around an auto-generated waypoint
            stays outside the exclusion zone, not just the waypoint point
            itself.
        """
        assert cluster_mobility_mode in ("random", "periodic", "markov"), \
            "cluster_mobility_mode must be 'random', 'periodic' or 'markov'"
        assert waypoint_hold_steps is not None and waypoint_hold_steps >= 1, \
            "waypoint_hold_steps (>=1) is required (bounds cluster_mobility_mode='random''s own random " \
            "per-group dwell, or is the fixed dwell for periodic/markov)"
        if cluster_mobility_mode in ("periodic", "markov"):
            assert num_waypoints is not None and num_waypoints >= 2, \
                "num_waypoints (>=2) is required for periodic/markov cluster_mobility_mode"
            if waypoints is None:
                assert cluster_home_sector is not None, \
                    "cluster_home_sector is required for auto-generated periodic/markov waypoints " \
                    "(unless explicit waypoints are given)"

        # Initialization
        self.dtype, self.device, self.generator = ut_loc.dtype, ut_loc.device, generator
        self.deviation_radius = deviation_radius
        self.member_walk_step = member_walk_step
        self.topo = topo
        self._sampler = UeDropper(topo, self.dtype, self.device, generator)
        self.member_group_idx = member_group_idx
        self._members_per_group = torch.bincount(member_group_idx).tolist()
        self.cluster_mobility_mode = cluster_mobility_mode
        self.waypoint_min_dist_from_site = (waypoint_min_dist_from_site if waypoint_min_dist_from_site is not None
                                            else topo.min_bs_ut_dist)
        self._step_count = 0

        # Reference points
        self.ref_xy = torch.tensor(start_xy_list, dtype=self.dtype, device=self.device)  # [num_groups, 2]
        num_groups = self.ref_xy.shape[0]
        self.hold_counter = torch.zeros(num_groups, dtype=torch.long, device=self.device)
        self.waypoint_hold_steps = waypoint_hold_steps

        if cluster_mobility_mode == "random":
            self.hold_duration = torch.randint(1, waypoint_hold_steps + 1, (num_groups,),
                                               generator=self.generator, device=self.device)
        else:
            self.num_waypoints = num_waypoints
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
            else:
                # Auto-generate: waypoint 0 is always the group's own
                # starting position (class docstring); waypoints 1.. are
                # FRESH points sampled inside the group's own home sector,
                # cycling through _HOTSPOT_REGION_TYPES (near, mid, far) --
                # the SAME UeDropper.hotspot_region_points machinery
                # HotspotDemandMobility uses for its own clusters, so both
                # mobility families place their hotspots via the one
                # canonical near/mid/far region definition
                # (CellularTopology.hotspot_region_membership) instead of
                # two different, unrelated geometries.
                home_sector_t = torch.as_tensor(cluster_home_sector, dtype=torch.long, device=self.device)
                if (home_sector_t < 0).any():
                    logger.warning("ReferencePointGroupMobility: %d group(s) fell outside every sector's "
                                  "rhombus -- falling back to sector 0 for those",
                                  int((home_sector_t < 0).sum()))
                    home_sector_t = torch.where(home_sector_t < 0, torch.zeros_like(home_sector_t), home_sector_t)
                self.waypoints = torch.zeros(num_groups, num_waypoints, 2, dtype=self.dtype, device=self.device)
                self.waypoints[:, 0, :] = self.ref_xy
                for w in range(1, num_waypoints):
                    region_type = _HOTSPOT_REGION_TYPES[w % len(_HOTSPOT_REGION_TYPES)]
                    for sector in torch.unique(home_sector_t).tolist():
                        idx = (home_sector_t == sector).nonzero(as_tuple=True)[0]
                        self.waypoints[idx, w, :] = self._sampler.hotspot_region_points(
                            sector, region_type, idx.numel(), self.deviation_radius,
                            min_dist_from_site=self.waypoint_min_dist_from_site)
            self.current_waypoint_idx = torch.zeros(num_groups, dtype=torch.long, device=self.device)

        # Persistent per-member height only -- (x, y) is resampled fresh every step().
        self._z = ut_loc[:, 2:3].clone()

        super().__init__(ut_loc)
        logger.info("ReferencePointGroupMobility constructed: num_groups=%d deviation_radius=%.2f "
                   "cluster_mobility_mode=%s", num_groups, deviation_radius, cluster_mobility_mode)

    def _next_waypoint_idx(self, group_idx: int) -> int:
        """Which waypoint index a "periodic"/"markov" group moves to next
        -- "periodic" advances cyclically, "markov" samples from its
        transition matrix row."""
        old_idx = int(self.current_waypoint_idx[group_idx])
        if self.cluster_mobility_mode == "periodic":
            new_idx = (old_idx + 1) % self.num_waypoints
        else:
            probs = self.waypoint_transition_matrix[old_idx]
            new_idx = int(torch.multinomial(probs, 1, generator=self.generator).item())
        return new_idx

    def _transition_waypoint(self, group_idx: int) -> None:
        """Instant hotspot teleport for one "periodic"/"markov" group."""
        old_idx = int(self.current_waypoint_idx[group_idx])
        new_idx = self._next_waypoint_idx(group_idx)
        self.current_waypoint_idx[group_idx] = new_idx
        self.ref_xy[group_idx] = self.waypoints[group_idx, new_idx]
        self.hold_counter[group_idx] = 0
        logger.info("waypoint transition: step=%d cluster_id=%d old_waypoint=%d new_waypoint=%d cluster_center=%s",
                   self._step_count, group_idx, old_idx, new_idx, self.ref_xy[group_idx].tolist())

    def step(self) -> torch.Tensor:
        logger.function("ReferencePointGroupMobility.step start")
        self._step_count += 1

        # Cluster reference point: hold, then instant teleport. `due` marks
        # which groups transition THIS step -- their members get a fresh
        # cluster (valid_points) below instead of a random-walk update.
        self.hold_counter += 1
        if self.cluster_mobility_mode == "random":
            due = self.hold_counter >= self.hold_duration
            for i in due.nonzero(as_tuple=True)[0].tolist():
                dest = self._sampler.valid_points(torch.zeros(1, 2, dtype=self.dtype, device=self.device),
                                                  self.topo.default_drop_radius)
                self.ref_xy[i] = dest[0]
                self.hold_counter[i] = 0
                self.hold_duration[i] = torch.randint(1, self.waypoint_hold_steps + 1, (1,),
                                                      generator=self.generator, device=self.device)[0]
                logger.info("waypoint transition: step=%d cluster_id=%d new_center=%s",
                           self._step_count, i, self.ref_xy[i].tolist())
        else:  # "periodic"/"markov"
            due = self.hold_counter >= self.waypoint_hold_steps
            for i in due.nonzero(as_tuple=True)[0].tolist():
                self._transition_waypoint(i)
        logger.debug("ReferencePointGroupMobility.step: ref_xy updated for %d groups", self.ref_xy.shape[0])

        # Members of a group that just transitioned: fresh cluster, fully
        # re-resampled within deviation_radius of the NEW reference point.
        # Every other member: persistent offset, perturbed by one small
        # random-walk step (reflected off the deviation_radius disk /
        # coverage footprint / site-exclusion zone) -- see class docstring.
        ref_per_member = self.ref_xy[self.member_group_idx]
        member_due = due[self.member_group_idx]
        xy = torch.empty_like(self.ut_loc[:, :2])
        fresh_idx = member_due.nonzero(as_tuple=True)[0]
        if fresh_idx.numel() > 0:
            xy[fresh_idx] = self._sampler.valid_points(ref_per_member[fresh_idx], self.deviation_radius,
                                                       min_dist_from_site=self.topo.min_bs_ut_dist)
        walk_idx = (~member_due).nonzero(as_tuple=True)[0]
        if walk_idx.numel() > 0:
            xy[walk_idx] = self._sampler.walk_points(
                self.ut_loc[walk_idx, :2], ref_per_member[walk_idx], self.deviation_radius,
                self.member_walk_step, min_dist_from_site=self.topo.min_bs_ut_dist)

        self.ut_loc = torch.cat([xy, self._z], dim=-1)
        logger.function("ReferencePointGroupMobility.step end")
        return self.ut_loc


def build_sector_biased_transition_matrix(num_states: int, sector_bias, bias_strength: float) -> np.ndarray:
    """[num_bs, num_states, num_states] per-sector Markov transition matrix
    for HotspotDemandMobility's own transition_matrix param -- sector s's
    own row is STICKY toward demand_states index sector_bias[s]
    (bias_strength probability of landing there on every transition,
    REGARDLESS of current state), the remainder split evenly across the
    other num_states-1 states.

    Creates genuine LONG-RUN per-sector demand heterogeneity: with the
    ordinary shared 2D transition_matrix (uniform default or otherwise),
    every sector's own independent chain converges to the SAME long-run
    state mix, just desynchronized in time (see HotspotDemandMobility's
    own class docstring) -- sectors differ in WHEN they're in each state,
    never in HOW OFTEN. This makes different sectors differ in how often
    too, e.g. some persistently mid-heavy, others persistently far-heavy
    -- so the truly-optimal tilt can genuinely vary sector to sector,
    unlike a homogeneous-demand scenario where one action (e.g. max
    downtilt) ends up close to universally correct (see
    project_config3_heterogeneous_demand memory for the diagnostic that
    motivated this).

    :param sector_bias: [num_bs] int, each sector's own preferred
        demand_states row index.
    :param bias_strength: probability mass on the preferred state, same
        for every sector -- 1/num_states reduces to the ordinary uniform
        default (no bias at all).
    """
    sector_bias = np.asarray(sector_bias, dtype=int)
    num_bs = len(sector_bias)
    other_prob = (1.0 - bias_strength) / (num_states - 1)
    matrix = np.full((num_bs, num_states, num_states), other_prob)
    for s in range(num_bs):
        matrix[s, :, sector_bias[s]] = bias_strength
    return matrix


class HotspotDemandMobility(MobilityModel):
    """PER-SECTOR spatial-demand-cycling mobility: `num_clusters_per_sector`
    clusters per sector, each holding `num_ues_per_cluster` members (an int,
    same for every cluster, or a per-cluster list for a near-even split
    when the total doesn't divide evenly -- see SimulationEngine's own
    derivation). Cluster
    IDENTITY (member UE assignment) is permanent; only which (sector,
    hotspot-type) a cluster is currently anchored to changes over time.

    Every `hotspot_hold_steps` intervals, EVERY sector independently
    advances its OWN demand state, per `state_transition_mode` -- not one
    shared network-wide clock (see class history: an earlier design
    advanced all 21 sectors through the same deterministic cyclic order in
    lockstep, which meant every sector entered its own worst state, e.g.
    far-heavy, SIMULTANEOUSLY -- no sector was ever in a better state at
    the same moment a neighbor could "borrow" capacity from, which
    collapses per-sector tilt intelligence to no better than a single
    network-wide tilt):
      "markov"   -- sector s's next state is sampled from
                 `transition_matrix[s's current state index]`, independently
                 per sector -- genuine per-sector randomness, restoring the
                 spatial diversity the lockstep design lacked.
      "periodic" -- sector s's next state is `(current_index + 1) %
                 num_states`, deterministic -- but each sector's STARTING
                 state is already an independent random draw (see
                 `_state_idx` init below), so different sectors begin at
                 different points in the SAME cycle and stay staggered
                 apart forever (a fixed offset, never drifting or
                 resynchronizing) -- deterministic and fully reproducible
                 per sector, useful for controlled comparisons (e.g. two
                 tilt controllers scored against the IDENTICAL demand
                 trajectory) at the cost of a less organic-looking pattern
                 than "markov"'s.
    `demand_states` lists the possible states
    (`(n_near, n_mid, n_far)`, each row summing to `num_clusters_per_sector`)
    -- only which state each sector is IN, and when it moves to a
    different one, is independent per sector. At each transition:
      1. Every cluster is re-matched to a (possibly different) sector --
         eligible only for its CURRENT sector or one of that sector's
         `neighbor_ids` (cross-site allowed) -- via a randomized
         `scipy.optimize.linear_sum_assignment` bipartite matching over the
         whole network at once (not per-sector/per-site independently,
         since eligibility overlaps between adjacent sectors). Feasibility
         is checked explicitly and raises RuntimeError if violated (see
         _match_clusters_to_sectors) rather than silently misassigning --
         though by construction ("stay in your own current sector" is
         always a feasible witness, since every sector always holds
         exactly num_clusters_per_sector clusters) this should never
         actually trigger in normal operation.
      2. Within each sector, its newly-matched clusters are randomly
         shuffled and sliced by THAT SECTOR's OWN newly-drawn state's
         (n_near,n_mid,n_far) counts -- pure random labeling, no matching
         needed (hotspot TYPE doesn't affect a cluster's sector-eligibility).
      3. Each cluster gets a FRESHLY sampled point (never a reused/fixed
         waypoint -- "the waypoints can be anywhere within the hotspot
         area") inside its new (sector, type) region
         (UeDropper.hotspot_region_points), then a full member redraw
         (UeDropper.valid_points) -- consistent with the fact that the ref
         point itself is fresh every time, even for a cluster that
         happens to land back in the same (sector, type).

    Between transitions: the same persistent-offset small random walk
    (UeDropper.walk_points) ReferencePointGroupMobility uses.

    Hotspot regions are computed from each sector's OWN rhombus geometry
    only (CellularTopology.hotspot_region_membership) -- no neighbor-sector
    extension, and no separate threshold config -- region shape is fixed
    by the near/mid/far rule and the sector's own fine-grid resolution.

    :ivar cluster_sector: [num_clusters] int, each cluster's CURRENT sector.
    :ivar cluster_type: [num_clusters] str ("near"/"mid"/"far"), each
        cluster's CURRENT hotspot type.
    :ivar ref_xy: [num_clusters, 2] each cluster's current reference point.
    :ivar _state_idx: [num_bs] int, each sector's OWN current index into
        demand_states -- per-sector, not a single shared scalar.
    """

    def __init__(self, topo, generator, num_clusters_per_sector: int, num_ues_per_cluster,
                ut_height: float, deviation_radius: float, member_walk_step: float,
                hotspot_hold_steps: int, demand_states, state_transition_mode: str = "markov",
                transition_matrix=None, dtype=None, device=None, deviation_radius_tangential: float = None):
        """
        :param deviation_radius: the RADIAL semi-axis of each cluster's
            member-scatter ellipse (the axis near/mid/far region membership
            actually cares about -- see hotspot_region_membership, purely a
            radial-distance cutoff). Named without "_radial" for backward
            compatibility -- this was the sole (circular) scatter radius
            before ellipse support was added.
        :param deviation_radius_tangential: the other (azimuthal) semi-axis.
            None (default): equals deviation_radius, i.e. the original
            isotropic circular scatter.
        :param demand_states: [num_states, len(_HOTSPOT_REGION_TYPES)] --
            (n_near, n_mid, n_far) per sector, each row summing to
            num_clusters_per_sector -- the set of states EVERY sector's own
            independent chain draws from.
        :param state_transition_mode: "markov" | "periodic" -- see class
            docstring. "periodic" ignores transition_matrix entirely (its
            advance rule is fixed: +1 mod num_states).
        :param transition_matrix: [num_states, num_states] row-stochastic
            -- "markov" only: sector s's next state is sampled from
            transition_matrix[s's current state index], independently per
            sector, every hotspot_hold_steps intervals (same
            row-per-current-state Markov convention as
            ReferencePointGroupMobility's own waypoint_transition_matrix).
            Defaults to UNIFORM (every state equally likely regardless of
            current state) when not given -- the simplest choice that
            still guarantees per-sector independence/desynchronization,
            with no extra tuning required.
        """
        assert state_transition_mode in ("markov", "periodic"), \
            "state_transition_mode must be 'markov' or 'periodic'"
        dtype = dtype if dtype is not None else topo.bs_loc.dtype
        device = device if device is not None else topo.bs_loc.device
        self.dtype, self.device, self.generator = dtype, device, generator
        self.topo = topo
        self._sampler = UeDropper(topo, dtype, device, generator)
        self.deviation_radius = deviation_radius
        self.deviation_radius_tangential = (deviation_radius_tangential if deviation_radius_tangential is not None
                                            else deviation_radius)
        self.member_walk_step = member_walk_step
        self.hotspot_hold_steps = hotspot_hold_steps
        self.state_transition_mode = state_transition_mode
        # Generic-caller parity: e.g. save_mobility_animation/
        # debug_pipeline_audit.py read this attribute directly off whatever
        # mobility instance is in use, same as ReferencePointGroupMobility.
        # Compound form (2026-10-01) matches config.yaml's own unified
        # cluster_mobility_mode naming -- "hotspot_demand-<submode>", not
        # the family name alone -- see SimulationEngine's own parsing.
        self.cluster_mobility_mode = f"hotspot_demand-{state_transition_mode}"

        demand_states = np.asarray(demand_states, dtype=int)
        num_region_types = len(_HOTSPOT_REGION_TYPES)
        assert demand_states.ndim == 2 and demand_states.shape[1] == num_region_types, \
            f"demand_states must be [num_states, {num_region_types}] {_HOTSPOT_REGION_TYPES}"
        assert np.all(demand_states.sum(axis=1) == num_clusters_per_sector), \
            f"every demand_states row must sum to num_clusters_per_sector ({num_clusters_per_sector})"
        self.demand_states = demand_states
        self.num_states = demand_states.shape[0]

        if state_transition_mode == "markov":
            if transition_matrix is None:
                transition_matrix = np.full((self.num_states, self.num_states), 1.0 / self.num_states)
            matrix = torch.as_tensor(transition_matrix, dtype=torch.float64, device=self.device)
            if matrix.dim() == 3:
                # Per-sector transition matrices [num_bs, num_states,
                # num_states] -- e.g. build_sector_biased_transition_matrix
                # below. Every sector's own independent chain still
                # desynchronizes in TIME (see class docstring), but now can
                # also differ in its own LONG-RUN state distribution --
                # unlike the shared 2D matrix, where every sector's chain
                # converges to the identical mix over time.
                assert matrix.shape == (topo.num_bs, self.num_states, self.num_states), \
                    (f"per-sector transition_matrix must be {topo.num_bs}x{self.num_states}x"
                     f"{self.num_states}, got {tuple(matrix.shape)}")
                row_sums = matrix.sum(dim=2)
            else:
                assert matrix.shape == (self.num_states, self.num_states), \
                    f"transition_matrix must be {self.num_states}x{self.num_states}, got {tuple(matrix.shape)}"
                row_sums = matrix.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
                "transition_matrix rows must each sum to 1"
            self.transition_matrix = matrix

        self._step_count = 0
        self.transitioned_this_step = False
        self.num_clusters_per_sector = num_clusters_per_sector
        num_bs = topo.num_bs
        num_clusters = num_bs * num_clusters_per_sector
        self.num_clusters = num_clusters

        # Per-sector INITIAL state, via randomized greedy graph coloring
        # over REAL cross-site neighbor adjacency (topo.neighbor_ids) --
        # guarantees no two real geometric neighbors ever start in the SAME
        # state (2026-10-12 fix). A plain per-sector independent uniform
        # draw (the old approach) only makes desync LIKELY on average
        # across the whole network, not guaranteed for any specific
        # neighbor pair -- confirmed empirically: sector 0 and BOTH its own
        # real neighbors landing on the identical starting phase purely by
        # chance under one seed, meaning that sector's demand state (and
        # its neighbors') moved in permanent lockstep for the entire run
        # (periodic mode never resynchronizes OR drifts apart -- a fixed
        # relative offset forever, see class docstring), silently
        # collapsing the very own-vs-neighbor demand DIVERSITY this
        # per-sector-independent design exists to provide.
        #
        # Guaranteed to succeed, no backtracking/failure case: standard
        # greedy-coloring bound -- processing vertices in ANY order, at
        # most degree(v) <= max_degree of a vertex's neighbors are already
        # colored by the time it's colored, so (max_degree + 1) colors
        # always suffice regardless of order. This project's own hex-grid
        # sector-adjacency graph has max degree 2 (confirmed: every sector
        # has at most 2 real cross-site neighbors), so num_states=3 colors
        # (>= max_degree + 1 = 3) is always enough -- asserted below so a
        # future change that breaks this assumption fails loudly instead
        # of silently.
        neighbor_ids_np = topo.neighbor_ids
        max_degree = int((neighbor_ids_np >= 0).sum(axis=1).max()) if num_bs > 0 else 0
        assert self.num_states >= max_degree + 1, (
            f"demand_states needs at least {max_degree + 1} states to guarantee a valid "
            f"starting-phase coloring for this topology's own max sector degree "
            f"({max_degree}), got {self.num_states}")
        order = torch.randperm(num_bs, generator=self.generator, device=self.device).cpu().numpy()
        state_idx = np.full(num_bs, -1, dtype=int)
        for s in order:
            used = {int(state_idx[n]) for n in neighbor_ids_np[s] if n >= 0 and state_idx[n] >= 0}
            available = [c for c in range(self.num_states) if c not in used]
            # Randomize WHICH available color is picked (not just the
            # lowest index) so different seeds produce genuinely different
            # valid colorings, not the same greedy pattern every run.
            pick = int(torch.randint(len(available), (1,), generator=self.generator, device=self.device).item())
            state_idx[s] = available[pick]
        self._state_idx = state_idx

        self.cluster_sector = np.repeat(np.arange(num_bs), num_clusters_per_sector)
        counts_per_sector = demand_states[self._state_idx]  # [num_bs, 3]
        self.cluster_type = np.concatenate(
            [np.repeat(np.array(_HOTSPOT_REGION_TYPES, dtype=object), counts_per_sector[s])
             for s in range(num_bs)])

        self.ref_xy = self._sample_all_ref_points(self.cluster_sector, self.cluster_type)
        ref_xy_list = self.ref_xy.tolist()
        ut_loc, member_group_idx = self._sampler.clustered(
            ref_xy_list, num_ues_per_cluster, deviation_radius, ut_height,
            cluster_sector=self.cluster_sector, tangential_radius=self.deviation_radius_tangential)
        self.member_group_idx = member_group_idx
        self._z = ut_loc[:, 2:3].clone()

        super().__init__(ut_loc)
        # num_ues_per_cluster: int (same for every cluster) or a
        # length-num_clusters list (near-even split, SimulationEngine's own
        # default now) -- logged as a plain value or a min-max range either way.
        if isinstance(num_ues_per_cluster, int):
            ues_desc = str(num_ues_per_cluster)
        else:
            ues_lo, ues_hi = min(num_ues_per_cluster), max(num_ues_per_cluster)
            ues_desc = str(ues_lo) if ues_lo == ues_hi else f"{ues_lo}-{ues_hi}"
        logger.info("HotspotDemandMobility constructed: num_bs=%d num_clusters_per_sector=%d "
                   "num_ues_per_cluster=%s num_clusters=%d hotspot_hold_steps=%d num_states=%d "
                   "(per-sector independent Markov chains)",
                   num_bs, num_clusters_per_sector, ues_desc, num_clusters,
                   hotspot_hold_steps, demand_states.shape[0])

    def _draw_next_states(self, current_idx: np.ndarray) -> np.ndarray:
        """Per-sector independent Markov draw: sector s's next state index
        is sampled from self.transition_matrix[current_idx[s]] (2D, shared)
        or self.transition_matrix[s, current_idx[s]] (3D, per-sector) --
        its OWN chain, not a network-wide shared clock."""
        current_idx_t = torch.as_tensor(current_idx, device=self.device)
        if self.transition_matrix.dim() == 3:
            sector_idx_t = torch.arange(len(current_idx), device=self.device)
            probs = self.transition_matrix[sector_idx_t, current_idx_t]  # [num_bs, num_states]
        else:
            probs = self.transition_matrix[current_idx_t]  # [num_bs, num_states]
        return torch.multinomial(probs, 1, generator=self.generator).squeeze(-1).cpu().numpy()

    def _advance_periodic_states(self, current_idx: np.ndarray) -> np.ndarray:
        """Deterministic per-sector cycle: +1 mod num_states. Every sector
        applies the SAME rule, but since each sector's OWN starting index
        was already an independent random draw (see __init__), sectors
        stay at a fixed relative offset from each other forever -- never
        resynchronizing -- rather than all advancing through the identical
        state at the same time (the earlier, now-fixed lockstep bug)."""
        return (current_idx + 1) % self.num_states

    def _sample_all_ref_points(self, cluster_sector: np.ndarray, cluster_type: np.ndarray) -> torch.Tensor:
        """Fresh point per cluster inside its own (sector, type) region --
        batched per (sector, type) GROUP (up to num_bs*2 groups), not once
        per cluster, matching the codebase's existing "loop over
        sectors/groups, vectorize within" idiom."""
        ref_xy = torch.zeros(self.num_clusters, 2, dtype=self.dtype, device=self.device)
        for s in range(self.topo.num_bs):
            for region_type in _HOTSPOT_REGION_TYPES:
                idx = np.where((cluster_sector == s) & (cluster_type == region_type))[0]
                if idx.size == 0:
                    continue
                ref_xy[idx] = self._sampler.hotspot_region_points(
                    s, region_type, idx.size, self.deviation_radius,
                    tangential_radius=self.deviation_radius_tangential)
        return ref_xy

    def _match_clusters_to_sectors(self) -> np.ndarray:
        """Network-wide randomized bipartite matching: each of self.num_clusters
        clusters (currently anchored per self.cluster_sector) is re-matched
        to one of self.num_clusters "slots" (num_clusters_per_sector slots
        per sector -- the SAME total capacity every sector always has,
        regardless of which demand state is active; only the TYPE
        breakdown within that fixed capacity varies, handled separately by
        _assign_types_within_sector), respecting the eligibility
        constraint that a cluster may
        only move to its CURRENT sector or one of that sector's
        neighbor_ids (cross-site allowed). Solved once for the whole
        network via scipy.optimize.linear_sum_assignment (random cost on
        eligible pairs, a large constant on ineligible ones) since
        eligibility overlaps between adjacent sectors -- an independent
        per-sector or per-site draw could double-book or orphan clusters.

        :output: [num_clusters] int, each cluster's NEW sector.
        """
        num_bs = self.topo.num_bs
        k = self.num_clusters_per_sector
        n = self.num_clusters
        neighbor_ids = self.topo.neighbor_ids  # [num_bs, max_neighbors], -1-padded

        eligible_sector = np.zeros((n, num_bs), dtype=bool)
        for c in range(n):
            s0 = self.cluster_sector[c]
            eligible_sector[c, s0] = True
            nbrs = neighbor_ids[s0]
            eligible_sector[c, nbrs[nbrs >= 0]] = True
        slot_sector = np.repeat(np.arange(num_bs), k)  # [n] -- slot j belongs to sector j//k
        eligible = eligible_sector[:, slot_sector]  # [n, n]

        # Random cost drawn from the SAME shared generator as everything
        # else in this class, so the whole simulation stays reproducible
        # from one environment_seed -- not a fresh, untracked numpy RNG.
        cost = torch.rand(n, n, generator=self.generator, dtype=torch.float64, device=self.device).cpu().numpy()
        big = 1e6
        cost = np.where(eligible, cost, big)
        row_ind, col_ind = linear_sum_assignment(cost)
        assignment = np.empty(n, dtype=int)
        assignment[row_ind] = col_ind

        bad = np.where(cost[np.arange(n), assignment] >= big / 2)[0]
        if bad.size > 0:
            raise RuntimeError(
                f"HotspotDemandMobility: no feasible cluster-to-sector reassignment found for "
                f"{bad.size} cluster(s) {bad.tolist()} (current sectors "
                f"{self.cluster_sector[bad].tolist()}) -- this should be structurally impossible "
                "(every sector always holds exactly num_clusters_per_sector clusters, so 'stay put' "
                "is always feasible), so this points at a real bug in neighbor_ids/eligibility "
                "construction, not a config tuning issue."
            )
        return slot_sector[assignment]

    def _assign_types_within_sector(self, new_sector: np.ndarray, counts_per_sector: np.ndarray) -> np.ndarray:
        """Random shuffle + slice by EACH sector's OWN (n_near,n_mid,n_far)
        counts, independently per sector -- no matching needed here, since
        hotspot type doesn't affect a cluster's sector-eligibility.

        :param counts_per_sector: [num_bs, len(_HOTSPOT_REGION_TYPES)] -- sector s's own counts may
            differ from every other sector's, since each sector's demand
            state is now drawn from its own independent Markov chain.
        """
        new_type = np.empty(self.num_clusters, dtype=object)
        for s in range(self.topo.num_bs):
            idx = np.where(new_sector == s)[0]
            type_pattern = np.repeat(np.array(_HOTSPOT_REGION_TYPES, dtype=object), counts_per_sector[s])
            perm = torch.randperm(idx.size, generator=self.generator, device=self.device).cpu().numpy()
            new_type[idx[perm]] = type_pattern
        return new_type

    def _member_radial_dir(self) -> torch.Tensor:
        """[num_ut, 2] unit vector per member, from its own cluster's home
        site to its own cluster's current reference point -- the axis
        valid_points/walk_points compress their ellipse's radial semi-axis
        along, so a member's own offset never crosses a near/mid/far
        boundary regardless of which direction around its sector its
        cluster happens to sit."""
        site_xy = self.topo.bs_loc[0, :, :2].to(dtype=self.dtype, device=self.device)  # [num_bs, 2]
        sector_per_member = torch.as_tensor(self.cluster_sector, device=self.device)[self.member_group_idx]
        site_per_member = site_xy[sector_per_member]
        radial_vec = self.ref_xy[self.member_group_idx] - site_per_member
        return radial_vec / radial_vec.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def step(self) -> torch.Tensor:
        logger.function("HotspotDemandMobility.step start")
        self._step_count += 1
        due = (self._step_count % self.hotspot_hold_steps == 0)
        # Public flag -- SimulationEngine.step_environment reads this right
        # after calling step() to know whether cluster_sector/cluster_type
        # (and therefore anchor status, for anchor_selection_mode="balanced"
        # or dynamic "random") need re-deriving this interval.
        self.transitioned_this_step = due

        if due:
            self._state_idx = (self._draw_next_states(self._state_idx) if self.state_transition_mode == "markov"
                              else self._advance_periodic_states(self._state_idx))
            counts_per_sector = self.demand_states[self._state_idx]  # [num_bs, 3]
            new_sector = self._match_clusters_to_sectors()
            new_type = self._assign_types_within_sector(new_sector, counts_per_sector)
            self.cluster_sector, self.cluster_type = new_sector, new_type
            self.ref_xy = self._sample_all_ref_points(new_sector, new_type)
            logger.debug("HotspotDemandMobility.step: transitioned to per-sector state_idx=%s",
                        self._state_idx.tolist())
            radial_dir = self._member_radial_dir()
            xy = self._sampler.valid_points(self.ref_xy[self.member_group_idx], self.deviation_radius,
                                            min_dist_from_site=float(self.topo.min_bs_ut_dist),
                                            radial_dir=radial_dir, tangential_radius=self.deviation_radius_tangential)
        else:
            radial_dir = self._member_radial_dir()
            xy = self._sampler.walk_points(
                self.ut_loc[:, :2], self.ref_xy[self.member_group_idx], self.deviation_radius,
                self.member_walk_step, min_dist_from_site=float(self.topo.min_bs_ut_dist),
                radial_dir=radial_dir, tangential_radius=self.deviation_radius_tangential)

        self.ut_loc = torch.cat([xy, self._z], dim=-1)
        logger.function("HotspotDemandMobility.step end")
        return self.ut_loc
