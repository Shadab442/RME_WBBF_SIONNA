"""Per-sector electrical tilt controllers.

Three independent families live here:

* ``AdaptiveLegacyTiltController`` (below): a real deployable per-interval
  controller -- owns one tilt value per sector, updates from that
  interval's own measurement only (never a power-table search).

* Coverage-search selectors/wrappers (middle of this file): given a
  KpiCalculator and a power table (helpers.kpi_calculator.KpiCalculator.
  compute_power_table), pick a per-sector tilt assignment maximizing pooled
  coverage.

* ``RLTiltController`` (bottom of this file): a thin bridge between the
  simulation's per-interval pooled state/reward and a pluggable RL policy
  (drl.factory.create_policy) -- contains no RL/torch-specific logic
  itself, mirrors DynamicTiltController's role of wrapping a pluggable
  per-interval decision algorithm, just for a learned one instead of a
  search.
"""

import numpy as np
import torch


class AdaptiveLegacyTiltController:
    """Reactive per-sector tilt controller: every measurement interval,
    each sector's tilt is nudged down if it's overreaching into distant
    cells, nudged up if its own cell edge is underserved, held otherwise.
    Framing (an "overshoot" indicator counterbalanced against a "bad
    coverage" indicator, decided independently per sector every interval)
    follows the classical rule-based RET literature's standard shape, e.g.
    Buenestado et al.'s two-stage fuzzy logic controller (IEEE TVT 66(5),
    2017 / PhD thesis, Universidad de Malaga, 2017, ch. 4) -- see that work
    for the full three-indicator, fuzzy-logic version this is inspired by.
    This implementation is a simplified, crisp-threshold adaptation, not a
    reproduction of it:

    * Only two indicators are used, not three. The paper's third indicator
      -- "useless overlap" (a sector radiating comparable-to-serving power
      into NEARBY UEs already well served by another sector) -- is dropped
      here: under this simulation's max-SINR hard attachment and clean
      (noise-free) large-scale channel, a UE well inside sector i's own
      footprint is served by i almost by definition (steep pathloss means
      no real contest), so the only place another sector can still be
      comparably strong is right at the cell boundary -- which is the same
      physical location "overshoot" (from the far side) and "bad coverage"
      (a sector's own edge) already cover. Keeping it would add a
      threshold that structurally never fires in this setup.
    * Decisions are crisp thresholds, not fuzzy sets -- no membership
      functions or weighted-average defuzzification. With only two
      indicators left, the paper's stage-1 OR-combination collapses to a
      single condition anyway, so the fuzzy machinery has nothing left to
      combine.

    Two per-sector indicators, recomputed every interval from the full
    per-sector power matrix (ground truth, not RSRP samples):

    * Overshoot (``n_os``): fraction of the WHOLE UE population that (a) is
      served by another sector, (b) sits farther than
      ``distance_threshold_m`` from this sector's own site, and (c) still
      receives power from this sector within ``overlap_margin_db`` of what
      its true server delivers -- i.e. this sector is radiating a
      genuinely competitive signal well outside its own footprint. No
      explicit sector-adjacency notion is needed: a UE served by a sector
      many cells away will essentially never pass the "comparable power"
      test given realistic pathloss falloff, so the candidate population
      self-restricts to boundary-adjacent UEs.
    * Bad coverage (``r_bc``): of this sector's OWN served UEs, the
      farthest fraction by distance (``edge_percentile``, i.e. its cell
      edge, not a fixed radius), the share with SINR below
      ``coverage_threshold_db`` -- reuses the same threshold the
      search-based methods optimize for, rather than a separate untuned
      RSRP scale.

    Per sector, per interval: downtilt by ``tilt_step_deg`` if
    ``n_os > os_threshold`` and coverage is fine; uptilt if
    ``r_bc > bc_threshold`` and overshoot is fine; hold if both or neither
    fire (conflicting or absent signals).

    :ivar tilt_deg: [num_sectors] current tilt per sector.
    """

    def __init__(
        self,
        num_sectors: int,
        initial_tilt_deg: float,
        theta_min_deg: float,
        theta_max_deg: float,
        distance_threshold_m: float,
        coverage_threshold_db: float,
        overlap_margin_db: float,
        edge_percentile: float,
        tilt_step_deg: float,
        os_threshold: float = 0.006,
        bc_threshold: float = 0.33,
        dtype=torch.float32,
        device="cpu",
    ):
        """
        :param distance_threshold_m: near/far distance threshold for the
            overshoot indicator [m] -- e.g. the topology's own hex cell
            radius (``CellularTopology.grid.cell_radius``), a sector's
            natural coverage footprint.
        :param coverage_threshold_db: SINR threshold for "bad coverage" --
            reuses the same threshold the search-based methods optimize
            for, rather than a separate untuned RSRP scale.
        :param overlap_margin_db: how close (in dB) a non-serving sector's
            power must be to the true serving sector's to count as a real,
            comparable interference contribution (overshoot).
        :param edge_percentile: percentile (of a sector's own served UEs'
            distance to site) defining that sector's cell edge.
        :param os_threshold, bc_threshold: crisp fraction-of-population
            cutoffs for the overshoot / bad-coverage indicators -- not
            values from the paper it's inspired by (whose own boundaries
            were tuned to a real network's RSRP scale, not this simulation's
            fraction-of-population units). Defaults are empirically
            calibrated to THIS topology, not a general prior: measured
            directly (a standalone diagnostic sweeping every candidate tilt)
            before picking them -- n_os stayed within roughly 0.002-0.01
            across the whole tilt range regardless of tilt (an initial guess
            of 0.05 was 5-30x too high to ever fire), while r_bc rose
            from roughly 0.28 at the shallowest tilt to roughly 0.44 at the
            deepest -- so 0.006 and 0.33 were chosen as the values that
            actually sit inside each indicator's observed operating range
            (letting individual sectors cross in either direction depending
            on their own local geometry) rather than outside it (which
            silently makes one branch permanently unreachable and the
            controller a one-way ratchet -- confirmed as the failure mode
            of the earlier 0.05/0.20 guess).
        """
        self.theta_min_deg = theta_min_deg
        self.theta_max_deg = theta_max_deg
        self.tilt_deg = torch.full((num_sectors,), float(initial_tilt_deg), dtype=dtype, device=device)
        self.distance_threshold_m = distance_threshold_m
        self.coverage_threshold_db = coverage_threshold_db
        self.overlap_margin_db = overlap_margin_db
        self.edge_percentile = edge_percentile
        self.tilt_step_deg = tilt_step_deg
        self.os_threshold = os_threshold
        self.bc_threshold = bc_threshold

    def update(self, ut_loc: torch.Tensor, power_w: torch.Tensor, noise_power_w: float,
              bs_xy: torch.Tensor) -> torch.Tensor:
        """Advance ``self.tilt_deg`` by one interval and return it.

        :param ut_loc: [num_ut, 3] current UE positions (one realization).
        :param power_w: [num_sectors, num_ut] received power [W] from EVERY
            sector to every UE (not just the serving one) -- the full
            per-sector power matrix, e.g. from
            helpers.kpi_calculator.KpiCalculator.compute_power_matrix_w.
        :param noise_power_w: noise power [W].
        :param bs_xy: [num_sectors, 2] per-sector (x, y) position.
        """
        dtype, device = self.tilt_deg.dtype, self.tilt_deg.device
        ut_loc = torch.as_tensor(ut_loc, dtype=dtype, device=device)
        power_w = torch.as_tensor(power_w, dtype=dtype, device=device)
        bs_xy = torch.as_tensor(bs_xy, dtype=dtype, device=device)
        num_sectors, num_ut = power_w.shape

        total_w = power_w.sum(dim=0, keepdim=True)
        interference_w = total_w - power_w
        sinr = power_w / (interference_w + noise_power_w)
        sinr_db = 10.0 * torch.log10(sinr)  # [num_sectors, num_ut] -- every candidate server's SINR
        best_sinr_db, serving_idx = sinr_db.max(dim=0)  # [num_ut]
        power_db = 10.0 * torch.log10(power_w)  # [num_sectors, num_ut]
        serving_power_db = torch.gather(power_db, 0, serving_idx.unsqueeze(0)).squeeze(0)  # [num_ut]

        dist_to_site = torch.linalg.norm(
            ut_loc[None, :, :2] - bs_xy[:, None, :], dim=-1
        )  # [num_sectors, num_ut]

        n_os = torch.zeros(num_sectors, dtype=dtype, device=device)
        r_bc = torch.zeros(num_sectors, dtype=dtype, device=device)

        for i in range(num_sectors):
            served_elsewhere = serving_idx != i
            # A real, comparable interference contribution: sector i's power
            # is within overlap_margin_db of the UE's TRUE serving power.
            power_gap_db = serving_power_db - power_db[i]
            comparable = served_elsewhere & (power_gap_db <= self.overlap_margin_db)
            is_far = dist_to_site[i] > self.distance_threshold_m
            n_os[i] = (comparable & is_far).to(dtype).mean()

            served_by_i = serving_idx == i
            if served_by_i.any():
                dist_i = dist_to_site[i][served_by_i]
                edge_threshold = torch.quantile(dist_i, self.edge_percentile / 100.0)
                edge_mask = dist_i >= edge_threshold
                if edge_mask.any():
                    edge_sinr_db = best_sinr_db[served_by_i][edge_mask]
                    r_bc[i] = (edge_sinr_db < self.coverage_threshold_db).to(dtype).mean()

        interference_problem = n_os > self.os_threshold
        coverage_problem = r_bc > self.bc_threshold
        # downtilt_deg convention (ElectricalDowntilt): 0 = boresight, + = down
        # -- MORE downtilt pulls reach in (fixes overshoot), LESS downtilt
        # extends reach (fixes bad own-cell-edge coverage).
        increase_downtilt = interference_problem & ~coverage_problem
        decrease_downtilt = coverage_problem & ~interference_problem
        delta_tilt = increase_downtilt.to(dtype) * self.tilt_step_deg + decrease_downtilt.to(dtype) * (-self.tilt_step_deg)

        self.tilt_deg = torch.clamp(self.tilt_deg + delta_tilt, self.theta_min_deg, self.theta_max_deg)
        return self.tilt_deg


class GlobalTiltSelector:
    """Picks the single common tilt (same index applied to every sector)
    that maximizes pooled coverage -- exhaustive search over every tilt in
    the power table. Cheap enough (a handful of candidate tilts) that
    ``warm_start`` is accepted only so callers can treat every selector
    identically; it's always ignored here.
    """

    def select(self, kpi_calc, power_table: np.ndarray, threshold_db: float, warm_start=None,
              weights: np.ndarray = None) -> np.ndarray:
        num_tilts, num_sectors, _ = power_table.shape
        coverages = np.array([
            kpi_calc.coverage_from_assignment(power_table, np.full(num_sectors, t, dtype=int), threshold_db,
                                              weights=weights)
            for t in range(num_tilts)
        ])
        best_t = int(np.argmax(coverages))
        return np.full(num_sectors, best_t, dtype=int)


class LocalTiltSelector:
    """Picks a per-sector tilt assignment via coordinate ascent: each
    sector, in turn, picks whichever candidate tilt maximizes TOTAL pooled
    coverage given every other sector's CURRENT tilt -- not the sector's
    own self-interest, since a sector's tilt change also affects every
    other sector's interference and so its coverage too. "Keep the current
    tilt" is always among the candidates, so pooled coverage is
    non-decreasing round over round -- the result is guaranteed no worse
    than ``warm_start`` (defaults to GlobalTiltSelector's pick if not
    given). Converges when a full round changes no sector's tilt; this is
    still a coupled, non-convex problem, so the result can be a local
    optimum, just never a worse one than the start.

    :ivar last_num_rounds: rounds run by the most recent ``select()`` call.
    :ivar last_coverage_trace: that call's pooled coverage per round
        (element 0 is the starting coverage) -- e.g. for a convergence plot.
    """

    def __init__(self, max_rounds: int = 20):
        self.max_rounds = max_rounds
        self.last_num_rounds = None
        self.last_coverage_trace = None

    def select(self, kpi_calc, power_table: np.ndarray, threshold_db: float, warm_start=None,
              weights: np.ndarray = None) -> np.ndarray:
        num_tilts, num_sectors, _ = power_table.shape
        if warm_start is None:
            warm_start = GlobalTiltSelector().select(kpi_calc, power_table, threshold_db, weights=weights)
        assignment = np.array(warm_start, dtype=int).copy()
        coverage_trace = [kpi_calc.coverage_from_assignment(power_table, assignment, threshold_db, weights=weights)]

        num_rounds_run = 0
        for round_idx in range(self.max_rounds):
            num_rounds_run = round_idx + 1
            changed = False
            for s in range(num_sectors):
                trial_coverages = np.empty(num_tilts)
                for t in range(num_tilts):
                    trial = assignment.copy()
                    trial[s] = t
                    trial_coverages[t] = kpi_calc.coverage_from_assignment(power_table, trial, threshold_db,
                                                                           weights=weights)
                best_t = int(np.argmax(trial_coverages))
                if best_t != assignment[s]:
                    assignment[s] = best_t
                    changed = True
            coverage_trace.append(kpi_calc.coverage_from_assignment(power_table, assignment, threshold_db,
                                                                    weights=weights))
            if not changed:
                break

        self.last_num_rounds = num_rounds_run
        self.last_coverage_trace = np.array(coverage_trace)
        return assignment


class StaticTiltController:
    """Wraps a selector; calibrates ONCE, on its first ``select()`` call,
    and freezes that result for every later call -- e.g. Static Global /
    Static Local, a one-time calibration (like a drive test) rather than an
    ongoing controller.
    """

    def __init__(self, selector):
        self.selector = selector
        self.assignment = None

    def select(self, kpi_calc, power_table: np.ndarray, threshold_db: float, weights: np.ndarray = None) -> np.ndarray:
        if self.assignment is None:
            self.assignment = self.selector.select(kpi_calc, power_table, threshold_db, weights=weights)
        return self.assignment


class DynamicTiltController:
    """Wraps a selector; recomputes on every ``select()`` call, warm-started
    from its own previous result (cheaper than restarting from scratch each
    time, and arguably more realistic -- a real controller wouldn't reset
    to a naive baseline every interval either) -- e.g. Dynamic Global /
    Dynamic Local.
    """

    def __init__(self, selector):
        self.selector = selector
        self.assignment = None

    def select(self, kpi_calc, power_table: np.ndarray, threshold_db: float, weights: np.ndarray = None) -> np.ndarray:
        self.assignment = self.selector.select(kpi_calc, power_table, threshold_db, warm_start=self.assignment,
                                               weights=weights)
        return self.assignment


class RLTiltController:
    """Bridges the simulation's per-interval pooled (state, reward) to any
    ``drl.base.TiltPolicy`` -- e.g. drl.independent_dqn.IndependentDqn,
    built via drl.factory.create_policy. No RL/torch-specific code lives
    here; this class only handles the ONE-INTERVAL-LAG bookkeeping (the same
    causal pattern AdaptiveLegacyTiltController and Dynamic Local Causal
    already use): the action returned by ``step()`` was decided from the
    PREVIOUS call's observations, and is applied for the interval that
    produces THIS call's observations/reward.

    Every stored transition (observations, action, reward, next_observations)
    is genuinely complete -- there is no dangling transition needing a
    ``terminal`` flag, since the caller simply stops calling ``step()`` after
    the last interval rather than needing to bootstrap past it. ``terminal``
    is exposed anyway (always False by ``step()``'s default) only so a
    caller that DOES want the last transition to skip bootstrapping can ask
    for it explicitly.

    :ivar tilt_idx: [num_sectors] current tilt INDEX (into whatever
        candidate list the caller applies it against, e.g.
        DOWNTILT_SWEEP_DEG) -- the action currently in effect.
    """

    def __init__(self, policy, num_sectors: int, initial_tilt_idx: int = 0):
        self.policy = policy
        self.tilt_idx = np.full(num_sectors, initial_tilt_idx, dtype=np.int64)
        self._prev_observations = None
        self._prev_actions = None

    def step(self, observations: np.ndarray, rewards: np.ndarray, training: bool,
            terminal: bool = False) -> np.ndarray:
        """
        :param observations: [num_sectors, num_features], resulting from
            ``self.tilt_idx`` (the action decided on the PREVIOUS call).
        :param rewards: [num_sectors], likewise resulting from that action.
        :param training: if True, learn from the completed transition (if
            any) and explore; if False, act greedily and don't learn.
        :output: the NEW ``self.tilt_idx``, decided from ``observations``,
            to apply for the next interval.
        """
        if training and self._prev_observations is not None:
            self.policy.observe(self._prev_observations, self._prev_actions, rewards,
                               observations, terminal)
        actions = self.policy.act(observations, training)
        self._prev_observations = observations
        self._prev_actions = actions
        self.tilt_idx = actions
        return self.tilt_idx
