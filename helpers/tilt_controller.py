"""Per-sector electrical tilt controllers.

Every controller here derives from ``TiltController`` (right below): pick a
starting tilt (``initial_select``), then advance it interval over interval
(``update``). Each subclass's own initial_select()/update() parameter list
reflects what THAT controller actually needs -- a coverage search needs a
power table, Adaptive Legacy needs raw geometry+power, RL needs
observations+reward -- deliberately not forced to a common signature, since
those inputs differ in KIND, not just naming; what's shared is the lifecycle
contract (these two method names) and, for each, a single persistent
per-sector tilt state advanced by ``update``.

Three families implement it:

* ``AdaptiveLegacyTiltController``: Reactive rule-based per-sector RET
  controller, scored at the current tilt and updated once per interval
  using measurements pooled over that interval.

* ``SearchTiltController`` (``StaticTiltController``/``DynamicTiltController``):
  given a power table, pick a per-sector tilt assignment maximizing pooled
  coverage -- once and frozen (Static) or every interval, warm-started from
  the previous pick (Dynamic).

* ``RLTiltController``: Independent per-sector RL based tilt controller,
  scored at the current tilt and updated from pooled measurements to
  select tilts for the next interval.

``GlobalTiltSelector``/``LocalTiltSelector`` are NOT controllers -- they're
the stateless search algorithms a ``SearchTiltController`` wraps and owns
(the Strategy to its Context); they never hold tilt state themselves.
"""

import numpy as np


class TiltController:
    """Common lifecycle every per-sector tilt controller in this project
    follows: ``initial_select()`` picks the starting tilt, ``update()``
    advances it thereafter.
    """

    def initial_select(self, *args, **kwargs):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError


class AdaptiveLegacyTiltController(TiltController):
    """
    Reactive per-sector electrical-tilt controller inspired by classical
    rule-based RET schemes, particularly the fuzzy-logic framework of
    Buenestado et al. (IEEE TVT, 2017). 

    Two indicators are computed independently for each sector at every control
    interval:

    * Overshoot (``n_os``): fraction of all UEs that are served by another
    sector, lie beyond ``distance_threshold_m`` from the considered sector,
    and still receive power from it within ``overlap_margin_db`` of their
    serving-sector power.

    * Bad coverage (``r_bc``): among the considered sector's own edge UEs,
    defined by ``edge_percentile``, the fraction whose SINR is below
    ``coverage_threshold_db``.

    The controller applies the following rule per sector:

    * downtilt if overshoot is high and edge coverage is acceptable;
    * uptilt if edge coverage is poor and overshoot is acceptable;
    * otherwise hold the current tilt.

    No fuzzy membership functions or defuzzification are used in Threshold 
    comparisons. No overlap indicators.

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
    ):
        """
        :param distance_threshold_m: Distance threshold [m] used to distinguish
            near and far UEs for overshoot detection.

        :param coverage_threshold_db: SINR threshold [dB] below which a UE is
            considered poorly covered.

        :param overlap_margin_db: Maximum received-power difference [dB] between
            the considered sector and the serving sector for a UE to count toward
            overshoot.

        :param edge_percentile: Distance percentile of a sector's served UEs used
            to define the cell-edge UE subset.

        :param os_threshold: Overshoot-ratio threshold above which overshoot is
            considered excessive.

        :param bc_threshold: Bad-coverage-ratio threshold above which cell-edge
            coverage is considered insufficient.
        """
        self.theta_min_deg = theta_min_deg
        self.theta_max_deg = theta_max_deg
        self.tilt_deg = np.full(num_sectors, float(initial_tilt_deg))
        self.distance_threshold_m = distance_threshold_m
        self.coverage_threshold_db = coverage_threshold_db
        self.overlap_margin_db = overlap_margin_db
        self.edge_percentile = edge_percentile
        self.tilt_step_deg = tilt_step_deg
        self.os_threshold = os_threshold
        self.bc_threshold = bc_threshold

    def initial_select(self) -> np.ndarray:
        """The starting tilt"""
        return self.tilt_deg

    def update(self, overshoot_per_sector: np.ndarray, per_sector_coverage: np.ndarray) -> np.ndarray:
        """Advance ``self.tilt_deg`` by one interval and return it -- pure
        decision logic; the measurement (n_os/r_bc's ingredients) is
        KpiManager.compute_ue_kpis's job (overshoot + edge-restricted
        per_sector_coverage), called by the caller before this.

        :param overshoot_per_sector: [num_sectors] n_os, from
            compute_ue_kpis(..., distance_threshold_m=..., overlap_margin_db=...)
        :param per_sector_coverage: [num_sectors] edge-restricted GOOD-coverage
            fraction, from compute_ue_kpis(..., edge_percentile=...) -- r_bc
            (bad-coverage fraction) is 1 minus this.
        """
        n_os = overshoot_per_sector
        r_bc = 1.0 - per_sector_coverage

        # Status check
        interference_problem = n_os > self.os_threshold
        coverage_problem = r_bc > self.bc_threshold

        # Increasing downtilt solves overshoot
        increase_downtilt = interference_problem & ~coverage_problem

        # Decreasing downtilt solves coverage
        decrease_downtilt = coverage_problem & ~interference_problem

        # Final tilt
        delta_tilt = increase_downtilt * self.tilt_step_deg + decrease_downtilt * (-self.tilt_step_deg)
        self.tilt_deg = np.clip(self.tilt_deg + delta_tilt, self.theta_min_deg, self.theta_max_deg)

        return self.tilt_deg


class GlobalTiltSelector:
    """Picks the single common tilt (same index applied to every sector)
    that maximizes pooled coverage -- exhaustive search over every tilt in
    the power table.
    """

    def select(self, kpi_manager, power_table: np.ndarray, threshold_db: float, warm_start=None,
              weights: np.ndarray = None) -> np.ndarray:
        num_tilts, num_sectors = power_table.shape[0], power_table.shape[2]
        coverages = np.array([
            kpi_manager.compute_ue_kpis(power_table, np.full(num_sectors, t, dtype=int), threshold_db,
                                     weights=weights)["coverage"]
            for t in range(num_tilts)
        ])

        # Best global tilt
        best_t = int(np.argmax(coverages))
        return np.full(num_sectors, best_t, dtype=int)


class LocalTiltSelector:
    """Selects per-sector tilts via coordinate ascent. Each sector sequentially 
    chooses the candidate tilt that maximizes total network coverage given the 
    current tilts of all other sectors. The process stops when a full round makes 
    no changes, yielding a local optimum of the coupled non-convex problem.

    :ivar last_num_rounds: rounds run by the most recent ``select()`` call.
    :ivar last_coverage_trace: that call's pooled coverage per round
        (element 0 is the starting coverage) -- e.g. for a convergence plot.
    """

    def __init__(self, max_rounds: int = 20):
        self.max_rounds = max_rounds
        self.last_num_rounds = None
        self.last_coverage_trace = None

    def select(self, kpi_manager, power_table: np.ndarray, threshold_db: float, warm_start=None,
              weights: np.ndarray = None) -> np.ndarray:
        num_tilts, num_sectors = power_table.shape[0], power_table.shape[2]
        if warm_start is None:
            warm_start = GlobalTiltSelector().select(kpi_manager, power_table, threshold_db, weights=weights)
        assignment = np.array(warm_start, dtype=int).copy()

        coverage_trace = [kpi_manager.compute_ue_kpis(power_table, assignment, threshold_db, weights=weights)["coverage"]]

        # Coverage ascent algorithm
        num_rounds_run = 0
        for round_idx in range(self.max_rounds):
            num_rounds_run = round_idx + 1
            changed = False
            for s in range(num_sectors):
                trial_coverages = np.empty(num_tilts)
                for t in range(num_tilts):
                    trial = assignment.copy()
                    trial[s] = t
                    trial_coverages[t] = kpi_manager.compute_ue_kpis(power_table, trial, threshold_db,
                                                                     weights=weights)["coverage"]
                best_t = int(np.argmax(trial_coverages))
                if best_t != assignment[s]:
                    assignment[s] = best_t
                    changed = True
            coverage_trace.append(
                kpi_manager.compute_ue_kpis(power_table, assignment, threshold_db, weights=weights)["coverage"])
            if not changed:
                break

        self.last_num_rounds = num_rounds_run
        self.last_coverage_trace = np.array(coverage_trace)
        return assignment


class SearchTiltController(TiltController):
    """Base for controllers that wrap a coverage-search selector (the
    Strategy) and own the persistent per-sector assignment across
    intervals (the Context) -- GlobalTiltSelector/LocalTiltSelector never
    hold tilt state themselves, only this class and its subclasses do.
    initial_select() picks the starting assignment; update() advances it
    from there (and falls back to initial_select() itself if called
    first, so callers can just call update() every interval, first one
    included).
    """

    def __init__(self, selector):
        self.selector = selector
        self.assignment = None

    def initial_select(self, kpi_manager, power_table: np.ndarray, threshold_db: float,
                       weights: np.ndarray = None) -> np.ndarray:
        self.assignment = self.selector.select(kpi_manager, power_table, threshold_db, weights=weights)
        return self.assignment

    def update(self, kpi_manager, power_table: np.ndarray, threshold_db: float, weights: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError


class StaticTiltController(SearchTiltController):
    """Calibrates ONCE (initial_select), freezes that result for every
    later update() -- e.g. Static Global / Static Local, a one-time
    calibration (like a drive test).
    """

    def update(self, kpi_manager, power_table: np.ndarray, threshold_db: float, weights: np.ndarray = None) -> np.ndarray:
        if self.assignment is None:
            return self.initial_select(kpi_manager, power_table, threshold_db, weights=weights)
        return self.assignment


class DynamicTiltController(SearchTiltController):
    """Recomputes on every update() call, warm-started from its own
    previous assignment -- e.g. Dynamic Global / Dynamic Local.
    """

    def update(self, kpi_manager, power_table: np.ndarray, threshold_db: float, weights: np.ndarray = None) -> np.ndarray:
        if self.assignment is None:
            return self.initial_select(kpi_manager, power_table, threshold_db, weights=weights)
        self.assignment = self.selector.select(kpi_manager, power_table, threshold_db, warm_start=self.assignment,
                                               weights=weights)
        return self.assignment


class RLTiltController(TiltController):
    """Connects the simulation to a generic DRL tilt policy.
    At each interval, the policy observes the current network state,
    receives the reward from the previously applied action, and
    selects the action for the next interval using one-interval-lag causal control.
    Each RL transition is stored as (state,action,reward,next state).

    :ivar tilt_idx: [num_sectors] current tilt INDEX -- the action currently in effect.
    """

    def __init__(self, policy, num_sectors: int, initial_tilt_idx: int = 0):
        self.policy = policy
        self.tilt_idx = np.full(num_sectors, initial_tilt_idx, dtype=np.int64)
        self._prev_observations = None
        self._prev_actions = None

    def initial_select(self) -> np.ndarray:
        """The starting tilt INDEX"""
        return self.tilt_idx

    def update(self, observations: np.ndarray, rewards: np.ndarray, training: bool,
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
