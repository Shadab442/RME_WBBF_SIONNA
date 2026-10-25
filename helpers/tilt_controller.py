"""Per-sector electrical tilt controllers.

Every controller here derives from ``TiltController`` (right below): pick a
starting tilt (``initial_select``), then advance it interval over interval
(``update``). Each subclass's own initial_select()/update() parameter list
reflects what THAT controller actually needs -- Adaptive Legacy needs raw
geometry+power, RL needs observations+reward -- deliberately not forced to
a common signature, since those inputs differ in KIND, not just naming;
what's shared is the lifecycle contract (these two method names) and, for
each, a single persistent per-sector tilt state advanced by ``update``.

Exactly two algorithms exist in this codebase (2026-09-22 -- Dynamic Local
Causal, No Tilt, and both the old coordinate-ascent Oracle AND its CP-SAT
replacement are all deleted; CP-SAT wasn't earning its own real-world cost
relative to the baseline pipeline, so it was dropped rather than kept as a
fourth comparison method. Nothing else is kept):

* ``AdaptiveLegacyTiltController``: Reactive rule-based per-sector RET
  controller, scored at the current tilt and updated once per interval
  using measurements pooled over that interval.

* ``RLTiltController``: Independent per-sector RL based tilt controller
  (DQN, or RandomPolicy for the untrained baseline -- see drl/factory.py),
  scored at the current tilt and updated from pooled measurements to
  select tilts for the next interval.
"""

import numpy as np

from helpers.utils import get_logger

logger = get_logger(__name__)


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
    * otherwise hold the current tilt -- UNLESS both indicators are flagged
      simultaneously, in which case the tilt moves toward whichever
      indicator is proportionally further past its own threshold (severity
      = value / threshold), rather than holding. A pure hold on "both
      flagged" was found (2026-09-22) to deadlock the controller
      permanently at its initial tilt whenever the two thresholds are not
      both loose enough for some tilt to clear coverage_problem before
      interference_problem does on the way up from theta_min -- exactly the
      failure mode observed under the current network config, where
      coverage_problem was structurally true across the entire tilt sweep.

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

    def update(self, overshoot_per_sector: np.ndarray, per_sector_coverage: np.ndarray,
              has_data: np.ndarray = None) -> np.ndarray:
        """Advance ``self.tilt_deg`` by one interval and return it -- pure
        decision logic; the measurement (n_os/r_bc's ingredients) is
        KpiManager.compute_ue_kpis's job (overshoot + edge-restricted
        per_sector_coverage), called by the caller before this.

        :param overshoot_per_sector: [num_sectors] n_os, from
            compute_ue_kpis(..., distance_threshold_m=..., overlap_margin_db=...)
        :param per_sector_coverage: [num_sectors] edge-restricted GOOD-coverage
            fraction, from compute_ue_kpis(..., edge_percentile=...) -- r_bc
            (bad-coverage fraction) is 1 minus this.
        :param has_data: [num_sectors] bool, optional -- False marks a sector
            with no real observation this interval (e.g. zero served UEs). 
            That sector's tilt is reset to 0 deg
        """
        logger.function("AdaptiveLegacyTiltController.update start")
        n_os = np.nan_to_num(overshoot_per_sector, nan=0.0)
        r_bc = 1.0 - np.nan_to_num(per_sector_coverage, nan=0.0)

        # Status check
        interference_problem = n_os > self.os_threshold
        coverage_problem = r_bc > self.bc_threshold
        logger.debug("AdaptiveLegacyTiltController.update: interference_problem=%d coverage_problem=%d sectors",
                    int(interference_problem.sum()), int(coverage_problem.sum()))

        # Both flagged at once: act on whichever is proportionally further
        # past its own threshold instead of holding (a hold here can
        # deadlock the tilt permanently -- see class docstring).
        both_problems = interference_problem & coverage_problem
        os_severity = n_os / self.os_threshold
        bc_severity = r_bc / self.bc_threshold
        tiebreak_downtilt = both_problems & (os_severity >= bc_severity)
        tiebreak_uptilt = both_problems & (os_severity < bc_severity)

        # Increasing downtilt solves overshoot
        increase_downtilt = (interference_problem & ~coverage_problem) | tiebreak_downtilt

        # Decreasing downtilt solves coverage
        decrease_downtilt = (coverage_problem & ~interference_problem) | tiebreak_uptilt

        # Final tilt
        delta_tilt = increase_downtilt * self.tilt_step_deg + decrease_downtilt * (-self.tilt_step_deg)
        self.tilt_deg = np.clip(self.tilt_deg + delta_tilt, self.theta_min_deg, self.theta_max_deg)
        logger.debug("AdaptiveLegacyTiltController.update: %d sectors changed tilt",
                    int((delta_tilt != 0).sum()))

        if has_data is not None:
            self.tilt_deg = np.where(has_data, self.tilt_deg, 0.0)
            if not has_data.all():
                logger.warning("AdaptiveLegacyTiltController.update: %d sector(s) with no data, tilt reset to 0",
                              int((~has_data).sum()))

        logger.function("AdaptiveLegacyTiltController.update end")
        return self.tilt_deg


class RLTiltController(TiltController):
    """Connects the simulation to a generic DRL tilt policy.

    :ivar tilt_idx: [num_sectors] current tilt INDEX -- the action currently in effect.
    """

    def __init__(self, policy, num_sectors: int, initial_tilt_idx: int = 0, variable_size: bool = False):
        """:param variable_size: if True, observations are a per-sector
            LIST of possibly different-length 1D arrays (e.g. sectors with
            fewer than 4 neighbors getting a smaller state) rather than one
            dense [num_sectors, num_features] array -- policy.act()/observe()
            already index one sector at a time so they need no change, but
            this class's own _prev_observations bookkeeping (normally a
            single dense array with boolean-mask batch updates, which
            requires uniform per-sector width) switches to an explicit
            per-sector list with per-sector loops instead.
        """
        self.policy = policy
        self.variable_size = variable_size
        self.tilt_idx = np.full(num_sectors, initial_tilt_idx, dtype=np.int64)
        self._prev_observations = None
        self._prev_actions = np.zeros(num_sectors, dtype=np.int64)
        self._prev_valid = np.zeros(num_sectors, dtype=bool)

    def initial_select(self) -> np.ndarray:
        """The starting tilt INDEX"""
        return self.tilt_idx

    def learn(self, observations: np.ndarray, rewards: np.ndarray, training: bool,
             has_data: np.ndarray = None, schedule: np.ndarray = None, terminal: bool = False) -> None:
        """Stores the transition (self._prev_observations/_prev_actions --
        this sector's own previously stored action -- -> rewards/
        observations) for scheduled+valid sectors. Does NOT pick a new
        action -- call decide() separately for that (update() below fuses
        both, in that order, for the ordinary "next_state" convention).

        :param observations: [num_sectors, num_features] dense array, OR --
            if variable_size -- a [num_sectors] list of per-sector 1D
            arrays (possibly different widths). This is the RESULT
            observation the stored transition's reward/outcome corresponds
            to -- under "current_state" reward timing this is NOT
            necessarily the same array passed to the decide() call that
            picked the action being credited.
        :param rewards: [num_sectors], only meaningful for scheduled
            sectors -- reward from each scheduled sector's own previously
            stored action.
        :param training: if True, learn from each scheduled sector's
            completed transition; if False, don't learn (still called
            every interval regardless -- see RLTiltController.update's own
            docstring for why "record but don't train" still matters for a
            policy with recurrent/windowed state).
        :param has_data: [num_sectors] bool, optional -- False marks a
            sector with no real observation this call. A transition is
            added to experience only if both the state it started from
            (this sector's own last-stored validity) and the outcome it
            produced (this call's has_data) had real data.
        :param schedule: [num_sectors] bool, optional -- False means this
            sector's window hasn't closed yet this call; nothing is
            recorded for it.
        """
        logger.function("RLTiltController.learn start: training=%s terminal=%s", training, terminal)
        num_sectors = len(observations)
        if has_data is None:
            has_data = np.ones(num_sectors, dtype=bool)
        if schedule is None:
            schedule = np.ones(num_sectors, dtype=bool)

        if self._prev_observations is not None and schedule.any():
            transition_valid = schedule & self._prev_valid & has_data
            if transition_valid.any():
                self.policy.observe(self._prev_observations, self._prev_actions, rewards,
                                   observations, terminal, mask=transition_valid, training=training)
                logger.debug("RLTiltController.learn: observed %d valid transitions (training=%s)",
                            int(transition_valid.sum()), training)
        logger.function("RLTiltController.learn end")

    def decide(self, observations: np.ndarray, training: bool,
              has_data: np.ndarray = None, schedule: np.ndarray = None) -> np.ndarray:
        """Picks a new tilt from `observations` for scheduled+has_data
        sectors (every other sector holds its previous tilt), and caches
        (observations, action, has_data) as this sector's own
        "previously stored action" for the NEXT learn() call to credit.
        Does NOT store a training transition itself -- call learn()
        separately for that.

        :param observations: [num_sectors, num_features] dense array, OR --
            if variable_size -- a [num_sectors] list of per-sector 1D
            arrays. This is the DECISION input -- under "current_state"
            reward timing, a forecast of the CURRENT interval (e.g.
            predicted occupancy), not necessarily the same array learn()
            is later called with.
        :param has_data: [num_sectors] bool, optional -- False marks a
            sector with no real observation this call; it holds its
            previous tilt, same fallback as "not scheduled".
        :param schedule: [num_sectors] bool, optional -- False means this
            sector's window hasn't closed yet this call; it holds
            ``self.tilt_idx`` unchanged and its stored previous-state is
            left untouched.
        """
        logger.function("RLTiltController.decide start: training=%s", training)
        num_sectors = len(observations)
        if has_data is None:
            has_data = np.ones(num_sectors, dtype=bool)
        if schedule is None:
            schedule = np.ones(num_sectors, dtype=bool)
        logger.debug("RLTiltController.decide: scheduled=%d/%d has_data=%d/%d",
                    int(schedule.sum()), num_sectors, int(has_data.sum()), num_sectors)

        # Only a scheduled sector with real data gets to pick a new tilt;
        # every other masked sector (not yet scheduled, OR scheduled but
        # dataless) holds its previous tilt via default_action=self.tilt_idx.
        act_mask = schedule & has_data
        actions = self.policy.act(observations, training, mask=act_mask, default_action=self.tilt_idx)

        if self._prev_observations is None:
            self._prev_observations = [None] * num_sectors if self.variable_size else np.zeros_like(observations)

        # Only scheduled sectors' stored state advances this call
        if self.variable_size:
            for i in range(num_sectors):
                if schedule[i]:
                    self._prev_observations[i] = observations[i]
        else:
            self._prev_observations[schedule] = observations[schedule]
        self._prev_actions[schedule] = actions[schedule]
        self._prev_valid[schedule] = has_data[schedule]
        self.tilt_idx[schedule] = actions[schedule]

        logger.function("RLTiltController.decide end")
        return self.tilt_idx

    def update(self, observations: np.ndarray, rewards: np.ndarray, training: bool,
              has_data: np.ndarray = None, schedule: np.ndarray = None,
              terminal: bool = False) -> np.ndarray:
        """The ordinary "next_state" convention, fused: learn() from the
        PREVIOUS action using `observations`/`rewards`, THEN decide() the
        NEXT action from that SAME `observations` array -- i.e. reward is
        credited to the previous action, and the same measurement that
        confirmed it also becomes the input for what comes next. See
        decide()/learn()'s own docstrings for the split, "current_state"-
        reward-timing version, where the two calls use DIFFERENT
        observation arrays and can happen in either order.
        """
        self.learn(observations, rewards, training, has_data, schedule, terminal)
        return self.decide(observations, training, has_data, schedule)
