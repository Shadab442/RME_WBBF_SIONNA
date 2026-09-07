"""Multi-agent tilt-control policy interface.

Every sector always has the same fixed action set (one entry per candidate
in DOWNTILT_SWEEP_DEG)
"""

from abc import ABC, abstractmethod


class TiltPolicy(ABC):
    """Per-sector discrete-tilt-index policy contract.

    :ivar step_losses: training loss per learning step (whatever units the
        concrete policy learns in -- e.g. one appended value per sector per
        interval for Dqn); empty for policies that don't learn
        (e.g. RandomPolicy).
    """

    def __init__(self):
        self.step_losses = []

    @abstractmethod
    def act(self, observations, training, mask=None, default_action=None):
        """:param observations: [num_sectors, num_features].
        :param training: if True, exploration (e.g. epsilon-greedy) is
            active; if False, act greedily.
        :param mask: [num_sectors] bool, optional -- False skips
            exploring/predicting for that sector, using default_action
            instead.
        :param default_action: [num_sectors] int, optional -- action used
            for a sector where mask is False (e.g. 0 for "no data", or that
            sector's own held-over tilt index for "not its turn"). Zeros if
            omitted.
        :output: [num_sectors] int array of chosen action (tilt) indices.
        """

    @abstractmethod
    def observe(self, observations, actions, rewards, next_observations, terminal, mask=None):
        """One completed transition per sector: observations/actions is
        what was seen/chosen on entry to the interval that produced
        rewards/next_observations. May trigger learning.

        :param observations, next_observations: [num_sectors, num_features].
        :param actions: [num_sectors] int tilt indices.
        :param rewards: [num_sectors] float.
        :param terminal: bool, True only for a transition with no valid
            next state to bootstrap from.
        :param mask: [num_sectors] bool, optional -- False means this
            sector's transition isn't real (either the state it started
            from or the outcome it produced was undefined) and should be
            discarded rather than learned from.
        """

    def end_episode(self):
        """Hook for policies that want to do bookkeeping at an episode
        boundary; no-op by default. ("Episode" == one tilt control
        interval in this project -- see train_dqn_tilt_controller.py.)"""

    @abstractmethod
    def save(self, path):
        """Persist learned parameters (if any) to path."""
