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
    :ivar step_loss_sectors: sector index for each entry in step_losses
        (same length, same append order) -- lets a caller attribute a loss
        value back to the sector it came from; empty wherever step_losses
        is empty.
    :ivar last_decision_info: {sector: {"epsilon", "exploratory",
        "q_values"}} from the most recent act() call, for whichever
        sectors act() actually decided (see act()'s own mask param) --
        diagnostics-only, not used by any policy's own decision logic.
        Empty for a policy that doesn't expose this (e.g. RandomPolicy) or
        wasn't asked to (Dqn's own expose_diagnostics flag).
    """

    def __init__(self):
        self.step_losses = []
        self.step_loss_sectors = []
        self.last_decision_info = {}

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
    def observe(self, observations, actions, rewards, next_observations, terminal, mask=None, training=True):
        """One completed transition per sector: observations/actions is
        what was seen/chosen on entry to the interval that produced
        rewards/next_observations.

        :param observations, next_observations: [num_sectors, num_features].
        :param actions: [num_sectors] int tilt indices.
        :param rewards: [num_sectors] float.
        :param terminal: bool, True only for a transition with no valid
            next state to bootstrap from.
        :param mask: [num_sectors] bool, optional -- False means this
            sector's transition isn't real (either the state it started
            from or the outcome it produced was undefined) and should be
            discarded rather than learned from.
        :param training: if True, may trigger learning (and any other
            training-only bookkeeping, e.g. target-network sync) in
            addition to recording the transition. If False, the
            transition is still recorded (so any recurrent/windowed state
            a concrete policy keeps -- e.g. Dqn's replay buffer feeding its
            wesn model -- stays current), but no learning happens: "keep
            state fresh without learning," the counterpart to act's own
            training=False ("act greedily, don't explore").
        """

    def end_episode(self):
        """Hook for policies that want to do bookkeeping at an episode
        boundary; no-op by default. ("Episode" == one tilt control
        interval in this project -- see train_dqn_tilt_controller.py.)"""

    @abstractmethod
    def save(self, path):
        """Persist learned parameters (if any) to path."""
