"""Baseline: pick a uniformly random tilt index per sector every interval.

Never learns -- exists as a cheap sanity check that the rest of the
pipeline (state computation, reward, training loop, plotting) is wired up
correctly, independent of whether IndependentDqn is actually learning
anything. If IndependentDqn can't beat this, something is likely broken,
not just "undertrained."
"""

import random

import numpy as np

from drl.base import TiltPolicy


class RandomPolicy(TiltPolicy):
    def __init__(self, num_sectors, num_actions, algorithm_seed=0):
        super().__init__()
        self.num_sectors = num_sectors
        self.num_actions = num_actions
        self.rng = random.Random(algorithm_seed)

    def act(self, observations, training, mask=None, default_action=None):
        actions = np.array(
            [self.rng.randrange(self.num_actions) for _ in range(self.num_sectors)],
            dtype=np.int64,
        )
        if mask is not None:
            defaults = np.zeros(self.num_sectors, dtype=np.int64) if default_action is None else default_action
            actions = np.where(mask, actions, defaults)
        return actions

    def observe(self, observations, actions, rewards, next_observations, terminal, mask=None):
        pass

    def save(self, path):
        pass
