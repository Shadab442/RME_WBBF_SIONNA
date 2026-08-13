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

    def act(self, observations, training):
        return np.array(
            [self.rng.randrange(self.num_actions) for _ in range(self.num_sectors)],
            dtype=np.int64,
        )

    def observe(self, observations, actions, rewards, next_observations, terminal):
        pass

    def save(self, path):
        pass
