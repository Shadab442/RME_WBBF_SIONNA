"""Baseline: pick a uniformly random tilt index per sector every interval.

Never learns -- exists as a cheap sanity check that the rest of the
pipeline (state computation, reward, training loop, plotting) is wired up
correctly, independent of whether Dqn is actually learning
anything. If Dqn can't beat this, something is likely broken,
not just "undertrained."
"""

import random

import numpy as np

from drl.base import TiltPolicy
from helpers.utils import get_logger

logger = get_logger(__name__)


class RandomPolicy(TiltPolicy):
    def __init__(self, num_sectors, num_actions, algorithm_seed=0):
        super().__init__()
        self.num_sectors = num_sectors
        self.num_actions = num_actions
        self.rng = random.Random(algorithm_seed)
        logger.info("RandomPolicy constructed: num_sectors=%d num_actions=%d algorithm_seed=%d",
                   num_sectors, num_actions, algorithm_seed)

    def act(self, observations, training, mask=None, default_action=None):
        logger.function("RandomPolicy.act start")
        actions = np.array(
            [self.rng.randrange(self.num_actions) for _ in range(self.num_sectors)],
            dtype=np.int64,
        )
        logger.debug("RandomPolicy.act: raw random actions=%s", actions.tolist())
        if mask is not None:
            defaults = np.zeros(self.num_sectors, dtype=np.int64) if default_action is None else default_action
            actions = np.where(mask, actions, defaults)
            logger.debug("RandomPolicy.act: mask applied, %d/%d sectors masked",
                        int((~mask).sum()), self.num_sectors)
        logger.function("RandomPolicy.act end")
        return actions

    def observe(self, observations, actions, rewards, next_observations, terminal, mask=None):
        pass

    def save(self, path):
        pass
