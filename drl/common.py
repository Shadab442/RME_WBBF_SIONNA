"""Shared Q-network and replay buffer.

No action masking anywhere here (unlike a variable-candidate-set problem)
-- every sector's action space is the same fixed size, always fully valid.
"""

from collections import deque

import numpy as np
import torch
from torch import nn


def build_mlp_stack(input_size, hidden_sizes, output_size):
    """input_size -> hidden_sizes[0] -> ... -> hidden_sizes[-1] -> output_size.

    Each hidden layer is Linear+ReLU; the final layer is a plain Linear (no
    activation -- these are Q-value heads).
    """
    sizes = [input_size, *hidden_sizes]
    layers = []
    for in_size, out_size in zip(sizes, sizes[1:]):
        layers += [nn.Linear(in_size, out_size), nn.ReLU()]
    layers.append(nn.Linear(sizes[-1], output_size))
    return nn.Sequential(*layers)


class Mlp(nn.Module):
    """Plain MLP Q-network: state (a small fixed-length feature vector, not
    a candidate sequence) -> one Q-value per discrete tilt index."""

    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        self.layers = build_mlp_stack(input_size, hidden_sizes, output_size)

    def forward(self, inputs):
        return self.layers(inputs)


class ReplayBuffer:
    """Fixed-capacity transition replay, sampled via a DEDICATED rng --
    never Python's global `random` module -- so algorithm-side randomness
    stays fully isolated and reproducible from an explicit seed, the same
    isolation principle used for the environment's own RNG
    (helpers/simulation_engine.py).
    """

    def __init__(self, capacity, rng):
        """:param rng: a random.Random instance (NOT the global `random`
            module) owning this buffer's sampling randomness."""
        self.items = deque(maxlen=capacity)
        self.rng = rng

    def add(self, *transition):
        self.items.append(tuple(np.array(value, copy=True) for value in transition))

    def sample(self, batch_size):
        samples = self.rng.sample(self.items, batch_size)
        return tuple(np.stack(values) for values in zip(*samples))

    def __len__(self):
        return len(self.items)
