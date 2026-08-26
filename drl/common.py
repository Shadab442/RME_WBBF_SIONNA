"""Shared Q-network and replay buffer.

No action masking anywhere here (unlike a variable-candidate-set problem)
-- every sector's action space is the same fixed size, always fully valid.
"""

from collections import deque

import numpy as np
import torch
from torch import nn


class Mlp(nn.Module):
    """Plain MLP Q-network: state (a small fixed-length feature vector, not
    a candidate sequence) -> one Q-value per discrete tilt index.

    input_size -> hidden_sizes[0] -> ... -> hidden_sizes[-1] -> output_size.
    Each hidden layer is Linear+ReLU; the final layer is a plain Linear (no
    activation -- these are Q-value heads).
    """

    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()
        sizes = [input_size, *hidden_sizes]
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers += [nn.Linear(in_size, out_size), nn.ReLU()]
        layers.append(nn.Linear(sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.layers(inputs)


class ReplayBuffer:
    """Fixed-capacity transition replay, sampled via a DEDICATED rng --
    never Python's global `random` module -- so algorithm-side randomness
    stays fully isolated and reproducible from an explicit seed, the same
    isolation principle used for the environment's own RNG
    (helpers/simulation_engine.py).

    Transitions are appended in real temporal order for whatever this
    buffer belongs to (e.g. one sector's own successive decisions under
    IndependentDqn's async schedule). With `sequence_length` given,
    `sample()` exploits that order to hand back contiguous WINDOWS instead
    of i.i.d. transitions -- for a recurrent Q-network (e.g. Wesn) that
    needs a run of recent history to build up state from, not a single
    vector.
    """

    def __init__(self, capacity, rng, sequence_length=None):
        """:param rng: a random.Random instance (NOT the global `random`
            module) owning this buffer's sampling randomness.
        :param sequence_length: if given, sample() returns length-T
            contiguous windows (T = sequence_length) instead of single
            transitions.
        """
        self.items = deque(maxlen=capacity)
        self.rng = rng
        self.sequence_length = sequence_length

    def add(self, *transition):
        self.items.append(tuple(np.array(value, copy=True) for value in transition))

    def sample(self, batch_size):
        if self.sequence_length is None:
            samples = self.rng.sample(self.items, batch_size)
            return tuple(np.stack(values) for values in zip(*samples))

        # Sequence sampling: batch_size random contiguous windows of length
        # sequence_length. Only the window's LAST transition is the real
        # training target -- the earlier steps only give the recurrent
        # network context to build up state from (see IndependentDqn._learn).
        items = list(self.items)
        max_start = len(items) - self.sequence_length
        starts = [self.rng.randint(0, max_start) for _ in range(batch_size)]
        windows = [items[start:start + self.sequence_length] for start in starts]

        # Per window: T transitions -> per-field [T, *field_shape] arrays.
        windows_by_field = [[np.stack(field) for field in zip(*window)] for window in windows]

        # Across the batch: per field, [batch, T, *field_shape].
        num_fields = len(windows_by_field[0])
        return tuple(np.stack([window[field] for window in windows_by_field])
                    for field in range(num_fields))

    def recent(self, n):
        """This buffer's last n transitions' FIRST field (its `obs`),
        oldest first -- used to assemble a recurrent policy's current
        decision window alongside the just-observed state.
        """
        return np.stack([item[0] for item in list(self.items)[-n:]])

    def __len__(self):
        return len(self.items)
