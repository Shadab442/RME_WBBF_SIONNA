from collections import deque
import numpy as np
from helpers.utils import get_logger

logger = get_logger(__name__)

class ReplayBuffer:
    """Fixed-capacity transition replay.

    Transitions are appended in real temporal order for whatever this
    buffer belongs to (e.g. one sector's own successive decisions under
    Dqn's async schedule). A transition's gap_before marks that it is NOT
    a genuine temporal successor of the item before it (an episode ended,
    or an interval was skipped/masked in between) -- sample()'s windowed
    path and contiguous_recent_length() use this so a recurrent window
    never silently splices two unrelated stretches of history together.
    """

    def __init__(self, capacity, rng, sequence_length=None):
        """
        :param rng: a random.Random instance owning this buffer's
            sampling randomness.
        :param sequence_length: if given, sample() returns length-T
            contiguous windows (T = sequence_length) instead of single
            transitions.
        """
        self.items = deque(maxlen=capacity)
        self.gaps = deque(maxlen=capacity)
        self.rng = rng
        self.sequence_length = sequence_length

    def add(self, *transition, gap_before=False):
        self.items.append(tuple(np.array(value, copy=True) for value in transition))
        self.gaps.append(bool(gap_before))

    def sample(self, batch_size):
        logger.function("ReplayBuffer.sample start: batch_size=%d sequence_length=%s",
                        batch_size, self.sequence_length)
        if self.sequence_length is None:
            samples = self.rng.sample(self.items, batch_size)
            result = tuple(np.stack(values) for values in zip(*samples))
            logger.debug("ReplayBuffer.sample: i.i.d. sample, field shapes=%s",
                        [v.shape for v in result])
            logger.function("ReplayBuffer.sample end")
            return result

        # Sequence sampling: batch_size DISTINCT contiguous windows of
        # length sequence_length (distinct START positions, withoutreplacement 
        items = list(self.items)
        max_start = len(items) - self.sequence_length
        starts = self.rng.sample(range(max_start + 1), batch_size)
        windows = [items[start:start + self.sequence_length] for start in starts]
        logger.debug("ReplayBuffer.sample: buffer_len=%d max_start=%d", len(items), max_start)

        # Per window: T transitions -> per-field [T, *field_shape] arrays.
        windows_by_field = [[np.stack(field) for field in zip(*window)] for window in windows]

        # Across the batch: per field, [batch, T, *field_shape].
        num_fields = len(windows_by_field[0])
        result = tuple(np.stack([window[field] for window in windows_by_field])
                       for field in range(num_fields))
        logger.debug("ReplayBuffer.sample: windowed sample, field shapes=%s", [v.shape for v in result])
        logger.function("ReplayBuffer.sample end")
        
        return result

    def recent(self, n):
        """This buffer's last n transitions' obs.

        :param n: n=0 returns a genuinely EMPTY (0, feature_dim) array --
            Python's own list[-0:] equals list[0:] (the WHOLE list, since
            -0 == 0), not "the last zero items", so this needs an explicit
            guard rather than relying on negative-index slicing.
        """
        if n == 0:
            feature_dim = self.items[0][0].shape[0] if self.items else 0
            return np.zeros((0, feature_dim))
        return np.stack([item[0] for item in list(self.items)[-n:]])

    def __len__(self):
        return len(self.items)
