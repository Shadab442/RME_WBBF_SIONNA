"""Generic per-(sector, dimension) time-series forecasting -- the shared
model underneath both spatial-occupancy forecasting and neighbor-overshoot
forecasting on the predicted side of the spatial DRL state. V1: AR(1)/
exponential-moving-average -- swap this class for a fancier model (Graph-
ESN, etc.) later without touching its callers.
"""

import numpy as np


class TimeSeriesForecaster:
    """Tracks one running estimate per (sector, dimension) and forecasts
    the next value from the observed history.

    :ivar estimate: [num_bs, num_dims] current forecast (also doubles as
        the last-updated running estimate).
    """

    def __init__(self, num_bs: int, num_dims: int, smoothing: float = 0.5):
        """
        :param smoothing: AR(1)/EMA weight on the newest observation, in
            (0, 1] -- 1.0 reduces to a naive persistence forecast ("predict
            last observed value"); lower values smooth over more history.
        """
        self.smoothing = smoothing
        self.estimate = np.zeros((num_bs, num_dims))
        self._initialized = np.zeros(num_bs, dtype=bool)

    def update(self, rows: np.ndarray, observed: np.ndarray) -> None:
        """Feed this window's real observation for `rows` (bool mask,
        [num_bs]) -- updates the running estimate in place.

        :param observed: [num_bs, num_dims] -- only `rows` are read.
        """
        first_observation = rows & ~self._initialized
        smoothed = rows & self._initialized

        # First observation for a row: no history yet, take it as-is
        self.estimate[first_observation] = observed[first_observation]

        # Exponential smoothing against the running estimate
        self.estimate[smoothed] = (
            self.smoothing * observed[smoothed] + (1.0 - self.smoothing) * self.estimate[smoothed]
        )
        self._initialized[rows] = True

    def forecast(self, rows: np.ndarray) -> np.ndarray:
        """Current forecast for `rows` -- call after update() to get the
        prediction for the NEXT window.
        :output: [len(rows) or num_bs, num_dims] depending on whether
            `rows` is an index array or a bool mask.
        """
        return self.estimate[rows]
