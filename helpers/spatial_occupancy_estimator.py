"""Per-sector-rhombus-grid occupancy/importance estimation -- deliberately
decoupled from radio-state estimation (see helpers/spatial_radio_state_estimator.py):
this module only answers "which grid cells matter," never "what's the
radio state there."
"""

import numpy as np

from algorithm.time_series_forecaster import TimeSeriesForecaster


class SpatialOccupancyEstimator:
    """Common contract: produce a per-cell importance value in [0, 1]
    (visits/total_slots, or a forecast of it) for whichever sectors'
    windows just closed.
    """

    def compute(self, closes, *args, **kwargs):
        raise NotImplementedError


class CausalSpatialOccupancyEstimator(SpatialOccupancyEstimator):
    """Owns the real per-slot visitation accumulator; importance is the
    just-elapsed window's real visits/total_slots fraction per cell.

    :ivar grid_visit: [num_bs, num_grid_cells] running visit counts this window.
    """

    def __init__(self, num_bs: int, num_grid_cells: int, measurement_slots_per_interval: int):
        self.num_grid_cells = num_grid_cells
        self.measurement_slots_per_interval = measurement_slots_per_interval
        self.grid_visit = np.zeros((num_bs, num_grid_cells))

    def accumulate(self, sector_idx: np.ndarray, cell_idx: np.ndarray) -> None:
        """Accumulate UE occupancy for one measurement slot.

        Each UE contributes one count to the geographical grid cell it occupies.
        Thus, multiple UEs occupying the same cell in the same slot contribute
        multiple occupancy counts.
        """
        resolved = (sector_idx >= 0) & (cell_idx >= 0)

        np.add.at(
            self.grid_visit,
            (sector_idx[resolved], cell_idx[resolved]),
            1.0,
        )


    def compute(self, closes: np.ndarray) -> tuple:
        """Return per-cell occupancy importance and raw occupancy count.

        importance[i, g] is the fraction of sector i's total UE occupancy
        observations occurring in grid cell g during the current window.

        Only rows indicated by `closes` correspond to completed windows.
        """
        visit_count = self.grid_visit.copy()

        total_visits = visit_count.sum(axis=1, keepdims=True)

        importance = np.divide(
            visit_count,
            total_visits,
            out=np.zeros_like(visit_count, dtype=float),
            where=total_visits > 0,
        )

        self.grid_visit[closes] = 0.0

        return importance, visit_count


class PredictedSpatialOccupancyEstimator(SpatialOccupancyEstimator):
    """Forecasts next-window importance from the causal estimator's real,
    just-elapsed window importance -- an extended version of it: consumes
    CausalSpatialOccupancyEstimator's output as its own input (composition,
    not inheritance).
    """

    def __init__(self, num_bs: int, num_grid_cells: int, smoothing: float = 0.5):
        self.forecaster = TimeSeriesForecaster(num_bs, num_grid_cells, smoothing)

    def compute(self, closes: np.ndarray, real_importance: np.ndarray) -> np.ndarray:
        """:param real_importance: [num_bs, num_grid_cells], this window's
            REAL importance -- from CausalSpatialOccupancyEstimator.compute().
        :output: [num_bs, num_grid_cells] forecasted next-window importance
            -- full shape (same convention as CausalSpatialOccupancyEstimator.compute),
            only `closes` rows are actually meaningful.
        """
        self.forecaster.update(closes, real_importance)
        return self.forecaster.estimate
