"""Given a per-cell importance matrix (see helpers/spatial_occupancy_estimator.py),
select the top-K important cells per sector and resolve the radio state
(coverage, neighbor overshoot) there. Deliberately separate from occupancy
estimation: this module only answers "what's the radio state," never
"which cells matter."
"""

import numpy as np

from algorithm.kriging import OrdinaryKriging


class SpatialRadioStateEstimator:
    """Common contract: given an importance matrix, return
    (top_k_identity, top_k_coverage, neighbor_overshoot) for whichever
    sectors' windows just closed.
    """

    def compute(self, closes, *args, **kwargs):
        raise NotImplementedError


def _select_top_k(importance: np.ndarray, top_k_locations: int, num_grid_cells: int) -> np.ndarray:
    """[num_bs, top_k_locations] cell indices, per sector, ranked by
    descending importance -- shared by both estimators below.
    """
    return np.argsort(-importance, axis=-1, kind="stable")[:, :top_k_locations]


class CausalSpatialRadioStateEstimator(SpatialRadioStateEstimator):
    """Owns the real per-slot coverage/neighbor-overshoot accumulator;
    reports the just-elapsed window's real values at the top-K important
    cells (importance comes from CausalSpatialOccupancyEstimator).

    :ivar grid_covered: [num_bs, num_grid_cells] running covered-visit sum this window.
    :ivar neighbor_served, neighbor_hit: [num_bs, max_neighbors] running counts this window.
    """

    def __init__(self, num_bs: int, num_grid_cells: int, max_neighbors: int,
                neighbor_ids: np.ndarray, top_k_locations: int):
        self.num_grid_cells = num_grid_cells
        self.top_k_locations = top_k_locations
        self.neighbor_ids = neighbor_ids
        self.grid_covered = np.zeros((num_bs, num_grid_cells))
        self.neighbor_served = np.zeros((num_bs, max_neighbors))
        self.neighbor_hit = np.zeros((num_bs, max_neighbors))

    def accumulate(
        self,
        sector_idx: np.ndarray,
        cell_idx: np.ndarray,
        covered_ue: np.ndarray,
        per_neighbor_served_count: np.ndarray,
        per_neighbor_overshoot_hit_count: np.ndarray,
    ) -> None:
        """Accumulate one measurement slot's spatial coverage and
        per-neighbor overshoot statistics.

        Each UE contributes one coverage observation to the geographical
        grid cell it occupies. Multiple UEs in the same cell are therefore
        counted individually.
        """
        resolved = (
            (sector_idx >= 0)
            & (cell_idx >= 0)
            & (cell_idx < self.num_grid_cells)
        )

        np.add.at(
            self.grid_covered,
            (sector_idx[resolved], cell_idx[resolved]),
            covered_ue[resolved].astype(float),
        )

        self.neighbor_served += per_neighbor_served_count
        self.neighbor_hit += per_neighbor_overshoot_hit_count

    def compute(self, closes: np.ndarray, importance: np.ndarray, visit_count: np.ndarray) -> tuple:
        """
        :param importance:
            [num_bs, num_grid_cells] spatial occupancy importance from
            CausalSpatialOccupancyEstimator, normalized across grid cells.

        :param visit_count:
            [num_bs, num_grid_cells] raw number of UE occupancy observations
            in each grid cell. Used as the denominator for spatial coverage.

        :output:
            top_k_identity:
                [num_bs, k] normalized identities of the most important cells.

            top_k_coverage:
                [num_bs, k] coverage fraction at those cells, or -1 if
                the selected cell has no UE observations.

            neighbor_overshoot:
                [num_bs, max_neighbors] per-neighbor overshoot fraction,
                or -1 where no observation/neighbor exists.

            grid_coverage:
                [num_bs, num_grid_cells] coverage fraction at EVERY cell
                (not just the top-K), -1 where unvisited -- the predicted
                estimator's kriging input (interpolates from wherever was
                actually observed, not just the top-K subset).
        """
        visited = visit_count > 0
        with np.errstate(invalid="ignore"):
            grid_coverage = np.where(visited, self.grid_covered / np.where(visited, visit_count, 1), -1.0)

        top_idx = _select_top_k(importance, self.top_k_locations, self.num_grid_cells)
        top_k_identity = top_idx / (self.num_grid_cells - 1)
        top_k_coverage = np.take_along_axis(grid_coverage, top_idx, axis=-1)

        has_neighbor_data = self.neighbor_served > 0
        with np.errstate(invalid="ignore"):
            neighbor_overshoot = np.where(has_neighbor_data,
                                          self.neighbor_hit / np.where(has_neighbor_data, self.neighbor_served, 1),
                                          -1.0)
        neighbor_overshoot = np.where(self.neighbor_ids >= 0, neighbor_overshoot, -1.0)

        self.grid_covered[closes] = 0.0
        self.neighbor_served[closes] = 0.0
        self.neighbor_hit[closes] = 0.0
        return top_k_identity, top_k_coverage, neighbor_overshoot, grid_coverage


class PredictedSpatialRadioStateEstimator(SpatialRadioStateEstimator):
    """Given a FORECASTED importance matrix (from
    PredictedSpatialOccupancyEstimator), selects the top-K cells and
    estimates coverage there by KRIGING from this window's REAL observed
    coverage (wherever UEs actually were, from
    CausalSpatialRadioStateEstimator) -- no propagation model, no channel
    evaluation, no tilt dependency: purely spatial interpolation from
    already-observed data. Neighbor overshoot is NOT estimated at all --
    the real, just-observed value is passed through unchanged (the top-K
    cells are this sector's own important locations, not the boundary
    population overshoot is actually about, so there's nothing to predict
    it from here).
    """

    def __init__(self, top_k_locations: int, num_grid_cells: int, grid_points: np.ndarray,
                length_scale: float, variance: float = 1.0):
        """
        :param grid_points: [num_bs, num_grid_cells, 2] fixed world (x, y)
            cell centers, from CellularTopology.build_sector_rhombus_grid.
        :param length_scale, variance: OrdinaryKriging's RBF covariance
            parameters -- V1 defaults (length_scale typically the same
            spatial_grid_cell_size_m used to size the grid), not yet
            empirically tuned.
        """
        self.top_k_locations = top_k_locations
        self.num_grid_cells = num_grid_cells
        self.grid_points = grid_points
        self.kriging = OrdinaryKriging(length_scale, variance)

    def compute(self, closes: np.ndarray, forecasted_importance: np.ndarray,
               real_grid_coverage: np.ndarray, real_neighbor_overshoot: np.ndarray) -> tuple:
        """:param forecasted_importance: [num_bs, num_grid_cells], from
            PredictedSpatialOccupancyEstimator.compute().
        :param real_grid_coverage: [num_bs, num_grid_cells], from
            CausalSpatialRadioStateEstimator.compute() -- this window's
            REAL coverage wherever visited, -1 elsewhere; kriging's
            training data.
        :param real_neighbor_overshoot: [num_bs, max_neighbors], from
            CausalSpatialRadioStateEstimator.compute() -- passed through
            unchanged, not predicted.
        :output: same shape/semantics as CausalSpatialRadioStateEstimator.compute
            (minus the grid_coverage return, which only the causal side needs
            to expose).
        """
        top_idx = _select_top_k(forecasted_importance, self.top_k_locations, self.num_grid_cells)
        top_k_identity = top_idx / (self.num_grid_cells - 1)

        top_k_coverage = np.full((forecasted_importance.shape[0], self.top_k_locations), -1.0)
        for i in np.where(closes)[0]:
            known = real_grid_coverage[i] >= 0
            if not known.any():
                continue 
            target_xy = self.grid_points[i, top_idx[i]]
            predicted = self.kriging.predict(self.grid_points[i, known], real_grid_coverage[i, known], target_xy)
            # Kriging is a linear predictor, not constrained to the [0, 1]
            # coverage-fraction range its inputs live in -- clip for safety.
            top_k_coverage[i] = np.clip(predicted, 0.0, 1.0)

        return top_k_identity, top_k_coverage, real_neighbor_overshoot
