"""Per-sector spatial grid accumulator for the local RL state/reward
(RME-WBBF RL Problem Formulation, slide 8): for every cell in a sector's
own observation region X_s, tracks smoothed occupancy rho_x and coverage
c_x (Kriged where unvisited). The same per-cell (visit, covered) counts
this window also directly give the sector's own local reward -- aggregate
over its own cells, no separate accumulator needed.

Replaces the old top-K-ranking estimator pair (spatial_occupancy_estimator.py
/ spatial_radio_state_estimator.py): there's no more ranking/selection step
now that every cell is reported, so one class covers what used to need two.

Two interchangeable radio_map_method options for filling in coverage at
UNVISITED cells (config-selected, meant to be compared empirically rather
than assumed -- see project memory):
  "sinr_direct"      -- Krige the observed SINR [dB] directly (a continuous
                        physical quantity, unlike a bounded coverage
                        fraction), then threshold the prediction.
  "rsrp_per_sector"  -- Krige each of the num_bs sectors' own RSRP [dB] map
                        separately, then reconstruct SINR from ALL sectors'
                        predictions (not a neighbor-limited subset -- an
                        empirical check found even a 10-neighbor halo
                        captures only ~56% of true interference on average
                        for this topology, so limiting the reconstruction
                        would silently reintroduce that bias).
"""

import numpy as np

from algorithm.kriging import OrdinaryKriging
from helpers.utils import get_logger

logger = get_logger(__name__)


class SpatialGridEstimator:
    """
    :ivar visit_count: [num_bs, num_grid_cells] running UE-occupancy count
        this window (reset to 0 for a sector once its window closes).
    :ivar covered_count: [num_bs, num_grid_cells] running covered-UE count
        this window, same reset.
    :ivar occupancy_ema: [num_bs, num_grid_cells] persistent smoothed
        occupancy share -- rho_x[k] = alpha*rho_tilde_x[k] + (1-alpha)*rho_x[k-1],
        never reset (only updated) across windows.
    """

    def __init__(self, num_bs: int, num_grid_cells: int, grid_points: np.ndarray,
                length_scale: float, coverage_threshold_db: float,
                radio_map_method: str = "sinr_direct", noise_power_w: float = None,
                occupancy_ema_alpha: float = 0.8, variance: float = 1.0):
        """
        :param grid_points: [num_bs, num_grid_cells, 2] fixed world (x, y)
            cell centers, from CellularTopology.build_sector_rhombus_grid --
            Kriging's target/known coordinates.
        :param length_scale, variance: OrdinaryKriging's RBF covariance
            parameters for filling in coverage at unvisited cells.
        :param coverage_threshold_db: SINR threshold -- both methods Krige a
            continuous quantity (SINR or RSRP) and threshold it into a
            binary coverage prediction at unvisited cells.
        :param radio_map_method: "sinr_direct" or "rsrp_per_sector" -- see
            module docstring.
        :param noise_power_w: required for "rsrp_per_sector" (SINR
            reconstruction needs the same noise floor KpiManager uses).
        :param occupancy_ema_alpha: EMA weight on this window's new
            occupancy observation, in (0, 1] -- 1.0 reduces to no smoothing
            ("use this window's value as-is").
        """
        assert radio_map_method in ("sinr_direct", "rsrp_per_sector"), \
            "radio_map_method must be 'sinr_direct' or 'rsrp_per_sector'"
        assert radio_map_method != "rsrp_per_sector" or noise_power_w is not None, \
            "rsrp_per_sector needs noise_power_w to reconstruct SINR"

        self.num_bs = num_bs
        self.num_grid_cells = num_grid_cells
        self.grid_points = grid_points
        self.coverage_threshold_db = coverage_threshold_db
        self.radio_map_method = radio_map_method
        self.noise_power_w = noise_power_w
        self.occupancy_ema_alpha = occupancy_ema_alpha
        self.kriging = OrdinaryKriging(length_scale, variance)

        self.visit_count = np.zeros((num_bs, num_grid_cells))
        self.covered_count = np.zeros((num_bs, num_grid_cells))
        self.occupancy_ema = np.zeros((num_bs, num_grid_cells))
        self._occupancy_initialized = np.zeros(num_bs, dtype=bool)

        # Network-wide visit total is tracked as a running cumulative sum,
        # NOT a fixed-length rolling window -- a sector's own visit_count
        # numerator covers "since its last reset" (or since t=0, for its
        # FIRST-ever window, which under async staggering is longer than a
        # steady-state window by its own phase offset). Snapshotting the
        # cumulative total at each sector's own last reset and diffing
        # against the current total reproduces exactly the same span as
        # that sector's own numerator, at startup and in steady state alike
        # -- a fixed K-slot window (see git history) gets this right in
        # steady state but silently under-counts the denominator on a
        # sector's first-ever (longer than K) window.
        self._cumulative_visits = 0.0
        self._last_reset_cumulative = np.zeros(num_bs)
        if radio_map_method == "sinr_direct":
            self.sinr_db_sum = np.zeros((num_bs, num_grid_cells))
        else:
            # [transmitting sector, geographic grid sector, cell]
            self.rsrp_db_sum = np.zeros((num_bs, num_bs, num_grid_cells))

    def accumulate(self, sector_idx: np.ndarray, cell_idx: np.ndarray, covered_ue: np.ndarray,
                   sinr_db: np.ndarray = None, power_db_all_sectors: np.ndarray = None) -> None:
        """Accumulate one measurement slot's occupancy/coverage. Each UE
        contributes one count to the geographic grid cell it occupies --
        multiple UEs in the same cell are each counted individually.

        :param sector_idx, cell_idx: [num_ue] geographic grid assignment,
            from CellularTopology.assign_to_sector_rhombus_grid -- -1 where
            a UE falls outside every sector's grid (excluded below).
        :param covered_ue: [num_ue] bool, this slot's per-UE coverage indicator.
        :param sinr_db: [num_ue] float, this slot's per-UE serving SINR --
            required for radio_map_method="sinr_direct".
        :param power_db_all_sectors: [num_bs, num_ue] float, every sector's
            own received power at every UE -- required for
            radio_map_method="rsrp_per_sector".
        """
        logger.function("SpatialGridEstimator.accumulate start")
        assert self.radio_map_method != "sinr_direct" or sinr_db is not None, \
            "sinr_direct needs sinr_db"
        assert self.radio_map_method != "rsrp_per_sector" or power_db_all_sectors is not None, \
            "rsrp_per_sector needs power_db_all_sectors"

        resolved = (sector_idx >= 0) & (cell_idx >= 0) & (cell_idx < self.num_grid_cells)
        self._cumulative_visits += float(resolved.sum())
        np.add.at(self.visit_count, (sector_idx[resolved], cell_idx[resolved]), 1.0)
        np.add.at(self.covered_count, (sector_idx[resolved], cell_idx[resolved]),
                  covered_ue[resolved].astype(float))
        if self.radio_map_method == "sinr_direct":
            np.add.at(self.sinr_db_sum, (sector_idx[resolved], cell_idx[resolved]), sinr_db[resolved])
        else:
            for b in range(self.num_bs):
                np.add.at(self.rsrp_db_sum[b], (sector_idx[resolved], cell_idx[resolved]),
                         power_db_all_sectors[b, resolved])
        logger.debug("SpatialGridEstimator.accumulate: resolved=%d/%d", int(resolved.sum()), resolved.size)
        logger.function("SpatialGridEstimator.accumulate end")

    def compute(self, closes: np.ndarray) -> tuple:
        """Resolves the just-elapsed window for every sector in `closes`:
        smoothed per-cell occupancy, per-cell coverage (Kriged where
        unvisited), and each sector's own local reward (this window's
        coverage aggregated over its own cells). Resets window counters for
        closing sectors; occupancy_ema persists (it's a running estimate,
        not a per-window quantity).

        :output: (occupancy [num_bs, num_grid_cells], coverage
            [num_bs, num_grid_cells], reward [num_bs] -- NaN for a sector
            with zero visits this window, i.e. no local UEs to score).
        """
        logger.function("SpatialGridEstimator.compute start: closes=%d", int(closes.sum()))

        # rho_tilde_x[k] = this window's occupancy SHARE OF THE WHOLE
        # NETWORK's visits over exactly the SAME span as this sector's own
        # visit_count numerator -- NOT self.visit_count.sum() (mixes a
        # closing sector's full window with others' live partial ones under
        # async staggering) and NOT a fixed K-slot rolling total either (a
        # sector's own FIRST-ever window, before it has closed once, spans
        # "since t=0" -- offset+K slots, longer than K -- so a fixed-K
        # denominator would under-count it). Diffing the running cumulative
        # total against the snapshot taken at this sector's own last reset
        # reproduces the exact span its numerator covers, at startup and in
        # steady state alike.
        total_visits_network = self._cumulative_visits - self._last_reset_cumulative  # [num_bs]
        denom = np.where(total_visits_network > 0, total_visits_network, 1.0)
        occupancy_tilde = np.where(total_visits_network[:, None] > 0,
                                   self.visit_count / denom[:, None], 0.0)

        first_observation = closes & ~self._occupancy_initialized
        smoothed = closes & self._occupancy_initialized
        self.occupancy_ema[first_observation] = occupancy_tilde[first_observation]
        self.occupancy_ema[smoothed] = (
            self.occupancy_ema_alpha * occupancy_tilde[smoothed]
            + (1.0 - self.occupancy_ema_alpha) * self.occupancy_ema[smoothed]
        )
        self._occupancy_initialized[closes] = True

        # c_x[k]: instantaneous coverage where visited (a real empirical
        # observation either way); where unvisited, Krige a continuous
        # physical quantity and threshold it -- see radio_map_method.
        visited = self.visit_count > 0
        with np.errstate(invalid="ignore"):
            coverage_instant = np.where(visited, self.covered_count / np.where(visited, self.visit_count, 1), -1.0)
        coverage = coverage_instant.copy()
        for i in np.where(closes)[0]:
            known = visited[i]
            if not known.any():
                logger.warning("SpatialGridEstimator.compute: sector=%d has zero visits this window "
                              "-- coverage left at -1 for every cell", i)
                continue
            unknown = ~known
            if not unknown.any():
                continue
            if self.radio_map_method == "sinr_direct":
                mean_sinr_db = np.where(known, self.sinr_db_sum[i] / np.where(known, self.visit_count[i], 1), 0.0)
                predicted_sinr_db = self.kriging.predict(self.grid_points[i, known], mean_sinr_db[known],
                                                         self.grid_points[i, unknown])
                coverage[i, unknown] = (predicted_sinr_db > self.coverage_threshold_db).astype(float)
            else:
                predicted_rsrp_db = np.zeros((self.num_bs, int(unknown.sum())))
                for b in range(self.num_bs):
                    mean_rsrp_db_b = np.where(known, self.rsrp_db_sum[b, i] / np.where(known, self.visit_count[i], 1), 0.0)
                    predicted_rsrp_db[b] = self.kriging.predict(self.grid_points[i, known], mean_rsrp_db_b[known],
                                                                self.grid_points[i, unknown])
                # RSRP-based attachment: the SERVER is whichever sector's
                # predicted power is strongest at each cell, not always the
                # geographic sector i (a neighbor can legitimately win a
                # boundary/overlap cell) -- matches KpiManager's own
                # argmax-power attachment rule.
                predicted_rsrp_w = np.power(10.0, predicted_rsrp_db / 10.0)
                serving_w = predicted_rsrp_w.max(axis=0)
                interference_w = predicted_rsrp_w.sum(axis=0) - serving_w
                predicted_sinr_db = 10.0 * np.log10(serving_w / (interference_w + self.noise_power_w))
                coverage[i, unknown] = (predicted_sinr_db > self.coverage_threshold_db).astype(float)

        # Local reward: this window's coverage aggregated over the sector's
        # OWN cells -- the same counts already accumulated above, not a
        # separate served-based accumulator.
        sector_visits = self.visit_count.sum(axis=1)
        with np.errstate(invalid="ignore"):
            reward = np.where(sector_visits > 0, self.covered_count.sum(axis=1) / np.where(sector_visits > 0, sector_visits, 1), np.nan)

        self.visit_count[closes] = 0.0
        self.covered_count[closes] = 0.0
        self._last_reset_cumulative[closes] = self._cumulative_visits
        if self.radio_map_method == "sinr_direct":
            self.sinr_db_sum[closes] = 0.0
        else:
            self.rsrp_db_sum[:, closes, :] = 0.0
        logger.debug("SpatialGridEstimator.compute: mean occupancy=%.4f mean coverage=%.4f mean reward=%.4f",
                    float(self.occupancy_ema[closes].mean()) if closes.any() else float("nan"),
                    float(np.nanmean(coverage[closes])) if closes.any() else float("nan"),
                    float(np.nanmean(reward[closes])) if closes.any() else float("nan"))
        logger.function("SpatialGridEstimator.compute end")
        return self.occupancy_ema, coverage, reward


def pool_observation_windows(occupancy_per_cell: np.ndarray, coverage_per_cell: np.ndarray,
                             windows: list) -> tuple:
    """Pools SpatialGridEstimator's per-cell (occupancy, coverage) into each
    sector's 9 fixed-purpose observation windows (CellularTopology.
    compute_observation_windows) -- NOT a partition, windows may overlap and
    may draw cells from other sectors (halo/corner context). Per the RL
    Problem Formulation slide's pooling formulas: occupancy is SUMMED over a
    window's members (rho_G = sum rho_x), coverage is MEANED (c_G = mean c_x).

    :param occupancy_per_cell, coverage_per_cell: [num_bs, num_grid_cells],
        SpatialGridEstimator.compute's outputs.
    :param windows: [num_bs] list of dict[window_name -> list[(sector_idx, cell_idx)]].
    :output: (occupancy_windows [num_bs, 9], coverage_windows [num_bs, 9], window_names).
    """
    logger.function("pool_observation_windows start")
    num_bs = len(windows)
    window_names = list(windows[0].keys())
    occupancy_windows = np.zeros((num_bs, len(window_names)))
    coverage_windows = np.zeros((num_bs, len(window_names)))
    for i in range(num_bs):
        for wi, wname in enumerate(window_names):
            members = windows[i][wname]
            sectors = [s for s, _ in members]
            cells = [c for _, c in members]
            occupancy_windows[i, wi] = occupancy_per_cell[sectors, cells].sum()
            coverage_windows[i, wi] = coverage_per_cell[sectors, cells].mean()
    logger.debug("pool_observation_windows: %d windows/sector", len(window_names))
    logger.function("pool_observation_windows end")
    return occupancy_windows, coverage_windows, window_names
