"""RegionalInfoProvider: everything the DRL loop needs per tilt-control
window that's keyed off "a sector + its actual (2-4) side-sharing
neighbors" -- both the STATE (this sector's own rhombus grid pooled into a
fixed grid of non-overlapping regions, PLUS one coarse (occupancy,
coverage) pair per neighbor) and the sector_neighbors REWARD (pooled
coverage over that same sector + neighbors footprint, one scalar total).
Merged into one class because both read the exact same per-slot inputs
and both key off the same sector + neighbor_ids relationship.

STATE geometry (replaces an earlier, now-removed design -- see git history
if you need it): a sector's OWN rhombus grid (CellularTopology.
build_sector_rhombus_grid, grid_n x grid_n cells, laid out in the grid's
own oblique per-sector basis) is pooled into block_size x block_size
blocks -- (grid_n // block_size)^2 REGIONS, fixed and IDENTICAL for every
sector (grid_n must be evenly divisible by block_size), non-overlapping,
and aligned to the sector's own orientation (a region is a parallelogram
in true (x, y), not a square -- it inherits the grid's own oblique shape).
A UE's own region is derived directly from CellularTopology.
assign_to_sector_rhombus_grid's own cell_idx (already a row-major index
over the sector's own (alpha, beta) grid) via integer block-grouping --
no separate geometry search needed, unlike the earlier per-sector
Cartesian-tile design this replaces.

Each of a sector's up to 4 neighbors contributes exactly one COARSE
(occupancy, coverage) pair -- its own whole-grid aggregate, computed
independently of that neighbor's own region breakdown (not a collapse of
it): total visits/covered anywhere in the neighbor's own geographic area,
not sub-divided into that neighbor's own regions at all.

Reward is accumulated directly over sector s's OWN window (in_xs = UEs
geographically in sector_set = [s] + its neighbors, accumulated from s's
last close to its next one), NOT pooled from each neighbor's own
separately-timed last completed window -- pooling from a neighbor's own
last-completed window would be a real temporal misalignment (part of the
"reward" could reflect conditions that predate the action being credited).
Accumulating directly over sector s's own window instead gives a clean
action -> measure X_s for exactly that action's own interval -> reward
causal chain, with no dependency on any neighbor's own close cadence, and
(bonus) no warm-up delay: reward is well-defined from sector s's own first
window, not gated on neighbors ever closing at all.

coverage_fill="kriged": both own-region coverage and a neighbor's coarse
coverage are derived from SpatialGridEstimator's own per-cell coverage
(reusing its EXISTING rsrp_per_sector Kriging -- Krige each sector's own
RSRP map, reconstruct SINR from ALL sectors' predictions, threshold --
rather than re-deriving that machinery here), aggregated over the fine
cells involved (a region's own block of cells; a neighbor's ENTIRE grid).
Since a region (and a neighbor's whole grid) never spans more than ONE
geographic sector now, this is a same-sector aggregate ALWAYS -- no
cross-sector "region spans multiple sectors closing asynchronously"
complexity remains; state_ready (see compute()) only depends on whether
THAT ONE sector's own coverage snapshot is valid yet.
"""

import numpy as np

from helpers.utils import get_logger

logger = get_logger(__name__)


class RegionalInfoProvider:
    """Per-sector region-grid occupancy/coverage STATE (own regions +
    neighbor coarse pairs) plus sector_neighbors scalar REWARD.

    :ivar visit_count, covered_count: [num_bs, num_own_regions] -- STATE's
        own-region counts, reset to 0 for a sector once its window closes.
    :ivar occupancy_ema: [num_bs, num_own_regions], persistent smoothed
        occupancy share, never reset.
    :ivar sector_visit_count, sector_covered_count: [num_bs] -- whole-grid
        (not blocked into regions) totals, same reset, used ONLY to answer
        "what is sector n's own coarse feedback" when n is someone's neighbor.
    :ivar sector_occupancy_ema: [num_bs], persistent, same EMA convention
        as occupancy_ema but one scalar per sector instead of per region.
    :ivar reward_visit_count, reward_covered_count: [num_bs] -- REWARD's
        own X_s-wide (sector + neighbors) totals accumulated over sector
        s's OWN currently-open window, reset when s itself closes.
    """

    def __init__(self, neighbor_ids: np.ndarray, grid_n: int, block_size: int,
                occupancy_ema_alpha: float = 0.8, coverage_fill: str = "zero"):
        """
        :param neighbor_ids: [num_bs, max_neighbors] int, -1-padded --
            CellularTopology.neighbor_ids.
        :param grid_n: rhombus grid side length (cells/side) -- must be
            evenly divisible by block_size.
        :param block_size: how many grid cells (per side) pool into one
            region -- e.g. grid_n=6, block_size=2 -> 9 regions/sector.
        :param coverage_fill: "zero" (0.0 where unvisited) | "kriged" (see
            module docstring).
        """
        if coverage_fill not in ("zero", "kriged"):
            raise ValueError(f"Unknown coverage_fill: {coverage_fill!r}")
        if grid_n % block_size != 0:
            raise ValueError(f"grid_n ({grid_n}) must be evenly divisible by block_size ({block_size})")

        self.num_bs = len(neighbor_ids)
        self.neighbor_sets = [ids[ids >= 0].tolist() for ids in neighbor_ids]
        self.grid_n = grid_n
        self.block_size = block_size
        self.blocks_per_side = grid_n // block_size
        self.num_own_regions = self.blocks_per_side ** 2
        self.cells_per_region = block_size ** 2
        self.occupancy_ema_alpha = occupancy_ema_alpha
        self.coverage_fill = coverage_fill

        # Static, sector-independent map: fine grid cell_idx (row-major
        # over a sector's own (alpha, beta)) -> which of its num_own_regions
        # blocks it falls in. Same for every sector -- block-grouping only
        # depends on grid_n/block_size, not on any sector's own geometry.
        alpha = np.arange(grid_n * grid_n) // grid_n
        beta = np.arange(grid_n * grid_n) % grid_n
        self.cell_to_region = (alpha // block_size) * self.blocks_per_side + (beta // block_size)

        # STATE -- own regions.
        self.visit_count = np.zeros((self.num_bs, self.num_own_regions))
        self.covered_count = np.zeros((self.num_bs, self.num_own_regions))
        self.occupancy_ema = np.zeros((self.num_bs, self.num_own_regions))
        self._occupancy_initialized = np.zeros(self.num_bs, dtype=bool)

        # STATE -- whole-sector totals, feeding neighbors' own coarse pair.
        self.sector_visit_count = np.zeros(self.num_bs)
        self.sector_covered_count = np.zeros(self.num_bs)
        self.sector_occupancy_ema = np.zeros(self.num_bs)
        self._sector_occupancy_initialized = np.zeros(self.num_bs, dtype=bool)

        # kriged only -- a region/neighbor-aggregate never spans more than
        # one geographic sector now, so readiness is just "has THIS sector
        # produced a valid (non-sentinel) coverage snapshot yet" -- a
        # zero-visit close leaves EVERY cell at -1 (SpatialGridEstimator's
        # own documented contract), which must not count as "completed".
        self.last_completed_coverage_per_cell = None
        self._coverage_per_cell_completed = np.zeros(self.num_bs, dtype=bool)

        # Network-wide visit total tracked as a running cumulative sum, not
        # a fixed-length window -- see spatial_grid_estimator.py's own
        # cumulative/snapshot-diff normalization (Finding 3 fix) for why a
        # fixed-K rolling window silently under-counts a sector's own
        # first-ever (longer than steady-state) window under async staggering.
        self._cumulative_visits = 0.0
        self._last_reset_cumulative = np.zeros(self.num_bs)

        # REWARD
        self.reward_visit_count = np.zeros(self.num_bs)
        self.reward_covered_count = np.zeros(self.num_bs)

    def accumulate(self, sector_idx_geo: np.ndarray, cell_idx: np.ndarray, covered_ue: np.ndarray) -> None:
        """One measurement slot.

        :param sector_idx_geo: [num_ue] geographic sector membership, from
            CellularTopology.assign_to_sector_rhombus_grid -- -1 excluded.
        :param cell_idx: [num_ue] this UE's own fine-grid cell index WITHIN
            its own geographic sector's grid, same source.
        :param covered_ue: [num_ue] bool.
        """
        logger.function("RegionalInfoProvider.accumulate start")
        resolved = sector_idx_geo >= 0
        self._cumulative_visits += float(resolved.sum())

        # STATE -- own regions: one vectorized scatter across ALL sectors
        # at once (a UE's region only depends on its OWN sector + cell_idx,
        # unlike the old X_s-spanning design's per-sector loop).
        region_idx = self.cell_to_region[cell_idx[resolved]]
        np.add.at(self.visit_count, (sector_idx_geo[resolved], region_idx), 1.0)
        np.add.at(self.covered_count, (sector_idx_geo[resolved], region_idx), covered_ue[resolved].astype(float))

        # STATE -- whole-sector totals (for neighbors' own coarse pair).
        np.add.at(self.sector_visit_count, sector_idx_geo[resolved], 1.0)
        np.add.at(self.sector_covered_count, sector_idx_geo[resolved], covered_ue[resolved].astype(float))

        # REWARD: every UE geographically in X_s = [s] + neighbors counts,
        # regardless of which region/block it falls in -- still an O(num_bs)
        # per-sector loop since sector_set varies per s (unchanged from the
        # design this class has always used for reward).
        for s in range(self.num_bs):
            in_xs = np.isin(sector_idx_geo, [s] + self.neighbor_sets[s])
            self.reward_visit_count[s] += float(in_xs.sum())
            self.reward_covered_count[s] += float(covered_ue[in_xs].sum())
        logger.function("RegionalInfoProvider.accumulate end")

    def compute(self, closes: np.ndarray, coverage_per_cell: np.ndarray = None) -> tuple:
        """Resolves the just-elapsed window for every sector in `closes`.

        :param coverage_per_cell: [num_bs, num_grid_cells] -- required
            (and only used) for coverage_fill="kriged".
        :output: (own_occupancy [num_bs, num_own_regions],
            own_coverage [num_bs, num_own_regions],
            neighbor_occupancy [num_bs] list of [n_neighbors[s]] arrays,
            neighbor_coverage [num_bs] list of [n_neighbors[s]] arrays,
            reward [num_bs], state_ready [num_bs] bool).
            Values for every sector are recomputed each call, but only
            closing sectors' occupancy_ema/sector_occupancy_ema is
            smoothed/persisted; non-closing sectors' entries are ignored by
            the caller (RLTiltController's schedule mask). reward is NaN
            for a closing sector with zero UEs in its own X_s this window.
            state_ready is always True for coverage_fill="zero"; for
            "kriged" a sector isn't ready until IT ITSELF has produced a
            valid coverage snapshot, AND -- for the purpose of consuming
            that sector's own state -- every one of ITS neighbors has too.
        """
        logger.function("RegionalInfoProvider.compute start: closes=%d", int(closes.sum()))
        assert self.coverage_fill != "kriged" or coverage_per_cell is not None, \
            "kriged needs coverage_per_cell"
        if self.coverage_fill == "kriged":
            # Snapshot ONLY sectors that both closed AND produced a valid
            # map this call -- see class/module docstring.
            if self.last_completed_coverage_per_cell is None:
                self.last_completed_coverage_per_cell = np.full_like(coverage_per_cell, -1.0)
            valid_this_call = closes & np.any(coverage_per_cell >= 0.0, axis=1)
            self.last_completed_coverage_per_cell[valid_this_call] = coverage_per_cell[valid_this_call]
            self._coverage_per_cell_completed[valid_this_call] = True

        total_visits_network = self._cumulative_visits - self._last_reset_cumulative  # [num_bs]

        # --- Own-region occupancy: EMA of this window's per-region visit
        # share of the whole network's visits, vectorized over all sectors.
        denom = np.where(total_visits_network > 0, total_visits_network, 1.0)
        occ_tilde = np.where(total_visits_network[:, None] > 0, self.visit_count / denom[:, None], 0.0)
        first_obs = closes & ~self._occupancy_initialized
        smoothed = closes & self._occupancy_initialized
        self.occupancy_ema[first_obs] = occ_tilde[first_obs]
        self.occupancy_ema[smoothed] = (self.occupancy_ema_alpha * occ_tilde[smoothed]
                                        + (1.0 - self.occupancy_ema_alpha) * self.occupancy_ema[smoothed])
        self._occupancy_initialized[closes] = True
        own_occupancy = self.occupancy_ema.copy()

        # --- Whole-sector occupancy (feeds neighbors' coarse pair), same EMA.
        sector_occ_tilde = np.where(total_visits_network > 0, self.sector_visit_count / denom, 0.0)
        first_obs_sector = closes & ~self._sector_occupancy_initialized
        smoothed_sector = closes & self._sector_occupancy_initialized
        self.sector_occupancy_ema[first_obs_sector] = sector_occ_tilde[first_obs_sector]
        self.sector_occupancy_ema[smoothed_sector] = (
            self.occupancy_ema_alpha * sector_occ_tilde[smoothed_sector]
            + (1.0 - self.occupancy_ema_alpha) * self.sector_occupancy_ema[smoothed_sector])
        self._sector_occupancy_initialized[closes] = True

        # --- Own-region coverage.
        if self.coverage_fill == "kriged":
            own_ready = self._coverage_per_cell_completed
            own_coverage = np.zeros((self.num_bs, self.num_own_regions))
            for s in np.where(own_ready)[0]:
                own_coverage[s] = (np.bincount(self.cell_to_region, weights=self.last_completed_coverage_per_cell[s],
                                               minlength=self.num_own_regions) / self.cells_per_region)
        else:
            own_ready = np.ones(self.num_bs, dtype=bool)
            visited = self.visit_count > 0
            with np.errstate(invalid="ignore"):
                own_coverage = np.where(visited, self.covered_count / np.where(visited, self.visit_count, 1), 0.0)

        # --- Neighbor coarse (occupancy, coverage): each sector's own
        # whole-grid aggregate, looked up per neighbor -- independent of
        # that neighbor's own region breakdown.
        if self.coverage_fill == "kriged":
            neighbor_coarse_coverage_by_sector = np.zeros(self.num_bs)
            for n in np.where(own_ready)[0]:
                neighbor_coarse_coverage_by_sector[n] = self.last_completed_coverage_per_cell[n].mean()
        else:
            sector_visited = self.sector_visit_count > 0
            with np.errstate(invalid="ignore"):
                neighbor_coarse_coverage_by_sector = np.where(
                    sector_visited, self.sector_covered_count / np.where(sector_visited, self.sector_visit_count, 1), 0.0)

        neighbor_occupancy, neighbor_coverage, state_ready_out = [], [], []
        for s in range(self.num_bs):
            neighbors = self.neighbor_sets[s]
            neighbor_occupancy.append(self.sector_occupancy_ema[neighbors].copy())
            neighbor_coverage.append(neighbor_coarse_coverage_by_sector[neighbors].copy())
            state_ready_out.append(bool(own_ready[s] and all(own_ready[n] for n in neighbors)))

        # Reset per-window counters for closing sectors.
        self.visit_count[closes] = 0.0
        self.covered_count[closes] = 0.0
        self.sector_visit_count[closes] = 0.0
        self.sector_covered_count[closes] = 0.0
        self._last_reset_cumulative[closes] = self._cumulative_visits

        # --- REWARD -- purely sector s's own just-elapsed window over its own X_s.
        reward = np.full(self.num_bs, np.nan)
        for s in np.where(closes)[0]:
            if self.reward_visit_count[s] > 0:
                reward[s] = self.reward_covered_count[s] / self.reward_visit_count[s]
            self.reward_visit_count[s] = 0.0
            self.reward_covered_count[s] = 0.0

        logger.function("RegionalInfoProvider.compute end")
        return own_occupancy, own_coverage, neighbor_occupancy, neighbor_coverage, reward, np.array(state_ready_out)
