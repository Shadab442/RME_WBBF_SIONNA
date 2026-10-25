"""CellAlignedInfoProvider: per-sector FINE-GRID-CELL occupancy + median
SINR (no near/mid/far pooling at all) -- an alternative to
RegionalInfoProvider's own region-pooled STATE, added 2026-09-22 as an
OPTIONAL state_type ("cell_aligned"), NOT a replacement -- region_aligned
stays fully intact and selectable via config.yaml's own
algorithms.drl.state_type, kept side by side for comparison.

Motivation (user's own diagnosis, 2026-09-22): region-pooling collapses
cells with OPPOSITE tilt-sensitivity into the same bucket -- e.g. a
near-BS boundary cell that GAINS coverage under downtilt and a
far-from-BS boundary cell that LOSES coverage under the SAME downtilt can
both land in the same "far"/cell-edge region, averaging away exactly the
signal an agent needs to learn a coherent policy. Per-cell state removes
that conflation: every one of a sector's own valid fine-grid cells gets
its own occupancy count and own median SINR, tracked independently rather
than pooled.

Near/mid/far region concepts (CellularTopology.hotspot_region_membership)
are used HERE ONLY to determine which cells are "valid" (the same
inside-rhombus/outside-min_bs_ut_dist test near|mid|far's own union
already performs) -- the near/mid/far LABEL itself is discarded; this
class's own STATE has no region concept in it at all. As of 2026-09-22,
near/mid/far is used ONLY for mobility placement (demand_states) --
nothing about this class depends on which specific antenna configuration
or threshold rule defines those regions.

REWARD is entirely UNCHANGED by this class's existence -- reward always
comes from RegionalInfoProvider (network-wide covered-UE count/delta),
regardless of which STATE representation is active; this class produces
STATE only, and SimulationEngine constructs it ADDITIONALLY (not
instead) alongside RegionalInfoProvider when state_type="cell_aligned".

Same "one real measurement, accumulate() then compute()" per-turn
lifecycle as RegionalInfoProvider (see its own module docstring) -- this
class's own accumulate()/compute() are called from the SAME
SimulationEngine._measure_and_pool call site, fed the SAME already-
computed per-UE sector_idx/cell_idx/covered_ue/sinr_db (no extra radio
computation of any kind).

Own-cell SINR summary is median (p50) ONLY, not RegionalInfoProvider's
own richer p25/p50/p75 spread -- per explicit user request, a coarser
per-cell summary than region_aligned's own per-region one, since per-cell
state already trades pooling-robustness for spatial resolution. 0.0 for a
cell with zero UEs this turn, same convention as RegionalInfoProvider's
own own_sinr_p50.

Known scope limit (own-cell breakdown ONLY, not the neighbor pair below):
the per-FINE-CELL own-occupancy breakdown only supports config.yaml's own
non_anchor_region_handling="exclude" (gated by accumulate's own
cell_own_mask) -- "rme"/"radio_map"'s own region_idx_override reassigns a
non-anchor UE to an INFERRED REGION, which has no per-fine-cell equivalent
here (a coarse near/mid/far guess maps to no specific fine grid cell), so
this class has no way to use it for THAT breakdown. SimulationEngine.
_measure_and_pool (2026-10-24) enforces this at the call site: whenever
region_idx_override is active, it passes cell_own_mask=location_reporting_
mask regardless (excluding non-anchor UEs from the own-cell breakdown,
same as "exclude" would) -- passing None there would otherwise be read as
"no filtering at all" (see accumulate's own cell_own_mask docstring),
silently leaking every non-anchor UE's TRUE cell position into cell-level
state regardless of anchor status. Found 2026-10-24 when a 20%-anchor
radio_map run and a 100%-anchor run produced byte-identical coverage over
a full training run.

The NEIGHBOR coarse pair (sector_visit_count/sector_visit_count's own far-
ring restriction, see accumulate) has no such limit -- it's a coarse
near/mid/far question, not a per-cell one, so it DOES use
region_idx_override when active, exactly like RegionalInfoProvider's own
all_occupancy/all_sinr_p50 (same fix, same day -- this class's own
separate/duplicate implementation of that neighbor pair had NOT been kept
in sync with RegionalInfoProvider's own 2026-10-23 far-ring+anchor-gating
fix until this same pass; state_type="cell_aligned" is config.yaml's own
committed default, so that earlier fix had never actually been active in
any real training run until now).
"""

import numpy as np


class CellAlignedInfoProvider:
    """Per-sector FINE-GRID-CELL occupancy + median SINR, own cells only
    (no region pooling) -- plus each sector's own FAR-region-only coarse
    (occupancy, median SINR) pair, reused for neighbor state exactly like
    RegionalInfoProvider's own all_occupancy/all_sinr_p50 (same far-ring-
    only + own_region_mask/region_idx_override gating, fixed 2026-10-24 --
    this used to be "every real UE, anchor or not, whole sector", a stale
    copy of logic RegionalInfoProvider had already moved past; see
    accumulate's own docstring).

    :ivar num_valid_cells: valid (inside-rhombus, outside-min_bs_ut_dist)
        fine-grid cell count -- IDENTICAL for every sector by construction
        (same rhombus geometry, only boresight rotation differs).
    :ivar valid_cell_idx: [num_bs, num_valid_cells] int -- each sector's
        own FIXED, ascending-flat-index-ordered list of valid cell
        indices -- own_occupancy/own_sinr_p50's own column order, the same
        every interval (a sector's own geometry never changes mid-run).
    """

    def __init__(self, num_bs: int, grid_n: int, grid_points: np.ndarray, topology):
        """
        :param grid_points: [num_bs, grid_n*grid_n, 2] -- SimulationEngine's
            own grid_points, same source hotspot_region_membership's other
            callers use.
        :param topology: CellularTopology -- needs its own
            hotspot_region_membership (for the valid-cell mask only, see
            module docstring).
        """
        self.num_bs = num_bs
        self.num_grid_cells = grid_n * grid_n

        self.valid_mask = np.zeros((num_bs, self.num_grid_cells), dtype=bool)
        for s in range(num_bs):
            m = topology.hotspot_region_membership(s, grid_points[s])
            self.valid_mask[s] = m["near"] | m["mid"] | m["far"]
            if s == 0:
                # cell_to_region (2026-10-24): flat [num_grid_cells] int, same
                # near=0/mid=1/far=2 convention as OccupancyEstimator.__init__ --
                # sector-independent since every sector shares identical rhombus
                # geometry (only boresight rotation differs, same assumption
                # valid_counts' own assert below already relies on). Needed for
                # the neighbor coarse pair's far-ring-only filter (see accumulate).
                self.cell_to_region = np.where(m["near"], 0, np.where(m["mid"], 1, 2))
        self._far_region_idx = 2
        valid_counts = self.valid_mask.sum(axis=1)
        self.num_valid_cells = int(valid_counts[0])
        assert np.all(valid_counts == self.num_valid_cells), (
            "CellAlignedInfoProvider assumes every sector has the same valid-cell count (true by "
            "construction -- identical rhombus geometry per sector, only boresight rotation differs) "
            f"-- got {valid_counts.tolist()}")
        self.valid_cell_idx = np.array([np.where(self.valid_mask[s])[0] for s in range(num_bs)])

        # Reverse lookup: flat cell_idx -> column position in
        # valid_cell_idx[s], -1 for an excluded (invalid) cell.
        self._cell_col = np.full((num_bs, self.num_grid_cells), -1, dtype=int)
        for s in range(num_bs):
            self._cell_col[s, self.valid_cell_idx[s]] = np.arange(self.num_valid_cells)

        self.visit_count = np.zeros((num_bs, self.num_valid_cells))
        self._cell_sinr_values = [[[] for _ in range(self.num_valid_cells)] for _ in range(num_bs)]

        self.sector_visit_count = np.zeros(num_bs)
        self._sector_sinr_values = [[] for _ in range(num_bs)]

    def accumulate(self, sector_idx_geo: np.ndarray, cell_idx: np.ndarray, covered_ue: np.ndarray,
                  sinr_db: np.ndarray, own_region_mask: np.ndarray = None,
                  region_idx_override: np.ndarray = None, cell_own_mask: np.ndarray = None) -> None:
        """One turn's measurement.

        :param sector_idx_geo: [num_ue] geographic sector membership.
        :param cell_idx: [num_ue] UE's own fine-grid cell index WITHIN its
            own geographic sector's grid, same source.
        :param covered_ue: [num_ue] bool -- UNUSED here (own-cell state is
            occupancy/SINR only, no coverage breakdown -- kept as a
            parameter for call-site symmetry with RegionalInfoProvider.
            accumulate, whose own caller passes the same tuple to both).
        :param sinr_db: [num_ue] float -- every real UE's own real measured
            SINR, for own-cell median SINR.
        :param own_region_mask, region_idx_override: SAME semantics as
            RegionalInfoProvider.accumulate -- used ONLY for the neighbor
            far-ring coarse pair below (a coarse near/mid/far question, not
            a per-cell one, so an INFERRED region can be used here, unlike
            the per-fine-cell own-occupancy breakdown -- see module
            docstring's "Known scope limit").
        :param cell_own_mask: [num_ue] bool, optional -- gates the per-FINE-
            CELL own-occupancy breakdown specifically (typically the
            anchor/location-reporting mask; there is no per-cell equivalent
            of an inferred region, see module docstring). Defaults to
            own_region_mask when not given. SimulationEngine passes these
            as two DIFFERENT masks whenever a region is being inferred:
            own_region_mask=None (paired with region_idx_override, same as
            what RegionalInfoProvider gets, for the neighbor pair) and
            cell_own_mask=location_reporting_mask (for the per-cell
            breakdown, which can't use the inference at all).
        """
        if cell_own_mask is None:
            cell_own_mask = own_region_mask
        resolved = sector_idx_geo >= 0

        # Own-cell breakdown (per fine grid cell) -- gated by cell_own_mask only.
        own_resolved = resolved if cell_own_mask is None else (resolved & cell_own_mask)

        sec_own = sector_idx_geo[own_resolved]
        col = self._cell_col[sec_own, cell_idx[own_resolved]]
        in_valid = col >= 0
        sec_valid = sec_own[in_valid]
        col_valid = col[in_valid]
        np.add.at(self.visit_count, (sec_valid, col_valid), 1.0)

        sinr_own_valid = sinr_db[own_resolved][in_valid]
        if sec_valid.size > 0:
            pairs, inverse = np.unique(np.stack([sec_valid, col_valid], axis=1), axis=0, return_inverse=True)
            for k, (s, c) in enumerate(pairs):
                self._cell_sinr_values[s][c].extend(sinr_own_valid[inverse == k].tolist())

        # Neighbor coarse pair (2026-10-24, matching RegionalInfoProvider.
        # accumulate's own fix): FAR-region-only, gated by own_region_mask/
        # region_idx_override exactly like RegionalInfoProvider's own
        # all_occupancy/all_sinr_p50 -- NOT "every real UE regardless of
        # anchor status" as this used to read. A non-anchor UE with a known
        # INFERRED region (region_idx_override active) still counts here;
        # one with no region info at all (own_region_mask excludes it,
        # region_idx_override is None) does not.
        region_resolved = resolved if own_region_mask is None else (resolved & own_region_mask)
        if region_idx_override is None:
            region_idx = self.cell_to_region[cell_idx[region_resolved]]
        else:
            region_idx = region_idx_override[region_resolved]
        sec_region = sector_idx_geo[region_resolved]
        sinr_region = sinr_db[region_resolved]
        far_mask = region_idx == self._far_region_idx
        sec_far = sec_region[far_mask]
        sinr_far = sinr_region[far_mask]
        np.add.at(self.sector_visit_count, sec_far, 1.0)
        for s in np.unique(sec_far):
            vals = sinr_far[sec_far == s]
            if vals.size > 0:
                self._sector_sinr_values[s].extend(vals.tolist())

    def compute(self) -> tuple:
        """This turn's own-cell occupancy/median-SINR + whole-sector
        coarse (occupancy, median SINR) -- own_occupancy/all_occupancy are
        RAW UE counts (same convention as RegionalInfoProvider's own),
        own_sinr_p50/all_sinr_p50 are 0.0 wherever nothing was visited
        this turn.

        :output: (own_occupancy [num_bs, num_valid_cells],
            own_sinr_p50 [num_bs, num_valid_cells],
            all_occupancy [num_bs], all_sinr_p50 [num_bs]).
        """
        own_occupancy = self.visit_count.copy()
        self.visit_count[:] = 0.0

        own_sinr_p50 = np.zeros((self.num_bs, self.num_valid_cells))
        for s in range(self.num_bs):
            for c in range(self.num_valid_cells):
                vals = self._cell_sinr_values[s][c]
                if vals:
                    own_sinr_p50[s, c] = np.percentile(vals, 50)
        self._cell_sinr_values = [[[] for _ in range(self.num_valid_cells)] for _ in range(self.num_bs)]

        all_occupancy = self.sector_visit_count.copy()
        self.sector_visit_count[:] = 0.0

        all_sinr_p50 = np.zeros(self.num_bs)
        for s in range(self.num_bs):
            vals = self._sector_sinr_values[s]
            if vals:
                all_sinr_p50[s] = np.percentile(vals, 50)
        self._sector_sinr_values = [[] for _ in range(self.num_bs)]

        return own_occupancy, own_sinr_p50, all_occupancy, all_sinr_p50
