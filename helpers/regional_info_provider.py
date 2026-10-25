"""RegionalInfoProvider: per-sector region STATE (own rhombus-grid regions
+ one coarse (occupancy, median SINR) pair per OTHER sector in the
network) and a GLOBAL scalar REWARD (total covered-UE count across the
WHOLE network, the SAME value given to every sector) -- reward changed
from "sector_neighbors" (pooled over sector + its actual neighbor_ids) to
global 2026-09-20: the project's real objective is overall network
coverage, not a per-sector/local proxy for it, so the reward should say
so directly rather than approximate it locally. neighbor_ids is no
longer used for reward at all now (STATE never used it either -- see
below); self.neighbor_sets is kept only because other callers, e.g.
scripts/debug_pipeline_audit.py's own sector-panel visualization, still
read it.

One measurement per tilt-control interval (helpers/simulation_engine.py):
accumulate() is called once per "turn" -- once for the whole network
(async_schedule="sync") or once per sector in sequence within the same
interval (Gauss-Seidel, "async"). own_occupancy/all_occupancy are this
turn's raw instantaneous value (no smoothing across turns -- EMA smoothing
was removed 2026-09-20: at this codebase's occupancy_ema_alpha=0.5, its
effective sample size was ~3 intervals of a single SECTOR's own ~25 UEs,
not the large pooled samples that make an average genuinely robust --
smoothing bought little and only delayed a real signal). compute()'s
`closes` mask picks which sector(s)' reward actually persists this call;
every sector's raw per-turn counters reset unconditionally at the end
regardless, since the next accumulate() always starts an entirely fresh
turn.

STATE geometry: a sector's own rhombus grid (grid_n x grid_n cells, its
own oblique basis) partitions into the near/mid/far hotspot regions --
the ONE region definition in this codebase (see
CellularTopology.hotspot_region_membership), same rule OccupancyEstimator
uses. A UE's region comes directly from assign_to_sector_rhombus_grid's
cell_idx via OccupancyEstimator's own cell_to_region lookup -- no geometry
search needed. Unlike the earlier equal-size Cartesian-block scheme, the
3 regions do NOT all contain the same number of fine cells.

own_occupancy (per own-region) and all_occupancy (whole-sector, feeding a
neighbor's own coarse pair) are RAW UE COUNTS (2026-09-21 -- previously
each a SHARE of the whole network's visits this turn; changed per
explicit request -- a region/sector with more UEs in it now reads as a
literally bigger number, not a fraction of a shifting network-wide
total). own_coverage/all_coverage were removed entirely the same day (not
replaced by anything on the OWN side; all_coverage's own role feeding a
neighbor's state is replaced by all_sinr_p50, see below) -- own_occupancy/
all_occupancy are the ONLY two quantities this class still computes from
raw visit/covered counts; everything else covered-UE-related now lives
only in REWARD.

Every sector's own whole-grid coarse (occupancy, median SINR) pair is
computed here for EVERY sector (all_occupancy/all_sinr_p50, [num_bs]
arrays, independent of its own region breakdown) -- but which of those
[num_bs] entries a caller actually uses in its own state is decided
downstream, not here: drl.interface.compute_region_state restricts this
to just the 2 real cross-site neighbors (CellularTopology.neighbor_ids,
zero-padded) since 2026-09-20 (see that function's own docstring for the
empirical justification -- compute_single_sector_tilt_perturbation's
finding that everything past the 2 front neighbors carries near-noise-
level information, later confirmed by phase 7's own feature-ablation ΔR^2
test). This class itself stays agnostic to that choice -- it always
computes the full [num_bs] pair, same as before.

STATE also includes own-region SINR PERCENTILES (p25/p50/p75, 2026-09-20
addition) -- see accumulate()'s own sinr_db param and compute()'s own
docstring. phase 7's own feature-ablation found these add only a small
ΔR^2 on top of occupancy+coverage for predicting a sector's own local
reward -- kept anyway per explicit user decision, not because the data
showed a strong case for them. all_sinr_p50 (2026-09-21 addition) is the
same idea at whole-sector scope (median, not the full p25/p50/p75 spread)
-- a neighbor's own coarse "how good is it there right now" signal,
replacing all_coverage in that role.

REWARD is the covered-UE count, normalized by num_ut (a coverage RATIO,
2026-09-20 -- see compute()'s own docstring), across the WHOLE network
this turn -- the same scalar value returned for every `closes`
sector, since the objective is overall network coverage, not any one
sector's own local slice of it.

own_occupancy/all_occupancy/all_sinr_p50 are ALWAYS computed directly from
real per-UE data. all_occupancy/all_sinr_p50/reward (whole-sector/network
scope) count every real UE, anchor or not. own_occupancy/own_sinr_* (the
own-region breakdown) for a non-anchor UE depend on config.yaml's own
non_anchor_region_handling (2026-09-20, see accumulate()'s
own_region_mask/region_idx_override params, both supported and
combinable): "exclude" (current default) -- a non-anchor UE simply
doesn't contribute to its own-region breakdown at all; "rme" --
SimulationEngine's own _build_region_idx,
argmax(SpatialGridEstimator.infer_region_probabilities(...)), a non-anchor
UE counts as a full unit in its OWN INFERRED region instead. "rme" is not
the default -- phase 5's own kriging_hard/soft, cgr_hard/soft situations
found it makes region estimates WORSE than plain exclusion (systematically
over-assigning UEs to the "far" region) as of the sigma_db calibration
this was tested with -- but the mechanism itself is intact and one config
line away, kept for after that calibration is redone.
Separately (also deleted 2026-09-20): a THIRD path (coverage_fill="kriged")
used to throw all of this away and substitute a whole-grid Kriged/CGR-predicted coverage map -- inventing a
coverage value even for grid cells with ZERO real UEs of any kind,
anchor or not. That was never anyone's intended design (RME's only real
job is inferring a non-anchor UE's own region, nothing about geography
with no UE present at all) -- removed entirely, along with
coverage_fill/radio_map_estimator's own config.yaml keys. state_ready is
therefore always True now (no Kriging-readiness gating left to check).
"""

import numpy as np

from helpers.occupancy_estimator import OccupancyEstimator
from helpers.utils import get_logger

logger = get_logger(__name__)


class RegionalInfoProvider:
    """Per-sector region-grid occupancy STATE (own regions + neighbor
    coarse pairs) plus a GLOBAL scalar REWARD.

    :ivar sector_visit_count, sector_covered_count: [num_bs] -- whole-grid
        (not blocked into regions) totals, per-turn reset -- feeds both
        all_occupancy (STATE) and the network-wide sum (REWARD).
        sector_covered_count is the only covered-UE bookkeeping left in
        this class (own_coverage/all_coverage were removed 2026-09-21) --
        it exists purely for REWARD.
    """

    def __init__(self, neighbor_ids: np.ndarray, grid_n: int, num_ut: int,
                near_mask: np.ndarray, mid_mask: np.ndarray):
        """
        :param neighbor_ids: [num_bs, max_neighbors] int, -1-padded --
            CellularTopology.neighbor_ids.
        :param grid_n: rhombus grid side length (cells/side).
        :param num_ut: total real UE count -- REWARD's own normalizer (see
            compute()'s own docstring); fixed for the whole run.
        :param near_mask, mid_mask: [grid_n, grid_n] bool, mutually
            exclusive -- the SAME calibrated radial near/mid partition
            CellularTopology.hotspot_region_membership uses (see
            SimulationEngine._calibrate_hotspot_region_lut), so
            OccupancyEstimator's own region definition can't silently
            diverge from wherever real UEs actually get placed (2026-10-12
            fix -- see OccupancyEstimator's own module docstring for the
            drift this replaces).
        """
        self.num_bs = len(neighbor_ids)
        self.neighbor_sets = [ids[ids >= 0].tolist() for ids in neighbor_ids]
        self.grid_n = grid_n
        self.num_ut = num_ut

        # STATE -- own regions: raw visit-count bookkeeping is shared with
        # the offline occupancy predictor's own data pipeline (see
        # helpers/occupancy_estimator.py); cell_to_region re-exposed here
        # since callers (e.g. scripts/debug_pipeline_audit.py) already
        # reach for it on this class.
        self._occupancy = OccupancyEstimator(self.num_bs, grid_n, near_mask, mid_mask)
        self._finish_init()

    @classmethod
    def from_cell_to_region(cls, neighbor_ids: np.ndarray, num_ut: int, cell_to_region: np.ndarray,
                            region_names: tuple) -> "RegionalInfoProvider":
        """Alternate constructor (2026-10-15, explicit request), mirrors
        OccupancyEstimator.from_cell_to_region's own rationale: for a
        state representation whose own partition isn't two (near_mask,
        mid_mask) boolean masks -- e.g. a polar grid. ADDITIVE -- __init__
        and every existing near/mid/far caller are untouched.

        :param cell_to_region: [grid_n*grid_n] int, already flattened --
            see OccupancyEstimator.from_cell_to_region's own docstring.
        :param region_names: per-bin names, sets self.region_names/
            num_own_regions for THIS instance (grid_n is derived from
            cell_to_region's own length here, not passed separately).
        """
        obj = cls.__new__(cls)
        obj.num_bs = len(neighbor_ids)
        obj.neighbor_sets = [ids[ids >= 0].tolist() for ids in neighbor_ids]
        obj.grid_n = int(round(len(cell_to_region) ** 0.5))
        obj.num_ut = num_ut
        obj._occupancy = OccupancyEstimator.from_cell_to_region(obj.num_bs, cell_to_region, region_names)
        obj._finish_init()
        return obj

    def _finish_init(self) -> None:
        """Shared tail of __init__/from_cell_to_region, once self._occupancy
        is set -- everything here is identical regardless of which
        partition self._occupancy itself uses.
        """
        self.cell_to_region = self._occupancy.cell_to_region
        self.num_own_regions = self._occupancy.num_own_regions
        self.region_names = self._occupancy.region_names
        # STATE -- own-region SINR percentiles (2026-09-20): unlike a
        # simple running count, a percentile needs the raw per-UE values
        # kept until compute() time -- reset to empty lists there every
        # call, same "one turn" lifecycle as everything else in this class.
        self._sinr_values = [[[] for _ in range(self.num_own_regions)] for _ in range(self.num_bs)]
        # STATE -- FAR-REGION-ONLY SINR pool (2026-10-23, was whole-sector),
        # feeding a neighbor's own coarse median-SINR feature -- same "every
        # real UE, anchor or not" inclusion rule as sector_visit_count_far/
        # sector_covered_count below (own_resolved-gated _sinr_values above
        # is a DIFFERENT, own-region-only pool), but now ALSO filtered to
        # just the far region (same near/mid/far cut as own_occupancy's own
        # "far", via self.cell_to_region) so a neighbor's own INTERIOR UEs
        # don't drown out the boundary-relevant ones that actually matter
        # for cross-sector interference -- explicit user decision, simpler
        # than a separate edge_percentile/distance mechanism since it
        # reuses the SAME region classification already computed for the
        # own-region features.
        self._sector_sinr_values = [[] for _ in range(self.num_bs)]
        # None for a partition with no literal "far" bin (e.g. the polar
        # grid's "far_a0"/"far_a1"/... -- see from_cell_to_region) -- that
        # instance's own all_occupancy/all_sinr_p50 are unused duplicates
        # anyway (SimulationEngine._measure_and_pool only keeps its first 4
        # compute() outputs), so accumulate() below just falls back to the
        # old whole-sector behavior rather than crashing on .index("far").
        self._far_region_idx = self.region_names.index("far") if "far" in self.region_names else None

        # STATE -- far-region-only totals, feeding neighbors' own coarse
        # pair (all_occupancy/all_sinr_p50). sector_visit_count (below,
        # UNFILTERED) is kept separately -- it no longer feeds all_occupancy,
        # only compute()'s own "was there any data this turn" reward gate.
        self.sector_visit_count_far = np.zeros(self.num_bs)
        self.sector_visit_count = np.zeros(self.num_bs)
        self.sector_covered_count = np.zeros(self.num_bs)
        # REWARD needs no separate state -- it's derived directly from
        # sector_covered_count (already tracked for STATE's all_occupancy)
        # at compute() time.

    def accumulate(self, sector_idx_geo: np.ndarray, cell_idx: np.ndarray, covered_ue: np.ndarray,
                  sinr_db: np.ndarray, own_region_mask: np.ndarray = None,
                  region_idx_override: np.ndarray = None) -> None:
        """One turn's measurement -- exactly one call, immediately followed
        by compute(), per turn (see module docstring).

        :param sector_idx_geo: [num_ue] geographic sector membership, from
            CellularTopology.assign_to_sector_rhombus_grid -- -1 excluded.
            This is ALWAYS the UE's real serving sector, known regardless
            of anchor status (which sector serves a UE is ordinary
            connectivity, not the precise position that may be unknown) --
            so whole-sector totals and reward below are NEVER affected by
            own_region_mask/region_idx_override, only the own-region
            breakdown is.
        :param cell_idx: [num_ue] this UE's own fine-grid cell index WITHIN
            its own geographic sector's grid, same source. Ignored where
            region_idx_override is given.
        :param covered_ue: [num_ue] bool -- every real UE's own real,
            known coverage status regardless of anchor status (SINR/
            coverage never depends on knowing location).
        :param sinr_db: [num_ue] float -- every real UE's own real measured
            SINR, for own-region p25/p50/p75 STATE (2026-09-20 addition --
            see compute()'s own docstring).
        :param own_region_mask, region_idx_override: see
            OccupancyEstimator.accumulate's own docstring; forwarded as-is
            to it, and used the same way for own-region SINR below.
            config.yaml's own non_anchor_region_handling picks which of
            these SimulationEngine actually passes: "exclude" (default)
            uses own_region_mask=location_reporting_mask, a non-anchor UE
            contributes NOTHING to the own-region breakdown; "rme" uses
            region_idx_override instead (SimulationEngine's own
            _build_region_idx, RME/Kriging-inferred), so a non-anchor UE
            counts as a full unit in its INFERRED region. Both None
            (default here) reduces to every real UE counting in its own
            real region everywhere -- ground truth.
        """
        logger.function("RegionalInfoProvider.accumulate start")
        resolved = sector_idx_geo >= 0
        own_resolved = resolved if own_region_mask is None else (resolved & own_region_mask)

        # STATE -- own regions. Every counted UE is exactly one full unit
        # in exactly one region -- no fractional splitting either here or
        # in self._occupancy.
        self._occupancy.accumulate(sector_idx_geo, cell_idx, own_region_mask, region_idx_override)
        if region_idx_override is None:
            region_idx = self.cell_to_region[cell_idx[own_resolved]]
        else:
            region_idx = region_idx_override[own_resolved]

        # STATE -- own-region SINR raw values, pooled per (sector, region)
        # until compute() turns them into p25/p50/p75 -- can't be an
        # incremental running sum like occupancy, a percentile needs the
        # raw distribution.
        sec_own = sector_idx_geo[own_resolved]
        sinr_own = sinr_db[own_resolved]
        for s in np.unique(sec_own):
            sec_mask = sec_own == s
            for r in range(self.num_own_regions):
                vals = sinr_own[sec_mask & (region_idx == r)]
                if vals.size > 0:
                    self._sinr_values[s][r].extend(vals.tolist())

        # STATE -- whole-sector totals (for neighbors' own coarse pair) --
        # ALWAYS every real UE, regardless of anchor status (serving
        # sector is ordinary connectivity, never actually unknown).
        np.add.at(self.sector_visit_count, sector_idx_geo[resolved], 1.0)
        np.add.at(self.sector_covered_count, sector_idx_geo[resolved], covered_ue[resolved].astype(float))
        sec_all = sector_idx_geo[resolved]
        sinr_all = sinr_db[resolved]

        # STATE -- all_occupancy/all_sinr_p50 (the neighbor-of-a-neighbor's
        # own coarse pair): FAR-region-only, same cell_to_region classification
        # as own_occupancy's own "far". UNLIKE sector_visit_count/
        # _covered_count above (anchor-independent -- "which sector serves
        # me" needs no location), knowing a UE sits specifically in the FAR
        # ring needs its actual position -- exactly what anchor status
        # gates everywhere else. So this reuses own_resolved/region_idx
        # (already computed above for own_occupancy) instead of the
        # anchor-independent `resolved`/`sec_all` -- a non-anchor UE's own
        # region is genuinely unknown here too, not just for its own
        # sector's own_occupancy (fixed 2026-10-23: the first version of
        # this used `resolved`, silently pretending non-anchor UEs' regions
        # were known, which made neighbor0/1_occupancy identical regardless
        # of anchor fraction -- a real bug, not a coincidence). Falls back
        # to every resolved UE (old whole-sector behavior, still
        # anchor-independent since no region check is needed) when this
        # instance has no literal "far" bin (see _far_region_idx's own
        # comment) -- that output is unused there anyway.
        if self._far_region_idx is not None:
            far_mask = region_idx == self._far_region_idx
            sec_far, sinr_far = sec_own[far_mask], sinr_own[far_mask]
        else:
            sec_far, sinr_far = sec_all, sinr_all
        np.add.at(self.sector_visit_count_far, sec_far, 1.0)
        for s in np.unique(sec_far):
            vals = sinr_far[sec_far == s]
            if vals.size > 0:
                self._sector_sinr_values[s].extend(vals.tolist())
        logger.function("RegionalInfoProvider.accumulate end")

    def compute(self, closes: np.ndarray) -> tuple:
        """Resolves the just-accumulated turn for every sector, but only
        PERSISTS (into reward) the sectors in `closes` -- every sector's
        raw per-turn counters are reset regardless, since the next
        accumulate() call always starts a fresh turn (see module
        docstring). own_occupancy/all_occupancy are NOT persisted at all
        -- every sector gets its own fresh instantaneous value every call.

        :param closes: [num_bs] bool -- which sector(s)' decision this turn
            actually applies to. All-True for async_schedule="sync" (one
            turn covers the whole network); exactly one True for "async"
            (each of the 21 sequential per-sector turns).
        :output: (own_occupancy [num_bs, num_own_regions] -- RAW UE count
            per own region (2026-09-21, no longer network-normalized),
            own_sinr_p25/p50/p75 [num_bs, num_own_regions] -- 2026-09-20
            addition, own-region SINR percentiles from this turn's raw
            per-UE values (see accumulate()'s own sinr_db param); 0.0 for
            a region with zero UEs this turn,
            all_occupancy [num_bs] -- RAW UE count per whole sector
            (2026-09-21, same change), all_sinr_p50 [num_bs] -- whole-
            sector median SINR (2026-09-21 addition, replaces all_coverage
            in feeding a neighbor's own coarse pair) -- the SAME [num_bs]
            array regardless of which sector is asking (a caller excludes
            its own index -- see drl.interface.compute_region_state),
            reward [num_bs], state_ready [num_bs] bool).
            own_occupancy/all_occupancy/all_sinr_p50 are this turn's raw
            instantaneous value for EVERY sector (no persistence, no
            `closes` gating -- see module docstring). Only `closes`
            sectors' own reward entry is meaningful to a caller -- every
            other sector's entry here reflects whatever THIS turn's
            snapshot happened to look like for it, not a real decision
            point. reward is the covered-UE count / num_ut (a coverage
            RATIO, not a raw count -- 2026-09-20) across the WHOLE network
            this turn (the SAME value for every closing sector, NaN if
            literally nobody was visited anywhere this turn). state_ready
            is always True (no Kriging-readiness gating -- see module
            docstring).
        """
        logger.function("RegionalInfoProvider.compute start: closes=%d", int(closes.sum()))
        # --- Own-region occupancy: this turn's RAW per-region UE count
        # (2026-09-21 -- no longer a share of the network's total visits;
        # no smoothing across turns either -- see module docstring).
        own_occupancy = self._occupancy.compute()

        # --- Whole-sector occupancy (feeds neighbors' coarse pair), same raw
        # convention -- FAR-REGION-ONLY now (see accumulate()'s own
        # comment); sector_visit_count itself (unfiltered) stays reserved
        # for the reward "any data this turn" gate below, not this.
        all_occupancy = self.sector_visit_count_far.copy()

        # --- Own-region SINR percentiles (2026-09-20 addition) -- from the
        # raw per-UE values pooled in accumulate(); 0.0 for a region with
        # no UEs this turn.
        own_sinr_p25 = np.zeros((self.num_bs, self.num_own_regions))
        own_sinr_p50 = np.zeros((self.num_bs, self.num_own_regions))
        own_sinr_p75 = np.zeros((self.num_bs, self.num_own_regions))
        for s in range(self.num_bs):
            for r in range(self.num_own_regions):
                vals = self._sinr_values[s][r]
                if vals:
                    own_sinr_p25[s, r] = np.percentile(vals, 25)
                    own_sinr_p50[s, r] = np.percentile(vals, 50)
                    own_sinr_p75[s, r] = np.percentile(vals, 75)
        self._sinr_values = [[[] for _ in range(self.num_own_regions)] for _ in range(self.num_bs)]

        # --- Whole-sector median SINR (2026-09-21 addition) -- from the
        # raw per-UE values pooled in accumulate() (every real UE,
        # regardless of anchor status -- see accumulate()'s own docstring);
        # 0.0 for an unvisited sector, same convention as everything else here.
        all_sinr_p50 = np.zeros(self.num_bs)
        for s in range(self.num_bs):
            vals = self._sector_sinr_values[s]
            if vals:
                all_sinr_p50[s] = np.percentile(vals, 50)
        self._sector_sinr_values = [[] for _ in range(self.num_bs)]

        state_ready_out = np.ones(self.num_bs, dtype=bool)

        # --- REWARD -- GLOBAL: covered-UE COUNT normalized by num_ut (a
        # coverage RATIO, not a raw count -- changed 2026-09-20: since
        # num_ut is fixed for the whole run, this is just a constant
        # rescale of the same signal, so it doesn't change what policy
        # gets learned, only the numeric scale -- done for the network's
        # own gradient/loss-scale hygiene and log readability, see
        # drl.interface.compute_reward's own docstring) across the WHOLE
        # network this turn, the SAME scalar for every `closes` sector
        # (see module docstring for why this replaced the earlier
        # per-sector X_s-scoped reward).
        reward = np.full(self.num_bs, np.nan)
        if self.sector_visit_count.sum() > 0:
            reward[closes] = self.sector_covered_count.sum() / self.num_ut

        # Every sector's raw per-turn counters reset regardless of `closes`
        # -- the next accumulate() call always starts an entirely fresh
        # turn (see module docstring); nothing here is meant to survive
        # across turns.
        # (self._occupancy's own visit_count already reset by compute() above.)
        self.sector_visit_count[:] = 0.0
        self.sector_visit_count_far[:] = 0.0
        self.sector_covered_count[:] = 0.0

        logger.function("RegionalInfoProvider.compute end")
        return (own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75,
               all_occupancy, all_sinr_p50, reward, state_ready_out)
