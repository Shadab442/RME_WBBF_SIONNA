"""OccupancyEstimator: per-sector per-region UE-occupancy
Decides how many UEs are present in each geographical region.

Region geometry: the SAME near/mid/far partition as
CellularTopology.hotspot_region_membership, over a sector's own rhombus
grid (grid_n x grid_n cells) -- passed in directly (near_mask/mid_mask,
the SAME calibrated arrays SimulationEngine._calibrate_hotspot_region_lut
installs into CellularTopology), not recomputed here. (2026-10-12: this
class used to compute its OWN, unrelated diagonal-wedge near/mid/far rule
-- the one hotspot_region_membership's own docstring documents as "tried
and rejected" on 2026-09-20 -- silently left behind when
hotspot_region_membership itself moved to the current radial rule on
2026-09-29. Confirmed via region_occupancy_validation.png: ~80%+ of real
UEs landed in "far" under every demand state, including near-heavy, since
most of a sector's area at ANY radial distance sits off the old rule's
narrow diagonal band.)
"""

import numpy as np


class OccupancyEstimator:
    """
    precomputes, which of every sector's fine grid cells count as near/mid/far.

    :ivar visit_count: [num_bs, num_own_regions] -- this interval's raw
        visit counts, reset to 0 every compute() call.
    """
    region_names = ("near", "mid", "far")

    def __init__(self, num_bs: int, grid_n: int, near_mask: np.ndarray, mid_mask: np.ndarray):
        """
        :param num_bs: number of sectors.
        :param grid_n: rhombus grid side length (cells/side).
        :param near_mask, mid_mask: [grid_n, grid_n] bool, mutually
            exclusive -- see this module's own docstring; both False means
            far. Flattened in the SAME row-major (i=idx//grid_n,
            j=idx%grid_n) order cell_idx itself uses.
        """
        self.num_bs = num_bs
        self.num_own_regions = len(self.region_names)

        assert near_mask.shape == (grid_n, grid_n) and mid_mask.shape == (grid_n, grid_n), \
            f"near_mask/mid_mask must be ({grid_n},{grid_n}) to match grid_n"
        assert not (near_mask & mid_mask).any(), "near_mask/mid_mask must be mutually exclusive"
        near_flat, mid_flat = near_mask.reshape(-1), mid_mask.reshape(-1)
        self.cell_to_region = np.where(near_flat, 0, np.where(mid_flat, 1, 2))

        self.visit_count = np.zeros((num_bs, self.num_own_regions))

    @classmethod
    def from_cell_to_region(cls, num_bs: int, cell_to_region: np.ndarray, region_names: tuple) -> "OccupancyEstimator":
        """Alternate constructor (2026-10-15, explicit request) for a
        state representation whose own partition ISN'T expressible as two
        (near_mask, mid_mask) boolean masks -- e.g. a polar grid (radial x
        azimuthal bins). Bypasses __init__ entirely (ADDITIVE -- every
        existing caller's own near_mask/mid_mask construction is untouched)
        so a precomputed, already-flattened cell_to_region lookup (any
        partition, any bin count) can be used directly.

        :param cell_to_region: [grid_n*grid_n] int, already flattened in
            the SAME row-major order cell_idx itself uses -- values
            0..len(region_names)-1.
        :param region_names: names for each bin, e.g. ("near","mid","far")
            for 3, or a polar grid's own per-bin labels for more -- sets
            self.region_names as an INSTANCE attribute, shadowing the
            class-level ("near","mid","far") default for THIS instance
            only; every normally-constructed instance is unaffected.
        """
        obj = cls.__new__(cls)
        obj.num_bs = num_bs
        obj.region_names = tuple(region_names)
        obj.num_own_regions = len(obj.region_names)
        obj.cell_to_region = np.asarray(cell_to_region, dtype=int)
        obj.visit_count = np.zeros((num_bs, obj.num_own_regions))
        return obj

    def accumulate(self, sector_idx_geo: np.ndarray, cell_idx: np.ndarray,
                  own_region_mask: np.ndarray = None, region_idx_override: np.ndarray = None) -> None:
        """Decide which UE falls in which region. Every UE counts as one
        full unit in exactly one region -- no fractional/probabilistic
        splitting. Two independent, combinable knobs for a non-anchor UE
        (whose real region isn't known -- see RegionalInfoProvider.
        accumulate and config.yaml's own non_anchor_region_handling):
        own_region_mask=False EXCLUDES it entirely (current production
        default, "exclude"); region_idx_override REASSIGNS it to a given
        region instead, e.g. an RME/Kriging-inferred one ("rme" mode,
        SimulationEngine's own _build_region_idx) -- kept switchable via
        config since it was found to make region estimates WORSE than
        plain exclusion at the time this was tested (see
        config.yaml's own non_anchor_region_handling comment), not because
        the mechanism itself is broken.

        :param sector_idx_geo: [num_ue] geographic sector membership, from
            CellularTopology.assign_to_sector_rhombus_grid.
        :param cell_idx: [num_ue] UE's own fine-grid cell index WITHIN
            its own geographic sector's grid, same source. Ignored where
            region_idx_override is given.
        :param own_region_mask: [num_ue] bool, optional -- True for a UE
            that counts toward the own-region breakdown at all (typically
            the anchor/location-reporting mask); False excludes it
            entirely from this accumulation. None (default) reduces to
            every real UE counting.
        :param region_idx_override: [num_ue] int, optional -- per-UE
            region index REPLACING the cell_idx-derived one, for whichever
            UEs pass own_region_mask. None (default) reduces to the plain
            cell_idx-derived region for every counted UE.
        """
        resolved = sector_idx_geo >= 0
        if own_region_mask is not None:
            resolved = resolved & own_region_mask
        if region_idx_override is None:
            region_idx = self.cell_to_region[cell_idx[resolved]]
        else:
            region_idx = region_idx_override[resolved]
        np.add.at(self.visit_count, (sector_idx_geo[resolved], region_idx), 1.0)

    def compute(self) -> np.ndarray:
        """This interval's own-region occupancy: each region's RAW visit
        count (2026-09-21 -- previously a share of the whole network's
        visits; changed to a raw UE count per explicit request).

        :output: own_occupancy [num_bs, num_own_regions].
        """
        own_occupancy = self.visit_count.copy()
        self.visit_count[:] = 0.0
        return own_occupancy
