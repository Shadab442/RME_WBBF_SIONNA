"""Environment-side RL interface: state/reward construction and action
interpretation -- the contract between the environment
(helpers/simulation_engine.py) and any policy (drl/factory.py).
"""

import logging

import numpy as np

from helpers.utils import get_logger

logger = get_logger(__name__)

_REGION_STATE_TYPES = ("region_aligned",)
_CELL_STATE_TYPES = ("cell_aligned",)
_POLAR_REGIONAL_NEIGHBOR_STATE_TYPES = ("polar_region_neighbor",)
_GLOBAL_REWARD_TYPES = ("global",)
_DELTA_REWARD_TYPES = ("delta_covered",)


_NUM_NEIGHBOR_SLOTS = 2

# SINR percentiles computed by RegionalInfoProvider.compute() (own_sinr_p25/
# p50/p75) -- hardcoded there (np.percentile(vals, 25/50/75), three separate
# calls, not a loop over a list) since they're a fixed part of the reward/
# state contract, not a tunable. Re-declared here as the single source of
# truth for LABELING those columns (state_feature_names below) so a future
# change to one side is at least forced to touch this constant too.
_SINR_PERCENTILES = (25, 50, 75)


def compute_region_state(own_tilt_norm, own_occupancy,
                         own_sinr_p25, own_sinr_p50, own_sinr_p75,
                         all_occupancy, all_sinr_p50, neighbor_ids):
    """o_i[k] = [own_occupancy, own SINR p25/p50/p75 (each num_own_regions,
    FIXED/identical layout for every sector), own tilt, the 2 REAL
    cross-site front neighbors' own coarse occupancy, coarse median SINR,
    tilt] -- FIXED width for every sector, using exactly
    _NUM_NEIGHBOR_SLOTS=2 neighbor slots, not every other sector (see
    helpers/regional_info_provider.py's own module docstring for the
    2-neighbor justification). SINR p25/p50/p75 kept per explicit user
    decision. own_coverage, all_coverage (replaced by all_sinr_p50 for the
    neighbor role) and already_moved were all removed 2026-09-21 per
    explicit user decision; neighbor tilt added the same day, reusing
    own_tilt_norm (already [num_sectors]) rather than a new provider
    output.

    :param own_tilt_norm: [num_sectors] float in [0,1].
    :param own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75:
        [num_sectors, num_own_regions] arrays, from RegionalInfoProvider.
        compute() -- own_occupancy is a RAW UE count (not normalized).
    :param all_occupancy, all_sinr_p50: [num_sectors] arrays, same source
        -- every sector's own whole-grid coarse aggregate (all_occupancy a
        RAW UE count, same convention as own_occupancy), identical
        regardless of which sector is asking; each sector's own row here
        is subset down to its own 2 real neighbors via neighbor_ids.
    :param neighbor_ids: [num_sectors, max_neighbors] int, -1-padded --
        CellularTopology.neighbor_ids. Only the first _NUM_NEIGHBOR_SLOTS
        real (>=0) entries per sector are used; a sector with fewer real
        neighbors (see the "N sectors with ZERO neighbors" warning --
        CellularTopology's own outer-ring sectors) gets zero-padded
        occupancy/SINR/tilt for the missing slot(s), not a crash or a
        repeated neighbor.
    :output: [num_sectors, 4*num_own_regions + 1 + 3*_NUM_NEIGHBOR_SLOTS] array.
    """
    logger.function("compute_region_state start")
    num_sectors = len(own_tilt_norm)
    num_own_regions = own_occupancy.shape[1]
    width = 4 * num_own_regions + 1 + 3 * _NUM_NEIGHBOR_SLOTS
    result = np.zeros((num_sectors, width))
    result[:, 0 * num_own_regions:1 * num_own_regions] = own_occupancy
    result[:, 1 * num_own_regions:2 * num_own_regions] = own_sinr_p25
    result[:, 2 * num_own_regions:3 * num_own_regions] = own_sinr_p50
    result[:, 3 * num_own_regions:4 * num_own_regions] = own_sinr_p75
    result[:, 4 * num_own_regions] = own_tilt_norm
    base = 4 * num_own_regions + 1
    for s in range(num_sectors):
        nb = neighbor_ids[s]
        nb = nb[nb >= 0][:_NUM_NEIGHBOR_SLOTS]
        nb_valid = np.zeros(_NUM_NEIGHBOR_SLOTS, dtype=bool)
        nb_valid[:len(nb)] = True
        nb_safe = np.zeros(_NUM_NEIGHBOR_SLOTS, dtype=int)
        nb_safe[:len(nb)] = nb
        result[s, base:base + _NUM_NEIGHBOR_SLOTS] = np.where(nb_valid, all_occupancy[nb_safe], 0.0)
        result[s, base + _NUM_NEIGHBOR_SLOTS:base + 2 * _NUM_NEIGHBOR_SLOTS] = np.where(
            nb_valid, all_sinr_p50[nb_safe], 0.0)
        result[s, base + 2 * _NUM_NEIGHBOR_SLOTS:base + 3 * _NUM_NEIGHBOR_SLOTS] = np.where(
            nb_valid, own_tilt_norm[nb_safe], 0.0)
    logger.debug("compute_region_state: shape=%s", result.shape)
    logger.function("compute_region_state end")
    return result


def compute_cell_state(own_tilt_norm, own_occupancy, own_sinr_p50, all_occupancy, all_sinr_p50, neighbor_ids):
    """o_i[k] = [own per-CELL occupancy, own per-cell median SINR (each
    num_valid_cells, FIXED/identical per-sector layout -- a sector's own
    geometry never changes mid-run), own tilt, the 2 REAL cross-site front
    neighbors' own coarse occupancy, coarse median SINR] -- FIXED width
    for every sector, exactly _NUM_NEIGHBOR_SLOTS=2 neighbor slots (same
    2-neighbor justification as compute_region_state).

    2026-09-22 addition (state_type="cell_aligned"), alongside -- not
    replacing -- compute_region_state's own region-pooled design. See
    helpers/cell_aligned_info_provider.py's own module docstring for the
    motivation: region-pooling conflates cells with OPPOSITE tilt-
    sensitivity (a near-BS boundary cell that gains coverage under
    downtilt and a far-from-BS boundary cell that loses it under the SAME
    downtilt can land in the same region), which per-cell state avoids by
    construction. Unlike compute_region_state, the neighbor block here is
    only 2 stats (occupancy, median SINR) -- no neighbor tilt -- per
    explicit spec.

    :param own_tilt_norm: [num_sectors] float in [0,1].
    :param own_occupancy, own_sinr_p50: [num_sectors, num_valid_cells]
        arrays, from CellAlignedInfoProvider.compute() -- own_occupancy is
        a RAW UE count (not normalized), own_sinr_p50 is that cell's
        median SINR this turn (0.0 if no UE visited it this turn).
    :param all_occupancy, all_sinr_p50: [num_sectors] arrays, same source
        -- every sector's own whole-grid coarse aggregate (same convention
        as compute_region_state's own all_occupancy/all_sinr_p50);
        identical regardless of which sector is asking, subset down to
        each sector's own 2 real neighbors via neighbor_ids.
    :param neighbor_ids: [num_sectors, max_neighbors] int, -1-padded --
        CellularTopology.neighbor_ids. Only the first _NUM_NEIGHBOR_SLOTS
        real (>=0) entries per sector are used; a sector with fewer real
        neighbors gets zero-padded occupancy/sinr_p50 for the missing
        slot(s), not a crash or a repeated neighbor.
    :output: [num_sectors, 2*num_valid_cells + 1 + 2*_NUM_NEIGHBOR_SLOTS] array.
    """
    logger.function("compute_cell_state start")
    num_sectors = len(own_tilt_norm)
    num_valid_cells = own_occupancy.shape[1]
    width = 2 * num_valid_cells + 1 + 2 * _NUM_NEIGHBOR_SLOTS
    result = np.zeros((num_sectors, width))
    result[:, 0:num_valid_cells] = own_occupancy
    result[:, num_valid_cells:2 * num_valid_cells] = own_sinr_p50
    result[:, 2 * num_valid_cells] = own_tilt_norm
    base = 2 * num_valid_cells + 1
    for s in range(num_sectors):
        nb = neighbor_ids[s]
        nb = nb[nb >= 0][:_NUM_NEIGHBOR_SLOTS]
        nb_valid = np.zeros(_NUM_NEIGHBOR_SLOTS, dtype=bool)
        nb_valid[:len(nb)] = True
        nb_safe = np.zeros(_NUM_NEIGHBOR_SLOTS, dtype=int)
        nb_safe[:len(nb)] = nb
        result[s, base:base + _NUM_NEIGHBOR_SLOTS] = np.where(nb_valid, all_occupancy[nb_safe], 0.0)
        result[s, base + _NUM_NEIGHBOR_SLOTS:base + 2 * _NUM_NEIGHBOR_SLOTS] = np.where(
            nb_valid, all_sinr_p50[nb_safe], 0.0)
    logger.debug("compute_cell_state: shape=%s", result.shape)
    logger.function("compute_cell_state end")
    return result


def compute_polar_regional_neighbor_state(own_tilt_norm, own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75,
                                          neighbor_region_occupancy, neighbor_region_sinr_p50, neighbor_ids):
    """o_i[k] = [own polar-grid occupancy, own SINR p25/p50/p75 (each
    num_polar_bins), own tilt, the 2 REAL cross-site front neighbors' OWN
    REGION-level (near/mid/far) breakdown: occupancy[3] + sinr_p50[3] +
    tilt[1] = 7 features per neighbor slot] -- state_type=
    "polar_region_neighbor" (2026-10-15, explicit request): own-sector
    state is polar-binned (SimulationEngine.polar_info_provider), but each
    neighbor's own state is exposed at REGION resolution (not collapsed to
    one whole-sector scalar like compute_region_state's own neighbor
    block) -- the neighbor depth matches a sector's OWN region-level
    detail (occupancy+sinr_p50 only, no p25/p75 -- same convention
    compute_cell_state's own neighbor block already uses, deliberately
    less detail than a sector's own features).

    :param own_tilt_norm: [num_sectors] float in [0,1].
    :param own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75:
        [num_sectors, num_polar_bins] arrays, from SimulationEngine.
        polar_info_provider.compute() -- own_occupancy is a RAW UE count.
    :param neighbor_region_occupancy, neighbor_region_sinr_p50:
        [num_sectors, num_own_regions] arrays -- EVERY sector's OWN
        near/mid/far breakdown, from the SAME RegionalInfoProvider every
        state_type already builds for reward (SimulationEngine.
        regional_info_provider's own own_occupancy/own_sinr_p50) --
        re-purposed here to feed a DIFFERENT sector's NEIGHBOR features,
        subset per sector down to its own 2 real neighbors via
        neighbor_ids, same mechanism compute_region_state's own
        all_occupancy/all_sinr_p50 neighbor lookup uses.
    :param neighbor_ids: [num_sectors, max_neighbors] int, -1-padded --
        CellularTopology.neighbor_ids, same convention as every other
        state builder in this file.
    :output: [num_sectors, 4*num_polar_bins + 1 + 7*_NUM_NEIGHBOR_SLOTS] array.
    """
    logger.function("compute_polar_regional_neighbor_state start")
    num_sectors = len(own_tilt_norm)
    num_polar_bins = own_occupancy.shape[1]
    num_own_regions = neighbor_region_occupancy.shape[1]
    per_neighbor_width = 2 * num_own_regions + 1  # occupancy[3] + sinr_p50[3] + tilt[1]
    width = 4 * num_polar_bins + 1 + per_neighbor_width * _NUM_NEIGHBOR_SLOTS
    result = np.zeros((num_sectors, width))
    result[:, 0 * num_polar_bins:1 * num_polar_bins] = own_occupancy
    result[:, 1 * num_polar_bins:2 * num_polar_bins] = own_sinr_p25
    result[:, 2 * num_polar_bins:3 * num_polar_bins] = own_sinr_p50
    result[:, 3 * num_polar_bins:4 * num_polar_bins] = own_sinr_p75
    result[:, 4 * num_polar_bins] = own_tilt_norm
    base = 4 * num_polar_bins + 1
    for s in range(num_sectors):
        nb = neighbor_ids[s]
        nb = nb[nb >= 0][:_NUM_NEIGHBOR_SLOTS]
        nb_valid = np.zeros(_NUM_NEIGHBOR_SLOTS, dtype=bool)
        nb_valid[:len(nb)] = True
        nb_safe = np.zeros(_NUM_NEIGHBOR_SLOTS, dtype=int)
        nb_safe[:len(nb)] = nb
        for i in range(_NUM_NEIGHBOR_SLOTS):
            off = base + i * per_neighbor_width
            if not nb_valid[i]:
                continue
            result[s, off:off + num_own_regions] = neighbor_region_occupancy[nb_safe[i]]
            result[s, off + num_own_regions:off + 2 * num_own_regions] = neighbor_region_sinr_p50[nb_safe[i]]
            result[s, off + 2 * num_own_regions] = own_tilt_norm[nb_safe[i]]
    logger.debug("compute_polar_regional_neighbor_state: shape=%s", result.shape)
    logger.function("compute_polar_regional_neighbor_state end")
    return result


def polar_regional_neighbor_state_feature_names(num_polar_bins, polar_bin_names, num_own_regions, region_names):
    """Index -> semantic feature name, in EXACTLY the column order
    compute_polar_regional_neighbor_state builds -- same role as
    state_feature_names/cell_state_feature_names, for state_type=
    "polar_region_neighbor".
    """
    if len(polar_bin_names) != num_polar_bins:
        raise ValueError(f"polar_bin_names length {len(polar_bin_names)} != num_polar_bins {num_polar_bins}")
    if len(region_names) != num_own_regions:
        raise ValueError(f"region_names length {len(region_names)} != num_own_regions {num_own_regions}")
    names = [f"own_occupancy_{b}" for b in polar_bin_names]
    for p in _SINR_PERCENTILES:
        names += [f"own_sinr_p{p}_{b}" for b in polar_bin_names]
    names.append("own_tilt_norm")
    for i in range(_NUM_NEIGHBOR_SLOTS):
        names += [f"neighbor{i}_occupancy_{r}" for r in region_names]
        names += [f"neighbor{i}_sinr_p50_{r}" for r in region_names]
        names.append(f"neighbor{i}_tilt_norm")
    return names


def cell_state_feature_names(num_valid_cells):
    """Index -> semantic feature name, in EXACTLY the column order
    compute_cell_state builds (own occupancy/sinr_p50 per cell, own tilt,
    then per neighbor slot: occupancy/sinr_p50) -- same role as
    state_feature_names, for the cell_aligned state_type.

    :param num_valid_cells: e.g. CellAlignedInfoProvider.num_valid_cells.
    :output: list[str], length == num_features_for("cell_aligned",
        num_valid_cells=num_valid_cells).
    """
    names = [f"own_occupancy_cell{c}" for c in range(num_valid_cells)]
    names += [f"own_sinr_p50_cell{c}" for c in range(num_valid_cells)]
    names.append("own_tilt_norm")
    for stat in ("occupancy", "sinr_p50"):
        names += [f"neighbor{i}_{stat}" for i in range(_NUM_NEIGHBOR_SLOTS)]
    return names


def state_feature_names(num_own_regions, region_names):
    """Index -> semantic feature name, in EXACTLY the column order
    compute_region_state builds (own occ/p25/p50/p75 per region, own
    tilt, then per neighbor slot: occupancy/sinr_p50/tilt) -- the single
    source of truth for diagnostics labeling (helpers/drl_diagnostics.py),
    so it can never drift from the real state layout.

    :param num_own_regions: region count, e.g. RegionalInfoProvider.
        num_own_regions.
    :param region_names: [num_own_regions] names, e.g.
        RegionalInfoProvider.region_names ("near", "mid", "far").
    :output: list[str], length == num_features_for("region_aligned",
        num_own_regions).
    """
    if len(region_names) != num_own_regions:
        raise ValueError(
            f"region_names length {len(region_names)} != num_own_regions {num_own_regions}")
    names = [f"own_occupancy_{r}" for r in region_names]
    for p in _SINR_PERCENTILES:
        names += [f"own_sinr_p{p}_{r}" for r in region_names]
    names.append("own_tilt_norm")
    for stat in ("occupancy", "sinr_p50", "tilt_norm"):
        names += [f"neighbor{i}_{stat}" for i in range(_NUM_NEIGHBOR_SLOTS)]
    return names


def num_features_for(state_type, num_own_regions=None, num_sectors=None, num_valid_cells=None,
                     num_polar_bins=None):
    """Feature count for a given state_type, so callers (e.g. sizing a
    policy's input layer) don't have to duplicate compute_region_state's/
    compute_cell_state's/compute_polar_regional_neighbor_state's own
    construction.

    :param num_sectors: UNUSED now (2026-09-20) -- state width is fixed at
        exactly _NUM_NEIGHBOR_SLOTS=2 neighbor slots regardless of network
        size, kept as a param for interface stability.
    :param num_valid_cells: required for state_type="cell_aligned" only --
        e.g. CellAlignedInfoProvider.num_valid_cells.
    :param num_polar_bins: required for state_type in ("polar_whole_neighbor",
        "polar_region_neighbor") -- e.g. SimulationEngine.num_polar_bins
        (2026-10-15 addition). "polar_whole_neighbor" reuses
        compute_region_state's own formula with num_polar_bins standing in
        for num_own_regions (its own "own" block is shape-generic, see
        compute_region_state's own docstring); "polar_region_neighbor" is
        its own formula (own polar block + REGION-depth neighbor block,
        see compute_polar_regional_neighbor_state's own docstring).
    :output: single int -- every sector shares the same fixed width.
    """
    logger.function("num_features_for start: state_type=%s", state_type)
    if state_type == "polar_whole_neighbor":
        num_features = 4 * num_polar_bins + 1 + 3 * _NUM_NEIGHBOR_SLOTS
        logger.function("num_features_for end: num_features=%d", num_features)
        return num_features
    if state_type in _REGION_STATE_TYPES:
        num_features = 4 * num_own_regions + 1 + 3 * _NUM_NEIGHBOR_SLOTS
        logger.function("num_features_for end: num_features=%d", num_features)
        return num_features
    if state_type in _CELL_STATE_TYPES:
        num_features = 2 * num_valid_cells + 1 + 2 * _NUM_NEIGHBOR_SLOTS
        logger.function("num_features_for end: num_features=%d", num_features)
        return num_features
    if state_type in _POLAR_REGIONAL_NEIGHBOR_STATE_TYPES:
        per_neighbor_width = 2 * num_own_regions + 1
        num_features = 4 * num_polar_bins + 1 + per_neighbor_width * _NUM_NEIGHBOR_SLOTS
        logger.function("num_features_for end: num_features=%d", num_features)
        return num_features
    logger.warning("num_features_for: unknown state_type=%s", state_type)
    raise ValueError(f"Unknown state_type: {state_type!r}")


def compute_reward(reward_type, covered_count_per_sector):
    """r_s[k+1], one of two mutually exclusive scoring conventions --
    "global": covered-UE count / num_ut across the WHOLE network -- a
    coverage RATIO (changed from a raw count 2026-09-20: num_ut is fixed
    for the whole run, so this is just a constant rescale of the same
    signal -- doesn't change what policy gets learned, only the numeric
    scale, done for the network's own gradient/loss-scale hygiene and
    log readability); the SAME scalar value for every sector (the
    objective is overall network coverage, not any one sector's own local
    slice of it -- changed from the earlier per-sector "sector_neighbors"
    reward 2026-09-20, see RegionalInfoProvider's own module docstring for
    why). No overshoot term (its physical effect already shows up as
    degraded coverage for boundary cells, via interference, rather than
    needing a separate penalty).
    "delta_covered" (2026-09-21 addition): the RAW (unnormalized) CHANGE in
    network-wide covered-UE count between this turn's pre- and post-
    decision measurements (SimulationEngine.drl_process_turn's own two-
    measurement bracket) -- under async_schedule="async" (one sector
    closes per turn, every other sector held fixed at that instant), this
    genuinely isolates the marginal effect of THAT sector's own action,
    a true per-agent credit signal (as opposed to "global"'s single
    network-wide scalar shared by every sector regardless of whose turn
    it is). Under "sync" (every sector closes at once) it collapses back
    to one shared scalar for the whole round, same limitation as "global"
    has there -- this reward type's own value proposition is async-
    specific.

    :param reward_type: "global" or "delta_covered" -- both are computed
        entirely upstream (RegionalInfoProvider.compute() for "global",
        SimulationEngine.drl_process_turn's own pre/post subtraction for
        "delta_covered"); this function is a named pass-through/validator
        for either, not a computation of its own.
    :param covered_count_per_sector: [num_sectors] -- NaN for a sector with
        no data (in scope) this window.
    :output: [num_sectors] numpy array.
    """
    logger.function("compute_reward start: reward_type=%s", reward_type)
    if reward_type in _GLOBAL_REWARD_TYPES or reward_type in _DELTA_REWARD_TYPES:
        if logger.isEnabledFor(logging.DEBUG):  # nanmean costs something, guard it
            logger.debug("compute_reward: mean reward=%.4f", np.nanmean(covered_count_per_sector))
        logger.function("compute_reward end: reward_type=%s", reward_type)
        return covered_count_per_sector
    logger.warning("compute_reward: unknown reward_type=%s", reward_type)
    raise ValueError(f"Unknown reward_type: {reward_type!r}")


def tilt_idx_to_deg(tilt_idx, downtilt_sweep_deg):
    """Action interpretation: a policy's chosen discrete index -> the
    actual per-sector tilt [deg] to apply.

    :param tilt_idx: [num_sectors] int array.
    :param downtilt_sweep_deg: candidate tilts [deg] the index selects from.
    :output: [num_sectors] float array.
    """
    return np.asarray(downtilt_sweep_deg)[tilt_idx]
