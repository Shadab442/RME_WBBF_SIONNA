"""Environment-side RL interface: state/reward construction and action
interpretation -- the contract between the environment
(helpers/simulation_engine.py) and any policy (drl/factory.py).
"""

import logging

import numpy as np

from helpers.utils import get_logger

logger = get_logger(__name__)

_REGION_STATE_TYPES = ("region_aligned",)
_SECTOR_NEIGHBORS_REWARD_TYPES = ("sector_neighbors",)


def compute_region_state(own_tilt_norm, neighbor_tilts_norm, own_occupancy, own_coverage,
                         neighbor_occupancy, neighbor_coverage, state_padding: str,
                         max_neighbors: int = 4):
    """o_i[k] = [own_occupancy (num_own_regions, FIXED/identical for every
    sector), own_coverage (num_own_regions), own tilt, up to 4 neighbors'
    own coarse occupancy, up to 4 neighbors' own coarse coverage, up to 4
    neighbor tilts] -- only the NEIGHBOR portion varies in width by how
    many neighbors a sector actually has; the own-region portion is always
    the same size (see helpers/regional_info_provider.py).

    :param own_tilt_norm: [num_sectors] float in [0,1].
    :param neighbor_tilts_norm: [num_sectors] list of [[real neighbor
        count]] float arrays in [0,1], one entry per ACTUAL neighbor.
    :param own_occupancy, own_coverage: [num_sectors, num_own_regions]
        arrays, from RegionalInfoProvider.compute().
    :param neighbor_occupancy, neighbor_coverage: [num_sectors] list of
        [[real neighbor count]] float arrays, same source.
    :param state_padding: "variable" -- every sector's true width, no
        padding for the neighbor portion (see
        helpers/regional_info_provider.py's own docstring for why a
        zero-padded slot is representationally ambiguous with a real
        all-zero neighbor/0deg neighbor). "zero" -- neighbor portion
        zero-filled up to max_neighbors for every sector.
    :output: "variable" -> [num_sectors] list of variable-length 1D arrays.
        "zero" -> [num_sectors, num_own_regions*2 + 1 + max_neighbors*3] array.
    """
    logger.function("compute_region_state start: state_padding=%s", state_padding)
    num_sectors = len(own_tilt_norm)
    num_own_regions = own_occupancy.shape[1]
    if state_padding == "variable":
        result = [
            np.concatenate([own_occupancy[s], own_coverage[s], [own_tilt_norm[s]],
                           neighbor_occupancy[s], neighbor_coverage[s], neighbor_tilts_norm[s]])
            for s in range(num_sectors)
        ]
        logger.debug("compute_region_state: widths=%s", [len(r) for r in result])
    elif state_padding == "zero":
        width = 2 * num_own_regions + 1 + 3 * max_neighbors
        result = np.zeros((num_sectors, width))
        result[:, :num_own_regions] = own_occupancy
        result[:, num_own_regions:2 * num_own_regions] = own_coverage
        result[:, 2 * num_own_regions] = own_tilt_norm
        base = 2 * num_own_regions + 1
        for s in range(num_sectors):
            nn = len(neighbor_tilts_norm[s])
            result[s, base:base + nn] = neighbor_occupancy[s]
            result[s, base + max_neighbors:base + max_neighbors + nn] = neighbor_coverage[s]
            result[s, base + 2 * max_neighbors:base + 2 * max_neighbors + nn] = neighbor_tilts_norm[s]
        logger.debug("compute_region_state: shape=%s", result.shape)
    else:
        raise ValueError(f"Unknown state_padding: {state_padding!r}")
    logger.function("compute_region_state end")
    return result


def num_features_for(state_type, num_own_regions=None, num_neighbors_per_sector=None,
                     state_padding=None, max_neighbors=4):
    """Feature count(s) for a given state_type, so callers (e.g. sizing a
    policy's input layer) don't have to duplicate compute_region_state's
    branching.

    :output: state_padding="zero" -> single int. state_padding="variable"
        -> a [num_sectors] list of per-sector ints (Dqn accepts either).
    """
    logger.function("num_features_for start: state_type=%s", state_type)
    if state_type in _REGION_STATE_TYPES:
        if state_padding == "zero":
            num_features = 2 * num_own_regions + 1 + 3 * max_neighbors
            logger.function("num_features_for end: num_features=%d", num_features)
            return num_features
        if state_padding == "variable":
            num_features = [2 * num_own_regions + 1 + 3 * n_neighbors
                            for n_neighbors in num_neighbors_per_sector]
            logger.function("num_features_for end: variable, %d sectors", len(num_features))
            return num_features
        raise ValueError(f"Unknown state_padding: {state_padding!r}")
    logger.warning("num_features_for: unknown state_type=%s", state_type)
    raise ValueError(f"Unknown state_type: {state_type!r}")


def compute_reward(reward_type, coverage_per_sector):
    """RL Problem Formulation slide 8: r_s[k+1] = mean coverage over UEs
    geographically located within this sector's own observation region X_s
    -- no overshoot term (its physical effect already shows up as
    degraded coverage for boundary cells, via interference, rather than
    needing a separate penalty).

    :param reward_type: "sector_neighbors" -- RegionalInfoProvider
        .compute()'s reward output, pooled over the sector + its actual
        (2-4) side-sharing neighbors' full areas (not just boundary-adjacent
        cells) -- already fully aggregated by its own estimator, this is
        a pass-through.
    :param coverage_per_sector: [num_sectors] -- NaN for a sector with no
        data (in scope) this window.
    :output: [num_sectors] numpy array.
    """
    logger.function("compute_reward start: reward_type=%s", reward_type)
    if reward_type in _SECTOR_NEIGHBORS_REWARD_TYPES:
        if logger.isEnabledFor(logging.DEBUG):  # nanmean costs something, guard it
            logger.debug("compute_reward: mean reward=%.4f", np.nanmean(coverage_per_sector))
        logger.function("compute_reward end: reward_type=%s", reward_type)
        return coverage_per_sector
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
