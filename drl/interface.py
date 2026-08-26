"""Environment-side RL interface: state/reward construction and action
interpretation -- the contract between the environment
(helpers/simulation_engine.py) and any policy (drl/factory.py).
"""

import numpy as np

_TOP_K_STATE_TYPES = ("top_k_neighbor", "predicted_top_k_neighbor")


def compute_state(state_type, own_tilt_norm, neighbor_overshoot, top_k_identity, top_k_coverage):
    """
    :param state_type:
        "top_k_neighbor" / "predicted_top_k_neighbor" 
    :param own_tilt_norm: [num_sectors] float in [0,1].
    :param neighbor_overshoot: [num_sectors, max_neighbors].
    :param top_k_identity, top_k_coverage: [num_sectors, top_k_locations]
        each -- top_k_identity is each selected grid cell's own index
        (normalized to [0,1]), always a real value (ranking a fixed-size
        grid always produces a definite top-k); top_k_coverage is that
        cell's coverage fraction, -1 if it had zero visits this window.
    :output: [num_sectors, num_features] numpy array.
    """
    if state_type in _TOP_K_STATE_TYPES:
        num_sectors = own_tilt_norm.shape[0]
        # [identity, coverage] interleaved per selected location.
        top_k_flat = np.stack([top_k_identity, top_k_coverage], axis=-1).reshape(num_sectors, -1)
        return np.concatenate([own_tilt_norm[:, None], top_k_flat, neighbor_overshoot], axis=1)
    raise ValueError(f"Unknown state_type: {state_type!r}")


def num_features_for(state_type, max_neighbors, top_k_locations):
    """Feature count for a given state_type, so callers (e.g. sizing a
    policy's input layer) don't have to duplicate compute_state's branching.
    """
    if state_type in _TOP_K_STATE_TYPES:
        return 1 + 2 * top_k_locations + max_neighbors
    raise ValueError(f"Unknown state_type: {state_type!r}")


def compute_reward(reward_type, coverage_per_sector, overshoot_per_sector,
                   reward_lambda_coverage, reward_lambda_overshoot):
    """
    :param reward_type:
        "hard" -- per-sector reward_lambda_coverage * coverage_per_sector[i]
            - reward_lambda_overshoot * overshoot_per_sector[i], both plain
            threshold-crossing fractions. ("soft", a continuous SINR-margin
            version, is a planned addition -- not yet implemented.)
    :output: [num_sectors] numpy array.
    """
    if reward_type == "hard":
        return reward_lambda_coverage * coverage_per_sector - reward_lambda_overshoot * overshoot_per_sector
    raise ValueError(f"Unknown reward_type: {reward_type!r}")


def tilt_idx_to_deg(tilt_idx, downtilt_sweep_deg):
    """Action interpretation: a policy's chosen discrete index -> the
    actual per-sector tilt [deg] to apply.

    :param tilt_idx: [num_sectors] int array.
    :param downtilt_sweep_deg: candidate tilts [deg] the index selects from.
    :output: [num_sectors] float array.
    """
    return np.asarray(downtilt_sweep_deg)[tilt_idx]
