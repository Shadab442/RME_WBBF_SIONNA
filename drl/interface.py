"""Environment-side RL interface: state/reward construction and action
interpretation -- the contract between the environment
(helpers/simulation_engine.py) and any policy (drl/factory.py).

Named/typed so a new state or reward formulation is a new branch here plus
a config value (state_type / reward_type), not an edit to whatever script
is running the comparison. Mirrors drl/factory.py's create_policy(name, ...)
pattern for the same reason.

Distinct from helpers.kpi_manager.KpiManager.compute_ue_kpis: that computes
raw per-sector KPIs (coverage fraction, overshoot fraction) -- general-
purpose measurements shared by DRL and AdaptiveLegacyTiltController alike.
This module turns those raw KPIs into an RL-specific state vector and
scalar reward, which is what's actually DRL-specific -- not the KPIs
themselves.
"""

import numpy as np


def compute_state(state_type, coverage_per_sector, overshoot_per_sector, tilt_deg_per_sector=None):
    """
    :param state_type: "coverage_overshoot" -- the only type implemented so
        far: [coverage, overshoot] per sector.
    :param tilt_deg_per_sector: unused by "coverage_overshoot"; kept as a
        parameter so a future tilt-including state_type doesn't change this
        function's call signature, just adds a branch.
    :output: [num_sectors, num_features] numpy array.
    """
    if state_type == "coverage_overshoot":
        return np.stack([coverage_per_sector, overshoot_per_sector], axis=1)
    raise ValueError(f"Unknown state_type: {state_type!r}")


def num_features_for(state_type):
    """Feature count for a given state_type, so callers (e.g. sizing a
    policy's input layer) don't have to duplicate compute_state's branching."""
    if state_type == "coverage_overshoot":
        return 2
    raise ValueError(f"Unknown state_type: {state_type!r}")


def compute_reward(reward_type, coverage_per_sector, overshoot_per_sector,
                   reward_lambda_coverage, reward_lambda_overshoot):
    """
    :param reward_type: "hard" -- the only type implemented so far:
        reward_lambda_coverage * coverage_per_sector -
        reward_lambda_overshoot * overshoot_per_sector, both plain
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
