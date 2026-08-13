"""The environment side of the dynamic tilt-control comparison: topology/
mobility/channel setup, per-interval pooling across all five methods
(Dynamic Local Oracle/Causal, Adaptive Legacy, No Tilt, DRL), the mobility
animation, and per-sector raw KPI computation (coverage, overshoot).

Everything here is parameterized off a merged config dict (common + the
calling script's own section -- see helpers.config.load_config), not
module-level globals, so the calling script (test_dynamic_scenario_tilts_effect.py)
stays a thin driver: load config, call configure_environment() then
run_history(), save/print the results.

Distinct from drl/interface.py: this module computes raw per-sector KPIs
(compute_per_sector_coverage_overshoot) and drives the environment/channel
side of the loop; drl/interface.py turns those raw KPIs into an RL-specific
state vector and reward, and interprets a policy's action -- that's a
DRL-problem-definition concern, not an environment one, even though it's
the environment that calls it every interval.
"""

import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import torch

from sionna.phy.channel.tr38901 import Antenna, AntennaArray, RMa, UMa, UMi
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from sionna.phy.constants import BOLTZMANN_CONSTANT

from drl.factory import create_policy
from drl.interface import compute_reward, compute_state, num_features_for, tilt_idx_to_deg
from .cellular_topology import CellularTopology
from .electrical_downtilt import ElectricalDowntilt
from .kpi_calculator import KpiCalculator
from .mobility import RandomWalkMobility, ReferencePointGroupMobility
from .tilt_controller import AdaptiveLegacyTiltController, DynamicTiltController, LocalTiltSelector, \
    RLTiltController
from .ue_drop import sample_cluster_start_xy, sample_clustered_ut_loc, sample_uniform_ut_loc

_SCENARIO_CHANNEL_MODEL = {"umi": UMi, "uma": UMa, "rma": RMa}

# No Tilt: fixed boresight (0 deg) on every sector, forever -- see
# ElectricalDowntilt's own docstring ("0 = boresight"), a valid tilt in its
# own right, just outside downtilt_sweep_deg's search range.
NO_TILT_DEG = 0.0


def configure_environment(cfg):
    """Build one persistent topology, mobility model, channel, and KPI path
    from a merged config dict.

    :output: (topology, mobility_list, channel_model, kpi_calculator,
        ut_orientations, ut_velocities, in_state, bs_xy).
    """
    scenario = cfg["scenario"]
    num_rings = cfg["num_rings"]
    num_ut = cfg["num_ut"]
    ut_height = cfg["ut_height"]
    m = cfg["m"]
    downtilt_sweep_deg = cfg["downtilt_sweep_deg"]
    antenna_window = cfg["antenna_window"]
    bs_tx_power_dbm = cfg["bs_tx_power_dbm"]
    channel_bandwidth_prb = cfg["channel_bandwidth_prb"]
    temperature = cfg["temperature"]
    carrier_frequency = cfg["carrier_frequency"]
    environment_seed = cfg["environment_seed"]
    mobility_model = cfg["mobility_model"]
    min_ut_speed = cfg["min_ut_speed"]
    max_ut_speed = cfg["max_ut_speed"]
    num_groups = cfg["num_groups"]
    deviation_radius_frac_area = cfg["deviation_radius_frac_area"]
    member_jitter_speed = cfg["member_jitter_speed"]
    num_realizations_per_slot = cfg["num_realizations_per_slot"]
    max_realization_cuda = cfg["max_realization_cuda"]

    assert num_ut % num_groups == 0, \
        "num_ut must be divisible by num_groups for equal-sized RPGM groups"
    assert mobility_model in ("rpgm", "random_walk"), \
        "mobility_model must be 'rpgm' or 'random_walk'"
    assert num_realizations_per_slot % max_realization_cuda == 0, \
        "num_realizations_per_slot must be a multiple of max_realization_cuda"
    members_per_group = num_ut // num_groups

    min_bs_ut_dist, isd, bs_height, min_ut_height, _, _, _, _ = \
        set_3gpp_scenario_parameters(scenario)
    scenario_params = {
        "isd": isd,
        "bs_height": bs_height,
        "min_bs_ut_dist": min_bs_ut_dist,
        "min_ut_height": min_ut_height,
    }
    topology = CellularTopology(scenario_params, num_rings=num_rings, batch_size=max_realization_cuda)

    mobility_list = []
    for i in range(num_realizations_per_slot):
        # Isolated per-realization RNG stream, seeded environment_seed + i --
        # NOT one generator shared across every realization: a shared stream
        # would make realization i's draws depend on exactly how many random
        # numbers every EARLIER realization happened to consume
        # (sample_valid_offset's rejection-sampling retries are
        # variable-length), so reproducing just one realization in isolation
        # wouldn't be possible. A per-realization seed also NEVER touches the
        # global torch RNG, so a policy's own algorithm-side seeding can't
        # perturb this run's environment trajectory, and vice versa.
        environment_generator = torch.Generator(device=topology.bs_loc.device)
        environment_generator.manual_seed(environment_seed + i)

        if mobility_model == "rpgm":
            start_xy_list = sample_cluster_start_xy(topology, num_groups, generator=environment_generator)
            deviation_radius = deviation_radius_frac_area * topology.default_drop_radius
            initial_ut_loc, member_group_idx = sample_clustered_ut_loc(
                topology, start_xy_list, members_per_group, deviation_radius, ut_height,
                dtype=topology.bs_loc.dtype, device=topology.bs_loc.device,
                generator=environment_generator,
            )
            mobility = ReferencePointGroupMobility(
                initial_ut_loc, member_group_idx, start_xy_list,
                deviation_radius=deviation_radius, topo=topology,
                min_speed=min_ut_speed, max_speed=max_ut_speed,
                member_jitter_speed=member_jitter_speed,
                generator=environment_generator,
            )
        else:
            initial_ut_loc = sample_uniform_ut_loc(
                topology, num_ut, ut_height,
                dtype=topology.bs_loc.dtype, device=topology.bs_loc.device,
                generator=environment_generator,
            )
            mobility = RandomWalkMobility(
                initial_ut_loc, topo=topology,
                min_speed=min_ut_speed, max_speed=max_ut_speed,
                generator=environment_generator,
            )
        mobility_list.append(mobility)

    bs_array = AntennaArray(
        num_rows=m, num_cols=1, polarization="single", polarization_type="V",
        antenna_pattern="38.901", carrier_frequency=carrier_frequency,
    )
    ut_array = Antenna(
        polarization="single", polarization_type="V",
        antenna_pattern="omni", carrier_frequency=carrier_frequency,
    )
    sector_etilts = [
        ElectricalDowntilt(bs_array, carrier_frequency=carrier_frequency,
                          downtilt_deg=downtilt_sweep_deg[0], window=antenna_window)
        for _ in range(topology.num_bs)
    ]

    channel_model_cls = _SCENARIO_CHANNEL_MODEL[scenario]
    channel_model_kwargs = dict(
        carrier_frequency=carrier_frequency, bs_array=bs_array, ut_array=ut_array,
        direction="downlink", enable_pathloss=True, enable_shadow_fading=True,
    )
    if scenario != "rma":
        channel_model_kwargs["o2i_model"] = "low"
    channel_model = channel_model_cls(**channel_model_kwargs)

    bs_tx_power_w = 10 ** ((bs_tx_power_dbm - 30.0) / 10.0)
    channel_bandwidth_hz = channel_bandwidth_prb * 12 * 30e3
    noise_power_w = BOLTZMANN_CONSTANT * temperature * channel_bandwidth_hz
    kpi_calculator = KpiCalculator(channel_model, sector_etilts, bs_tx_power_w, noise_power_w)

    ut_orientations = torch.zeros(max_realization_cuda, num_ut, 3,
                                  dtype=topology.bs_loc.dtype, device=topology.bs_loc.device)
    ut_velocities = torch.zeros_like(ut_orientations)
    in_state = torch.zeros(max_realization_cuda, num_ut, dtype=torch.bool, device=topology.bs_loc.device)

    # Sites/sectors share the same (x, y) geometry across the batch
    # dimension (only the channel draw is batched) -- any batch index works.
    bs_xy = topology.bs_loc[0, :, :2].detach()

    return topology, mobility_list, channel_model, kpi_calculator, ut_orientations, ut_velocities, in_state, bs_xy


def compute_per_sector_coverage_overshoot(ut_loc, power_w, noise_power_w, bs_xy,
                                          distance_threshold_m, overlap_margin_db,
                                          coverage_threshold_db):
    """Per-sector (coverage, overshoot) -- the same two ingredients
    AdaptiveLegacyTiltController computes internally (coverage here is the
    plain whole-population fraction used everywhere else in this project,
    NOT that class's edge-restricted r_bc; overshoot is exactly n_os,
    unchanged), factored out as a standalone function so both that
    controller and DRL's state/reward (drl.interface) can share one
    implementation. (AdaptiveLegacyTiltController itself still computes its
    own n_os inline for now -- not yet refactored to call this.)

    :param ut_loc: [num_ue, 3] one realization's UE positions (numpy or
        torch), pooled across measurement draws (each draw's UEs are
        treated as more population samples, same pooling convention used
        throughout this project).
    :param power_w: [num_sectors, num_ue] received power [W], same pooling.
    :output: (coverage_per_sector [num_sectors], overshoot_per_sector
        [num_sectors]) numpy arrays.
    """
    dtype = torch.float32
    device = "cpu"
    ut_loc = torch.as_tensor(ut_loc, dtype=dtype, device=device)
    power_w = torch.as_tensor(power_w, dtype=dtype, device=device)
    bs_xy = torch.as_tensor(bs_xy, dtype=dtype, device=device)
    num_sectors = power_w.shape[0]

    total_w = power_w.sum(dim=0, keepdim=True)
    interference_w = total_w - power_w
    sinr = power_w / (interference_w + noise_power_w)
    sinr_db = 10.0 * torch.log10(sinr)
    best_sinr_db, serving_idx = sinr_db.max(dim=0)
    power_db = 10.0 * torch.log10(power_w)
    serving_power_db = torch.gather(power_db, 0, serving_idx.unsqueeze(0)).squeeze(0)
    dist_to_site = torch.linalg.norm(ut_loc[None, :, :2] - bs_xy[:, None, :], dim=-1)

    coverage_per_sector = torch.zeros(num_sectors, dtype=dtype, device=device)
    overshoot_per_sector = torch.zeros(num_sectors, dtype=dtype, device=device)
    for i in range(num_sectors):
        served_by_i = serving_idx == i
        if served_by_i.any():
            coverage_per_sector[i] = (best_sinr_db[served_by_i] > coverage_threshold_db).to(dtype).mean()

        served_elsewhere = serving_idx != i
        power_gap_db = serving_power_db - power_db[i]
        comparable = served_elsewhere & (power_gap_db <= overlap_margin_db)
        is_far = dist_to_site[i] > distance_threshold_m
        overshoot_per_sector[i] = (comparable & is_far).to(dtype).mean()

    return coverage_per_sector.numpy(), overshoot_per_sector.numpy()


def run_history(cfg, topology, mobility_list, channel_model, kpi_calculator,
                ut_orientations, ut_velocities, in_state, bs_xy):
    """Evaluate Dynamic Local Oracle, Dynamic Local Causal, Adaptive Legacy,
    No Tilt, and DRL -- against the same per-CONTROL-INTERVAL pooled data,
    every interval. See test_dynamic_scenario_tilts_effect.py's module
    docstring for the full per-method description; this is the loop that
    implements it.
    """
    downtilt_sweep_deg = np.asarray(cfg["downtilt_sweep_deg"])
    coverage_threshold_db = cfg["coverage_threshold_db"]
    coordinate_ascent_max_rounds = cfg["coordinate_ascent_max_rounds"]
    sinr_percentiles = cfg["sinr_percentiles"]
    adaptive_legacy_overlap_margin_db = cfg["adaptive_legacy_overlap_margin_db"]
    adaptive_legacy_edge_percentile = cfg["adaptive_legacy_edge_percentile"]
    adaptive_legacy_tilt_step_deg = cfg["adaptive_legacy_tilt_step_deg"]
    adaptive_legacy_os_threshold = cfg["adaptive_legacy_os_threshold"]
    adaptive_legacy_bc_threshold = cfg["adaptive_legacy_bc_threshold"]
    num_ut = cfg["num_ut"]
    num_tilt_control_intervals = cfg["num_tilt_control_intervals"]
    measurement_interval_s = cfg["measurement_interval_s"]
    tilt_control_interval_s = cfg["tilt_control_interval_s"]
    max_realization_cuda = cfg["max_realization_cuda"]
    num_realizations_per_slot = cfg["num_realizations_per_slot"]
    policy_name = cfg["policy_name"]
    algorithm_seed = cfg["algorithm_seed"]
    state_type = cfg["state_type"]
    reward_type = cfg["reward_type"]
    reward_lambda_coverage = cfg["reward_lambda_coverage"]
    reward_lambda_overshoot = cfg["reward_lambda_overshoot"]
    dqn_kwargs = dict(cfg["dqn"])

    assert abs(tilt_control_interval_s / measurement_interval_s
              - round(tilt_control_interval_s / measurement_interval_s)) < 1e-9, \
        "tilt_control_interval_s must be an exact multiple of measurement_interval_s"
    measurement_slots_per_interval = round(tilt_control_interval_s / measurement_interval_s)

    num_bs = topology.num_bs
    distance_threshold_m = float(topology.grid.cell_radius)

    coverage_dynamic_local_oracle = np.zeros(num_tilt_control_intervals)
    coverage_dynamic_local_causal = np.zeros(num_tilt_control_intervals)
    coverage_adaptive_legacy = np.zeros(num_tilt_control_intervals)
    coverage_no_tilt = np.zeros(num_tilt_control_intervals)
    coverage_drl = np.zeros(num_tilt_control_intervals)
    # [len(sinr_percentiles), num_tilt_control_intervals] per method.
    sinr_percentiles_dynamic_local_oracle = np.zeros((len(sinr_percentiles), num_tilt_control_intervals))
    sinr_percentiles_dynamic_local_causal = np.zeros((len(sinr_percentiles), num_tilt_control_intervals))
    sinr_percentiles_adaptive_legacy = np.zeros((len(sinr_percentiles), num_tilt_control_intervals))
    sinr_percentiles_no_tilt = np.zeros((len(sinr_percentiles), num_tilt_control_intervals))
    sinr_percentiles_drl = np.zeros((len(sinr_percentiles), num_tilt_control_intervals))
    dynamic_local_oracle_tilt_deg_history = np.zeros((num_tilt_control_intervals, num_bs))
    dynamic_local_causal_tilt_deg_history = np.zeros((num_tilt_control_intervals, num_bs))
    adaptive_legacy_tilt_deg_history = np.zeros((num_tilt_control_intervals, num_bs))
    drl_tilt_deg_history = np.zeros((num_tilt_control_intervals, num_bs))
    # DRL-only: per-sector reward/overshoot (the same two numbers state and
    # reward are built from) and per-interval training loss (NaN before the
    # replay buffer clears warmup) -- no other method has these concepts.
    drl_reward_history = np.zeros((num_tilt_control_intervals, num_bs))
    drl_overshoot_history = np.zeros((num_tilt_control_intervals, num_bs))
    drl_loss_per_interval = np.full(num_tilt_control_intervals, np.nan)
    position_history = np.zeros((num_tilt_control_intervals, num_ut, 3))
    # Realization 0's per-cluster reference points, for drawing cluster
    # circles in the animation (RPGM only -- RandomWalkMobility has no
    # groups/reference points).
    has_clusters = hasattr(mobility_list[0], "ref_xy")
    ref_xy_history = np.zeros((num_tilt_control_intervals, mobility_list[0].ref_xy.shape[0], 2)) if has_clusters else None

    dynamic_local_oracle_controller = DynamicTiltController(LocalTiltSelector(max_rounds=coordinate_ascent_max_rounds))
    dynamic_local_causal_controller = DynamicTiltController(LocalTiltSelector(max_rounds=coordinate_ascent_max_rounds))

    # Adaptive Legacy: starts from the same downtilt_sweep_deg[0] every
    # sector_etilt is physically initialized to (configure_environment) --
    # the "as deployed, before any adaptation" starting point.
    adaptive_legacy_controller = AdaptiveLegacyTiltController(
        num_sectors=num_bs,
        initial_tilt_deg=downtilt_sweep_deg[0],
        theta_min_deg=downtilt_sweep_deg.min(),
        theta_max_deg=downtilt_sweep_deg.max(),
        distance_threshold_m=distance_threshold_m,
        coverage_threshold_db=coverage_threshold_db,
        overlap_margin_db=adaptive_legacy_overlap_margin_db,
        edge_percentile=adaptive_legacy_edge_percentile,
        tilt_step_deg=adaptive_legacy_tilt_step_deg,
        os_threshold=adaptive_legacy_os_threshold,
        bc_threshold=adaptive_legacy_bc_threshold,
        dtype=topology.bs_loc.dtype,
        device=topology.bs_loc.device,
    )

    # DRL: index 0 into downtilt_sweep_deg (the same "as deployed" starting
    # point Adaptive Legacy uses), decided/scored with the same one-interval
    # lag -- see module docstring.
    drl_policy = create_policy(
        policy_name, num_bs, num_features=num_features_for(state_type), num_actions=len(downtilt_sweep_deg),
        dqn_kwargs=dqn_kwargs, algorithm_seed=algorithm_seed, device="cpu",
    )
    drl_controller = RLTiltController(drl_policy, num_bs, initial_tilt_idx=0)

    def score(power_table, assignment):
        """(coverage, sinr percentiles) for one assignment -- pulls the
        per-UE SINR array from coverage_from_assignment's return_details
        only long enough to reduce it to sinr_percentiles; the full array
        itself is never kept."""
        coverage, sinr_db, _ = kpi_calculator.coverage_from_assignment(
            power_table, assignment, coverage_threshold_db, return_details=True
        )
        return coverage, np.percentile(sinr_db, sinr_percentiles)

    def score_power_w(power_w):
        """(coverage, sinr percentiles) directly from a [num_sectors,
        num_samples] power matrix (a fixed tilt, not a downtilt_sweep_deg
        index) -- same signal/interference/SINR formula as
        coverage_from_assignment/score above, applied without a
        power_table+tilt_idx lookup. Used by Adaptive Legacy, No Tilt, DRL."""
        total_w = power_w.sum(axis=0, keepdims=True)
        interference_w = total_w - power_w
        sinr = power_w / (interference_w + kpi_calculator.noise_power_w)
        best_sinr = sinr.max(axis=0)
        sinr_db = 10.0 * np.log10(best_sinr)
        coverage = float(np.mean(sinr_db > coverage_threshold_db))
        return coverage, np.percentile(sinr_db, sinr_percentiles)

    def build_power_table_slot(adaptive_tilt_deg, drl_tilt_deg):
        """One MEASUREMENT draw's pooled [num_tilts, num_sectors,
        num_realizations_per_slot*num_ut] power table, from every
        realization's CURRENT position -- plus, from the SAME per-chunk
        large-scale state (common random numbers, no extra channel draw):
        Adaptive Legacy's own power matrix at ``adaptive_tilt_deg``,
        DRL's own power matrix at ``drl_tilt_deg`` (both per-sector, not a
        downtilt_sweep_deg index; both pooled across every realization for
        scoring and separately for realization 0 alone, for each
        controller's own update -- see module docstring); and the No Tilt
        baseline's power matrix at a fixed 0-degree tilt, pooled across
        every realization.
        """
        num_chunks = num_realizations_per_slot // max_realization_cuda
        chunks = []
        adaptive_chunks = []
        adaptive_power_w_r0 = None
        drl_chunks = []
        drl_power_w_r0 = None
        no_tilt_chunks = []
        drl_tilt_list = drl_tilt_deg.tolist() if hasattr(drl_tilt_deg, "tolist") else list(drl_tilt_deg)
        for chunk in range(num_chunks):
            chunk_mobility = mobility_list[chunk * max_realization_cuda:(chunk + 1) * max_realization_cuda]
            ut_loc = torch.stack([mobility.ut_loc for mobility in chunk_mobility], dim=0)
            bs_virtual_loc = topology.mirror_bs_loc(ut_loc)
            channel_model.set_topology(
                ut_loc, topology.bs_loc, ut_orientations, topology.bs_orientations,
                ut_velocities, in_state, None, bs_virtual_loc,
            )
            state = kpi_calculator.generate_large_scale_state()
            chunks.append(kpi_calculator.compute_power_table(downtilt_sweep_deg, state))

            # compute_power_table (above) leaves every sector_etilt set to
            # downtilt_sweep_deg[-1] (its last sweep step) -- re-set each to
            # Adaptive Legacy's own per-sector tilt before reading its power
            # off the SAME state.
            for etilt, tilt in zip(kpi_calculator.sector_etilts, adaptive_tilt_deg.tolist()):
                etilt.set_tilt(tilt)
            adaptive_power_w = kpi_calculator.compute_power_matrix_w(state)  # [num_sectors, batch, num_ue]
            adaptive_power_w_np = adaptive_power_w.detach().cpu().numpy()
            adaptive_chunks.append(adaptive_power_w_np.reshape(adaptive_power_w_np.shape[0], -1))
            if chunk == 0:
                adaptive_power_w_r0 = adaptive_power_w_np[:, 0, :]  # mobility_list[0]'s realization

            # Same pattern for DRL's own current tilt.
            for etilt, tilt in zip(kpi_calculator.sector_etilts, drl_tilt_list):
                etilt.set_tilt(tilt)
            drl_power_w = kpi_calculator.compute_power_matrix_w(state)
            drl_power_w_np = drl_power_w.detach().cpu().numpy()
            drl_chunks.append(drl_power_w_np.reshape(drl_power_w_np.shape[0], -1))
            if chunk == 0:
                drl_power_w_r0 = drl_power_w_np[:, 0, :]

            # Same pattern for the fixed No Tilt baseline.
            for etilt in kpi_calculator.sector_etilts:
                etilt.set_tilt(NO_TILT_DEG)
            no_tilt_power_w = kpi_calculator.compute_power_matrix_w(state)
            no_tilt_power_w_np = no_tilt_power_w.detach().cpu().numpy()
            no_tilt_chunks.append(no_tilt_power_w_np.reshape(no_tilt_power_w_np.shape[0], -1))

        power_table = np.concatenate(chunks, axis=2)
        adaptive_power_w_pooled = np.concatenate(adaptive_chunks, axis=1)
        drl_power_w_pooled = np.concatenate(drl_chunks, axis=1)
        no_tilt_power_w_pooled = np.concatenate(no_tilt_chunks, axis=1)
        return (power_table, adaptive_power_w_pooled, adaptive_power_w_r0,
               drl_power_w_pooled, drl_power_w_r0, no_tilt_power_w_pooled)

    def build_pooled_interval(adaptive_tilt_deg, drl_tilt_deg, is_last_interval):
        """One TILT CONTROL INTERVAL's pooled data: measurement_slots_per_interval
        consecutive 1-second measurement draws (build_power_table_slot),
        each advancing mobility by measurement_interval_s, pooled together
        -- same mega-chunk/windowing pattern test_tilt_window_calibration.py
        validated, applied here with one fixed window instead of a sweep of
        candidates.
        """
        power_table_chunks = []
        adaptive_pooled_chunks = []
        adaptive_r0_chunks = []
        drl_pooled_chunks = []
        drl_r0_chunks = []
        no_tilt_chunks = []
        ut_loc_r0_chunks = []
        position_snapshot = None
        ref_xy_snapshot = None

        for sub_step in range(measurement_slots_per_interval):
            if sub_step == 0:
                position_snapshot = mobility_list[0].ut_loc.detach().cpu().numpy().copy()
                if has_clusters:
                    ref_xy_snapshot = mobility_list[0].ref_xy.detach().cpu().numpy().copy()
            ut_loc_r0_chunks.append(mobility_list[0].ut_loc.detach().cpu().numpy().copy())

            (power_table, adaptive_pooled, adaptive_r0,
             drl_pooled, drl_r0, no_tilt_pooled) = build_power_table_slot(adaptive_tilt_deg, drl_tilt_deg)
            power_table_chunks.append(power_table)
            adaptive_pooled_chunks.append(adaptive_pooled)
            adaptive_r0_chunks.append(adaptive_r0)
            drl_pooled_chunks.append(drl_pooled)
            drl_r0_chunks.append(drl_r0)
            no_tilt_chunks.append(no_tilt_pooled)

            is_last_sub_step = is_last_interval and (sub_step == measurement_slots_per_interval - 1)
            if not is_last_sub_step:
                for mobility in mobility_list:
                    mobility.step(measurement_interval_s)

        return (
            np.concatenate(power_table_chunks, axis=2),
            np.concatenate(adaptive_pooled_chunks, axis=1),
            np.concatenate(adaptive_r0_chunks, axis=1),
            np.concatenate(drl_pooled_chunks, axis=1),
            np.concatenate(drl_r0_chunks, axis=1),
            np.concatenate(no_tilt_chunks, axis=1),
            np.concatenate(ut_loc_r0_chunks, axis=0),
            position_snapshot,
            ref_xy_snapshot,
        )

    for interval in range(num_tilt_control_intervals):
        is_last_interval = interval == num_tilt_control_intervals - 1
        drl_tilt_deg_this_interval = tilt_idx_to_deg(drl_controller.tilt_idx, downtilt_sweep_deg)
        (
            pooled_power_table,
            pooled_adaptive_power_w,
            pooled_adaptive_power_w_r0,
            pooled_drl_power_w,
            pooled_drl_power_w_r0,
            pooled_no_tilt_power_w,
            pooled_ut_loc_r0,
            position_snapshot,
            ref_xy_snapshot,
        ) = build_pooled_interval(adaptive_legacy_controller.tilt_deg, drl_tilt_deg_this_interval, is_last_interval)
        position_history[interval] = position_snapshot
        if has_clusters:
            ref_xy_history[interval] = ref_xy_snapshot

        # Oracle: decide AND score using THIS interval's pooled power table -- no lag.
        dynamic_local_oracle_assignment = dynamic_local_oracle_controller.select(
            kpi_calculator, pooled_power_table, coverage_threshold_db
        )

        coverage_dynamic_local_oracle[interval], sinr_percentiles_dynamic_local_oracle[:, interval] = score(
            pooled_power_table, dynamic_local_oracle_assignment
        )
        dynamic_local_oracle_tilt_deg_history[interval] = downtilt_sweep_deg[dynamic_local_oracle_assignment]

        # Causal: score the assignment decided from data through the
        # previous interval against this interval's actual pooled power
        # table. Interval 0 falls back to Oracle's interval-0 pick (a free
        # reuse -- see module docstring). Decision happens after scoring,
        # from THIS interval's own pooled data (already a full 300-second
        # window -- no extra cross-interval pooling needed on top).
        causal_applied_assignment = (
            dynamic_local_causal_controller.assignment
            if dynamic_local_causal_controller.assignment is not None
            else dynamic_local_oracle_assignment
        )
        coverage_dynamic_local_causal[interval], sinr_percentiles_dynamic_local_causal[:, interval] = score(
            pooled_power_table, causal_applied_assignment
        )
        dynamic_local_causal_tilt_deg_history[interval] = downtilt_sweep_deg[causal_applied_assignment]
        dynamic_local_causal_controller.select(kpi_calculator, pooled_power_table, coverage_threshold_db)

        # Adaptive Legacy: score at the tilt already in effect (pooled live
        # power, like every other method's scoring), THEN update from a
        # SINGLE realization's measurements pooled over this whole interval
        # -- same one-interval lag as Dynamic Local Causal.
        adaptive_legacy_tilt_deg_history[interval] = adaptive_legacy_controller.tilt_deg.detach().cpu().numpy()
        coverage_adaptive_legacy[interval], sinr_percentiles_adaptive_legacy[:, interval] = score_power_w(
            pooled_adaptive_power_w
        )
        adaptive_legacy_controller.update(
            pooled_ut_loc_r0, pooled_adaptive_power_w_r0, kpi_calculator.noise_power_w, bs_xy
        )

        # No Tilt: fixed 0 degrees, forever -- just scored, never decided.
        coverage_no_tilt[interval], sinr_percentiles_no_tilt[:, interval] = score_power_w(pooled_no_tilt_power_w)

        # DRL: score at the tilt already in effect (pooled live power, like
        # every other method's scoring) for the coverage/SINR comparison,
        # THEN compute its own per-sector (coverage, overshoot) state/reward
        # off the SAME realization-0-only pooled data Adaptive Legacy uses
        # (drl.interface -- see that module for why state/reward live
        # there, not here), and step the policy -- same one-interval lag.
        # "Episode" == one interval here: this loop IS the training run.
        drl_tilt_deg_history[interval] = drl_tilt_deg_this_interval
        coverage_drl[interval], sinr_percentiles_drl[:, interval] = score_power_w(pooled_drl_power_w)
        drl_coverage_per_sector, drl_overshoot_per_sector = compute_per_sector_coverage_overshoot(
            pooled_ut_loc_r0, pooled_drl_power_w_r0, kpi_calculator.noise_power_w, bs_xy,
            distance_threshold_m, adaptive_legacy_overlap_margin_db, coverage_threshold_db,
        )
        drl_overshoot_history[interval] = drl_overshoot_per_sector
        drl_reward_history[interval] = compute_reward(
            reward_type, drl_coverage_per_sector, drl_overshoot_per_sector,
            reward_lambda_coverage, reward_lambda_overshoot,
        )
        drl_losses_before = len(drl_policy.step_losses)
        drl_observations = compute_state(state_type, drl_coverage_per_sector, drl_overshoot_per_sector)
        drl_controller.step(drl_observations, drl_reward_history[interval], training=True)
        if len(drl_policy.step_losses) > drl_losses_before:
            drl_loss_per_interval[interval] = np.mean(drl_policy.step_losses[drl_losses_before:])

        if interval % 10 == 0 or is_last_interval:
            print(
                f"interval {interval:3d}/{num_tilt_control_intervals - 1}: "
                f"dynamic_local_oracle={coverage_dynamic_local_oracle[interval]:.3f}, "
                f"dynamic_local_causal={coverage_dynamic_local_causal[interval]:.3f}, "
                f"adaptive_legacy={coverage_adaptive_legacy[interval]:.3f}, "
                f"no_tilt={coverage_no_tilt[interval]:.3f}, "
                f"drl={coverage_drl[interval]:.3f}"
            )

    no_tilt_deg = np.full(num_bs, NO_TILT_DEG)

    return (
        coverage_dynamic_local_oracle,
        coverage_dynamic_local_causal,
        coverage_adaptive_legacy,
        coverage_no_tilt,
        coverage_drl,
        sinr_percentiles_dynamic_local_oracle,
        sinr_percentiles_dynamic_local_causal,
        sinr_percentiles_adaptive_legacy,
        sinr_percentiles_no_tilt,
        sinr_percentiles_drl,
        dynamic_local_oracle_tilt_deg_history,
        dynamic_local_causal_tilt_deg_history,
        adaptive_legacy_tilt_deg_history,
        drl_tilt_deg_history,
        drl_reward_history,
        drl_overshoot_history,
        drl_loss_per_interval,
        no_tilt_deg,
        position_history,
        ref_xy_history,
        drl_policy,
    )


def save_mobility_animation(cfg, out_dir, topology, position_history, ref_xy_history=None,
                            deviation_radius=None, member_group_idx=None):
    """Save the same style of position animation as verify_mobility.py.
    One frame per TILT CONTROL INTERVAL (each interval's first measurement
    draw), not per measurement draw -- 150,000 individual frames would make
    for an unusable animation; this keeps the frame count to
    num_tilt_control_intervals, each frame spanning tilt_control_interval_s.

    One circle per cluster (its exact deviation_radius, not a statistical
    fit to its few member positions) is drawn and updated from
    ref_xy_history if given (RPGM only). UEs are colored by their cluster's
    (site, sector-within-site) combination -- a proper map coloring, so any
    two geometrically adjacent cells (same site/different sector, or
    different but neighboring sites) always get different colors -- computed
    once from the initial cluster positions and held fixed for the whole
    animation, rather than per-point group coloring (NUM_GROUPS=60 is far
    more clusters than a colormap can distinguish) or recomputing every
    frame (which would make colors flicker as clusters cross boundaries).
    """
    mobility_model = cfg["mobility_model"]
    tilt_control_interval_s = cfg["tilt_control_interval_s"]
    num_tilt_control_intervals = cfg["num_tilt_control_intervals"]
    animation_fps = cfg.get("animation_fps", 10)

    fig = topology.grid.show(show_sectors=True)
    ax = fig.gca()
    # grid.show()'s own layout leaves no room for a title added afterward;
    # FuncAnimation.save() bakes in a FIXED canvas (animations can't use
    # bbox_inches="tight" the way a static savefig can, since every frame
    # must share one frame size), so an unreserved title just gets clipped
    # by the top of that fixed canvas -- reserve the room explicitly.
    fig.subplots_adjust(top=0.90)
    xy0 = position_history[0, :, :2]

    colors = None
    if ref_xy_history is not None and member_group_idx is not None:
        from .scenario_view import compute_cell_colors
        cluster_colors, _ = compute_cell_colors(ref_xy_history[0], topology.site_loc, topology.num_sectors_per_site)
        colors = [cluster_colors[g] for g in member_group_idx.tolist()]
    scatter = ax.scatter(xy0[:, 0], xy0[:, 1], marker="x", color=colors if colors is not None else "tab:red")

    ellipses = None
    if ref_xy_history is not None:
        from .scenario_view import add_cluster_ellipses
        ellipses = add_cluster_ellipses(ax, ref_xy_history[0], deviation_radius)

    def update(frame):
        scatter.set_offsets(position_history[frame, :, :2])
        artists = [scatter]
        if ellipses is not None:
            for patch, (cx, cy) in zip(ellipses, ref_xy_history[frame]):
                patch.set_center((float(cx), float(cy)))
            artists.extend(ellipses)
        elapsed_s = frame * tilt_control_interval_s
        ax.set_title(f"{mobility_model}: tilt control interval {frame}, elapsed time {elapsed_s:g} s")
        return artists

    animation = FuncAnimation(fig, update, frames=num_tilt_control_intervals, blit=False)
    path = os.path.join(out_dir, "mobility_animation.gif")
    animation.save(path, writer=PillowWriter(fps=animation_fps))
    plt.close(fig)
    print(f"Saved: {path}")
