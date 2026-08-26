"""The environment + algorithm side of the dynamic tilt-control comparison:
SimulationEngine owns everything -- topology/mobility/channel setup,
environment stepping (mobility + channel state, no algorithm knowledge),
and all five methods' (Dynamic Local Oracle/Causal, Adaptive Legacy, No
Tilt, DRL) per-interval decisions.

__init__(cfg) builds the whole world (topology, mobility, channel, KpiManager)
and every controller from just a merged config dict; run_simulation() is
only the per-interval loop -- everything it needs is self.*, no nested
functions (every earlier closure here is now a method, sharing state via
self instead of closure capture). The mobility animation itself lives in
helpers.utils.save_mobility_animation, alongside this project's other
I/O-adjacent helpers (config loading, topology/UE visualization).
KpiManager (helpers/kpi_manager.py) sits in between: a pure function of
(state, tilt) to power/KPIs, plus this run's within-interval pooling.
"""

import numpy as np
import torch

from sionna.phy.channel.tr38901 import Antenna, AntennaArray, RMa, UMa, UMi
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from sionna.phy.constants import BOLTZMANN_CONSTANT

from drl.factory import create_policy
from drl.interface import compute_reward, compute_state, num_features_for, tilt_idx_to_deg
from drl.scheduler import SectorTiltControlScheduler
from .cellular_topology import CellularTopology
from .electrical_downtilt import ElectricalDowntilt
from .kpi_manager import IntervalMeasurementPool, KpiManager
from .large_scale_channel import LargeScaleChannel
from .mobility import RandomWalkMobility, ReferencePointGroupMobility
from .spatial_occupancy_estimator import CausalSpatialOccupancyEstimator, PredictedSpatialOccupancyEstimator
from .spatial_radio_state_estimator import CausalSpatialRadioStateEstimator, PredictedSpatialRadioStateEstimator
from .tilt_controller import AdaptiveLegacyTiltController, DynamicTiltController, LocalTiltSelector, \
    RLTiltController
from .ue_drop import UeDropper

_SCENARIO_CHANNEL_MODEL = {"umi": UMi, "uma": UMa, "rma": RMa}

# No Tilt: fixed boresight (0 deg) on every sector, forever
NO_TILT_DEG = 0.0


class SimulationEngine:
    """Evaluate Dynamic Local Oracle, Dynamic Local Causal, Adaptive Legacy,
    No Tilt, and DRL -- against the same per-CONTROL-INTERVAL pooled data,
    every interval. __init__ does all setup (topology/mobility/channel,
    config, history arrays, controllers); run_simulation() is only the
    per-interval loop.
    """

    def __init__(self, cfg):
        # Load configs -- config.yaml is grouped by topic; pull each
        # section once, then read off of it below.
        topology_cfg = cfg["topology"]
        antenna_cfg = cfg["antenna"]
        channel_cfg = cfg["channel"]
        kpi_cfg = cfg["kpi"]
        mobility_cfg = cfg["mobility"]
        simulation_cfg = cfg["simulation"]
        optimization_cfg = cfg["algorithms"]["optimization"]
        adaptive_legacy_cfg = cfg["algorithms"]["adaptive_legacy"]
        drl_cfg = cfg["algorithms"]["drl"]

        scenario = topology_cfg["scenario"]
        num_rings = topology_cfg["num_rings"]
        num_ut = topology_cfg["num_ut"]
        ut_height = topology_cfg["ut_height"]
        m = antenna_cfg["m"]
        antenna_window = antenna_cfg["antenna_window"]
        bs_tx_power_dbm = channel_cfg["bs_tx_power_dbm"]
        channel_bandwidth_prb = channel_cfg["channel_bandwidth_prb"]
        temperature = channel_cfg["temperature"]
        carrier_frequency = channel_cfg["carrier_frequency"]
        environment_seed = simulation_cfg["environment_seed"]
        mobility_model = mobility_cfg["mobility_model"]
        min_ut_speed = mobility_cfg["min_ut_speed"]
        max_ut_speed = mobility_cfg["max_ut_speed"]
        num_groups = mobility_cfg["num_groups"]
        deviation_radius_frac_area = mobility_cfg["deviation_radius_frac_area"]
        member_jitter_speed = mobility_cfg["member_jitter_speed"]
        num_realizations_per_slot = simulation_cfg["num_realizations_per_slot"]
        max_realization_cuda = simulation_cfg["max_realization_cuda"]

        self.downtilt_sweep_deg = np.asarray(antenna_cfg["downtilt_sweep_deg"])
        self.coverage_threshold_db = kpi_cfg["coverage_threshold_db"]
        self.search_enabled = optimization_cfg["enabled"]
        self.coordinate_ascent_max_rounds = optimization_cfg["coordinate_ascent_max_rounds"]
        self.sinr_percentiles = kpi_cfg["sinr_percentiles"]
        self.adaptive_legacy_overlap_margin_db = adaptive_legacy_cfg["overlap_margin_db"]
        self.adaptive_legacy_edge_percentile = adaptive_legacy_cfg["edge_percentile"]
        self.adaptive_legacy_tilt_step_deg = adaptive_legacy_cfg["tilt_step_deg"]
        self.adaptive_legacy_os_threshold = adaptive_legacy_cfg["os_threshold"]
        self.adaptive_legacy_bc_threshold = adaptive_legacy_cfg["bc_threshold"]
        self.num_ut = num_ut
        self.num_tilt_control_intervals = simulation_cfg["num_tilt_control_intervals"]
        self.measurement_interval_s = simulation_cfg["measurement_interval_s"]
        self.tilt_control_interval_s = simulation_cfg["tilt_control_interval_s"]
        self.max_realization_cuda = max_realization_cuda
        self.policy_name = drl_cfg["policy_name"]
        self.algorithm_seed = drl_cfg["algorithm_seed"]
        self.state_type = drl_cfg["state_type"]
        self.reward_type = drl_cfg["reward_type"]
        self.reward_lambda_coverage = drl_cfg["reward_lambda_coverage"]
        self.reward_lambda_overshoot = drl_cfg["reward_lambda_overshoot"]
        self.dqn_kwargs = dict(drl_cfg["dqn"])
        self.async_schedule = drl_cfg["async_schedule"]
        self.async_group_size = drl_cfg["async_group_size"]

        # Assertion check
        assert num_ut % num_groups == 0, \
            "num_ut must be divisible by num_groups for equal-sized RPGM groups"
        assert mobility_model in ("rpgm", "random_walk"), \
            "mobility_model must be 'rpgm' or 'random_walk'"
        assert num_realizations_per_slot % max_realization_cuda == 0, \
            "num_realizations_per_slot must be a multiple of max_realization_cuda"
        assert abs(self.tilt_control_interval_s / self.measurement_interval_s
                  - round(self.tilt_control_interval_s / self.measurement_interval_s)) < 1e-9, \
            "tilt_control_interval_s must be an exact multiple of measurement_interval_s"
        self.measurement_slots_per_interval = round(self.tilt_control_interval_s / self.measurement_interval_s)
        members_per_group = num_ut // num_groups

        # Set scenario params
        min_bs_ut_dist, isd, bs_height, min_ut_height, _, _, _, _ = \
            set_3gpp_scenario_parameters(scenario)
        scenario_params = {
            "isd": isd,
            "bs_height": bs_height,
            "min_bs_ut_dist": min_bs_ut_dist,
            "min_ut_height": min_ut_height,
        }

        # Cellular topology
        self.topology = CellularTopology(scenario_params, num_rings=num_rings, batch_size=max_realization_cuda)
        self.num_bs = self.topology.num_bs
        self.distance_threshold_m = float(self.topology.grid.cell_radius)

        # Initial locations and mobility models
        self.mobility_list = []
        for i in range(num_realizations_per_slot):
            # Environment RNG
            environment_generator = torch.Generator(device=self.topology.bs_loc.device)
            environment_generator.manual_seed(environment_seed + i)

            sampler = UeDropper(self.topology, generator=environment_generator)
            if mobility_model == "rpgm":
                start_xy_list = sampler.cluster_centers(num_groups)
                deviation_radius = deviation_radius_frac_area * self.topology.default_drop_radius
                initial_ut_loc, member_group_idx = sampler.clustered(
                    start_xy_list, members_per_group, deviation_radius, ut_height,
                )
                mobility = ReferencePointGroupMobility(
                    initial_ut_loc, member_group_idx, start_xy_list,
                    deviation_radius=deviation_radius, topo=self.topology,
                    min_speed=min_ut_speed, max_speed=max_ut_speed,
                    member_jitter_speed=member_jitter_speed,
                    generator=environment_generator,
                )
            else:
                initial_ut_loc = sampler.uniform(num_ut, ut_height)
                mobility = RandomWalkMobility(
                    initial_ut_loc, topo=self.topology,
                    min_speed=min_ut_speed, max_speed=max_ut_speed,
                    generator=environment_generator,
                )
            self.mobility_list.append(mobility)

        # BS and UE Antenna array
        bs_array = AntennaArray(
            num_rows=m, num_cols=1, polarization="single", polarization_type="V",
            antenna_pattern="38.901", carrier_frequency=carrier_frequency,
        )
        ut_array = Antenna(
            polarization="single", polarization_type="V",
            antenna_pattern="omni", carrier_frequency=carrier_frequency,
        )

        # Electrical downtilts
        sector_etilts = [
            ElectricalDowntilt(bs_array, carrier_frequency=carrier_frequency,
                              downtilt_deg=self.downtilt_sweep_deg[0], window=antenna_window)
            for _ in range(self.num_bs)
        ]

        # Channel model
        channel_model_cls = _SCENARIO_CHANNEL_MODEL[scenario]
        channel_model_kwargs = dict(
            carrier_frequency=carrier_frequency, bs_array=bs_array, ut_array=ut_array,
            direction="downlink", enable_pathloss=True, enable_shadow_fading=True,
        )
        if scenario != "rma":
            channel_model_kwargs["o2i_model"] = "low"
        self.channel_model = channel_model_cls(**channel_model_kwargs)
        self.large_scale_channel = LargeScaleChannel(self.channel_model)

        # Tx/noise power
        bs_tx_power_w = 10 ** ((bs_tx_power_dbm - 30.0) / 10.0)
        channel_bandwidth_hz = channel_bandwidth_prb * 12 * 30e3
        noise_power_w = BOLTZMANN_CONSTANT * temperature * channel_bandwidth_hz

        # Sites/sectors share the same (x, y) geometry
        bs_xy = self.topology.bs_loc[0, :, :2].detach()

        # KPI manager -- adjacency is pure geometry, computed by CellularTopology
        self.kpi_manager = KpiManager(sector_etilts, bs_tx_power_w, noise_power_w, bs_xy,
                                      self.topology.neighbor_ids, self.topology.max_neighbors,
                                      self.topology.sector_adjacency)
        self.measurement_pool = IntervalMeasurementPool(self.kpi_manager, self.num_bs)

        # UT orientation and mobility
        self.ut_orientations = torch.zeros(max_realization_cuda, num_ut, 3,
                                           dtype=self.topology.bs_loc.dtype, device=self.topology.bs_loc.device)
        self.ut_velocities = torch.zeros_like(self.ut_orientations)
        self.in_state = torch.zeros(max_realization_cuda, num_ut, dtype=torch.bool, device=self.topology.bs_loc.device)

        # Initialization of KPI history arrays. 
        n = self.num_tilt_control_intervals
        self.coverage_dynamic_local_oracle = np.full(n, np.nan)
        self.coverage_dynamic_local_causal = np.full(n, np.nan)
        self.coverage_adaptive_legacy = np.zeros(n)
        self.coverage_no_tilt = np.zeros(n)
        self.coverage_drl = np.zeros(n)
        self.sinr_percentiles_dynamic_local_oracle = np.full((len(self.sinr_percentiles), n), np.nan)
        self.sinr_percentiles_dynamic_local_causal = np.full((len(self.sinr_percentiles), n), np.nan)
        self.sinr_percentiles_adaptive_legacy = np.zeros((len(self.sinr_percentiles), n))
        self.sinr_percentiles_no_tilt = np.zeros((len(self.sinr_percentiles), n))
        self.sinr_percentiles_drl = np.zeros((len(self.sinr_percentiles), n))
        self.dynamic_local_oracle_tilt_deg_history = np.full((n, self.num_bs), np.nan)
        self.dynamic_local_causal_tilt_deg_history = np.full((n, self.num_bs), np.nan)
        self.adaptive_legacy_tilt_deg_history = np.zeros((n, self.num_bs))
        self.drl_tilt_deg_history = np.zeros((n, self.num_bs))

        # DRL-only
        self.drl_reward_history = np.zeros((n, self.num_bs))
        self.drl_overshoot_history = np.zeros((n, self.num_bs))
        self.drl_loss_per_interval = np.full(n, np.nan)
        self.position_history = np.zeros((n, self.num_ut, 3))

        self.has_clusters = hasattr(self.mobility_list[0], "ref_xy")
        self.ref_xy_history = (np.zeros((n, self.mobility_list[0].ref_xy.shape[0], 2))
                               if self.has_clusters else None)

        # Algorithms
        # Dynamic Local Oracle
        self.dynamic_local_oracle_controller = DynamicTiltController(
            LocalTiltSelector(max_rounds=self.coordinate_ascent_max_rounds))

        # Dynamic Local Causal
        self.dynamic_local_causal_controller = DynamicTiltController(
            LocalTiltSelector(max_rounds=self.coordinate_ascent_max_rounds))

        # Adaptive Legacy: starts from the same downtilt_sweep_deg[0] every
        # sector_etilt is physically initialized to, above
        self.adaptive_legacy_controller = AdaptiveLegacyTiltController(
            num_sectors=self.num_bs,
            initial_tilt_deg=self.downtilt_sweep_deg[0],
            theta_min_deg=self.downtilt_sweep_deg.min(),
            theta_max_deg=self.downtilt_sweep_deg.max(),
            distance_threshold_m=self.distance_threshold_m,
            coverage_threshold_db=self.coverage_threshold_db,
            overlap_margin_db=self.adaptive_legacy_overlap_margin_db,
            edge_percentile=self.adaptive_legacy_edge_percentile,
            tilt_step_deg=self.adaptive_legacy_tilt_step_deg,
            os_threshold=self.adaptive_legacy_os_threshold,
            bc_threshold=self.adaptive_legacy_bc_threshold,
        )

        # Per-sector rhombus spatial grid
        spatial_grid_cell_size_m = drl_cfg["spatial_grid_cell_size_m"]
        self.grid_points, self.grid_n = self.topology.build_sector_rhombus_grid(spatial_grid_cell_size_m)
        self.num_grid_cells = self.grid_n * self.grid_n
        self.tilt_deg_min = float(self.downtilt_sweep_deg.min())
        self.tilt_deg_max = float(self.downtilt_sweep_deg.max())
        self.top_k_locations = drl_cfg["top_k_locations"]

        # DRL
        self.drl_policy = create_policy(
            self.policy_name, self.num_bs,
            num_features=num_features_for(self.state_type, max_neighbors=self.kpi_manager.max_neighbors,
                                          top_k_locations=self.top_k_locations),
            num_actions=len(self.downtilt_sweep_deg), dqn_kwargs=self.dqn_kwargs,
            # CPU, not the pipeline default of cuda-if-available: measured
            # ~1.6x SLOWER on GPU for this workload (21 tiny 32-hidden-unit
            # networks, single-sample inference, batch_size=32 training) --
            # per-call kernel-launch/transfer overhead dominates at this
            # scale, leaving no compute for GPU parallelism to offset it.
            algorithm_seed=self.algorithm_seed, device="cpu",
        )
        self.drl_controller = RLTiltController(self.drl_policy, self.num_bs, initial_tilt_idx=0)

        # Per-sector tilt control decision schedule
        self.scheduler = SectorTiltControlScheduler(
            self.num_bs, self.measurement_slots_per_interval, self.async_schedule,
            self.async_group_size, self.algorithm_seed + 1_000_000,
        )

        self._interval_step_losses = []
        self._interval_last_reward = np.full(self.num_bs, np.nan)

        # Spatial occupancy + radio state estimators
        self.is_predicted_spatial = self.state_type == "predicted_top_k_neighbor"
        self.occupancy_estimator = CausalSpatialOccupancyEstimator(
            self.num_bs, self.num_grid_cells, self.measurement_slots_per_interval)
        self.radio_state_estimator = CausalSpatialRadioStateEstimator(
            self.num_bs, self.num_grid_cells, self.kpi_manager.max_neighbors,
            self.kpi_manager.neighbor_ids, self.top_k_locations)
        if self.is_predicted_spatial:
            self.predicted_occupancy_estimator = PredictedSpatialOccupancyEstimator(
                self.num_bs, self.num_grid_cells)
            self.predicted_radio_state_estimator = PredictedSpatialRadioStateEstimator(
                self.top_k_locations, self.num_grid_cells, self.grid_points,
                length_scale=spatial_grid_cell_size_m,
            )

    def drl_process_slot(self, drl_power_w_this_slot, ut_loc_this_slot, global_slot, live_tilt_deg):
        """per-measurement-slot orchestration loop for the DRL controller

        :param drl_power_w_this_slot: [num_bs, num_ut] resolved power at
            live_tilt_deg.
        :param ut_loc_this_slot: [num_ut, 3] this slot's UE positions.
        :param global_slot: cumulative measurement-slot index since the run started.
        :param live_tilt_deg: [num_bs] float, the tilt drl_power_w_this_slot
            was resolved at -- passed through to the predicted top-K
            estimator, which evaluates coverage at fixed cell coordinates
            under whatever's currently live.
        """
        # Compute UE KPIs
        kpis = self.kpi_manager.compute_ue_kpis(
            drl_power_w_this_slot[None, :, :], threshold_db=self.coverage_threshold_db,
            ut_loc=ut_loc_this_slot[None, :, :],
            distance_threshold_m=self.distance_threshold_m, overlap_margin_db=self.adaptive_legacy_overlap_margin_db,
            return_counts=True, return_per_neighbor=True,
        )

        # Accumulate sector coverage and overshoot
        self.measurement_pool.accumulate_sector_window(kpis)

        # Assign UEs to spatial grids
        sector_idx, cell_idx = self.topology.assign_to_sector_rhombus_grid(ut_loc_this_slot[:, :2])

        # Accumulate occupancy, spatial coverage and neighboring overshoot
        covered_ue = kpis["sinr_db"][0] > self.coverage_threshold_db
        self.occupancy_estimator.accumulate(sector_idx, cell_idx)
        self.radio_state_estimator.accumulate(
            sector_idx, cell_idx, covered_ue,
            kpis["per_neighbor_served_count"], kpis["per_neighbor_overshoot_hit_count"],
        )

        # Check if the window closes
        closes = self.scheduler.closes(global_slot)
        if not closes.any():
            return

        # Resolve interval KPIs
        coverage_per_sector, overshoot_per_sector = self.measurement_pool.resolve_sector_window(closes)
        has_data = ~(np.isnan(coverage_per_sector) | np.isnan(overshoot_per_sector))

        # Compute reward
        reward = compute_reward(
            self.reward_type, coverage_per_sector, overshoot_per_sector,
            self.reward_lambda_coverage, self.reward_lambda_overshoot,
        )
        self._interval_last_reward[closes] = reward[closes]

        # Compute spatial occupancy and radio state
        importance, visit_count = self.occupancy_estimator.compute(closes)
        top_k_identity, top_k_coverage, neighbor_overshoot, grid_coverage = self.radio_state_estimator.compute(
            closes, importance, visit_count)

        # If predicted mode enabled
        if self.is_predicted_spatial:
            forecasted_importance = self.predicted_occupancy_estimator.compute(closes, importance)
            top_k_identity, top_k_coverage, neighbor_overshoot = self.predicted_radio_state_estimator.compute(
                closes, forecasted_importance, grid_coverage, neighbor_overshoot)
        own_tilt_norm = (live_tilt_deg - self.tilt_deg_min) / (self.tilt_deg_max - self.tilt_deg_min)

        # Compute state
        observations = compute_state(
            self.state_type, own_tilt_norm=own_tilt_norm,
            neighbor_overshoot=neighbor_overshoot,
            top_k_identity=top_k_identity, top_k_coverage=top_k_coverage,
        )

        # Compute loss
        losses_before = len(self.drl_policy.step_losses)

        # Update RL controller
        self.drl_controller.update(observations, reward, training=True, has_data=has_data, schedule=closes)

        if len(self.drl_policy.step_losses) > losses_before:
            self._interval_step_losses.extend(self.drl_policy.step_losses[losses_before:])

    def step_environment(self, is_last_step):
        """Advances every realization's mobility by measurement_interval_s,
        then draws a fresh large-scale channel state per realization-chunk
        (batched by max_realization_cuda) -- returns a list of
        LargeScaleState, one per chunk. Positions are read for THIS
        (pre-step) instant; mobility only advances afterward (never past
        the very last measurement draw of the run).

        :output: list of LargeScaleState, one per realization-chunk.
        """
        num_realizations = len(self.mobility_list)
        num_chunks = num_realizations // self.max_realization_cuda
        states = []
        for chunk in range(num_chunks):
            # Mobility update
            chunk_mobility = self.mobility_list[chunk * self.max_realization_cuda:(chunk + 1) * self.max_realization_cuda]
            ut_loc = torch.stack([mobility.ut_loc for mobility in chunk_mobility], dim=0)
            bs_virtual_loc = self.topology.mirror_bs_loc(ut_loc)

            # Channel model update
            self.channel_model.set_topology(
                ut_loc, self.topology.bs_loc, self.ut_orientations, self.topology.bs_orientations,
                self.ut_velocities, self.in_state, None, bs_virtual_loc,
            )
            states.append(self.large_scale_channel.generate_state())

        if not is_last_step:
            for mobility in self.mobility_list:
                mobility.step(self.measurement_interval_s)

        return states

    def execute_algorithms(self, pooled, interval, drl_tilt_deg_this_interval):
        """Once per interval, given this interval's pooled KPI data: Oracle/
        Causal decide+score off the swept power table; Adaptive Legacy/DRL/
        No Tilt each score their own resolved power via KpiManager, Adaptive
        Legacy/DRL additionally getting overshoot/per-sector coverage back
        to drive their own decision update. Writes into self's pre-allocated
        history arrays at index `interval`.
        """
        pooled_adaptive_power_w_r0 = pooled["adaptive_power_w_r0"]
        pooled_drl_power_w_r0 = pooled["drl_power_w_r0"]
        pooled_no_tilt_power_w = pooled["no_tilt_power_w"]
        pooled_ut_loc_r0 = pooled["ut_loc_r0"]

        if self.search_enabled:
            pooled_power_table = pooled["power_table"]

            # Oracle: decide AND score using THIS interval's pooled power table -- no lag.
            dynamic_local_oracle_assignment = self.dynamic_local_oracle_controller.update(
                self.kpi_manager, pooled_power_table, self.coverage_threshold_db
            )
            oracle_kpis = self.kpi_manager.compute_ue_kpis(pooled_power_table, dynamic_local_oracle_assignment,
                                                            self.coverage_threshold_db)
            self.coverage_dynamic_local_oracle[interval] = oracle_kpis["coverage"]
            self.sinr_percentiles_dynamic_local_oracle[:, interval] = np.percentile(oracle_kpis["sinr_db"],
                                                                                    self.sinr_percentiles)
            self.dynamic_local_oracle_tilt_deg_history[interval] = self.downtilt_sweep_deg[dynamic_local_oracle_assignment]

            # Causal: score the assignment decided from data through the previous interval against
            # this interval's actual pooled power table.
            causal_applied_assignment = (
                self.dynamic_local_causal_controller.assignment
                if self.dynamic_local_causal_controller.assignment is not None
                else dynamic_local_oracle_assignment
            )
            causal_kpis = self.kpi_manager.compute_ue_kpis(pooled_power_table, causal_applied_assignment,
                                                            self.coverage_threshold_db)
            self.coverage_dynamic_local_causal[interval] = causal_kpis["coverage"]
            self.sinr_percentiles_dynamic_local_causal[:, interval] = np.percentile(causal_kpis["sinr_db"],
                                                                                    self.sinr_percentiles)
            self.dynamic_local_causal_tilt_deg_history[interval] = self.downtilt_sweep_deg[causal_applied_assignment]
            self.dynamic_local_causal_controller.update(self.kpi_manager, pooled_power_table, self.coverage_threshold_db)

        # Adaptive Legacy: score at the tilt already in effect, THEN update
        self.adaptive_legacy_tilt_deg_history[interval] = self.adaptive_legacy_controller.tilt_deg.copy()
        adaptive_kpis = self.kpi_manager.compute_ue_kpis(
            pooled_adaptive_power_w_r0[None, :, :], threshold_db=self.coverage_threshold_db,
            ut_loc=pooled_ut_loc_r0[None, :, :], edge_percentile=self.adaptive_legacy_edge_percentile,
            distance_threshold_m=self.distance_threshold_m, overlap_margin_db=self.adaptive_legacy_overlap_margin_db,
        )
        self.coverage_adaptive_legacy[interval] = adaptive_kpis["coverage"]
        self.sinr_percentiles_adaptive_legacy[:, interval] = np.percentile(adaptive_kpis["sinr_db"],
                                                                           self.sinr_percentiles)
        adaptive_has_data = ~(np.isnan(adaptive_kpis["overshoot"]) | np.isnan(adaptive_kpis["per_sector_coverage"]))
        self.adaptive_legacy_controller.update(adaptive_kpis["overshoot"], adaptive_kpis["per_sector_coverage"],
                                               has_data=adaptive_has_data)

        # No Tilt: fixed 0 degrees, forever -- just scored, never decided.
        no_tilt_kpis = self.kpi_manager.compute_ue_kpis(pooled_no_tilt_power_w, threshold_db=self.coverage_threshold_db)
        self.coverage_no_tilt[interval] = no_tilt_kpis["coverage"]
        self.sinr_percentiles_no_tilt[:, interval] = np.percentile(no_tilt_kpis["sinr_db"], self.sinr_percentiles)

        # DRL
        self.drl_tilt_deg_history[interval] = drl_tilt_deg_this_interval
        drl_kpis = self.kpi_manager.compute_ue_kpis(
            pooled_drl_power_w_r0[None, :, :], threshold_db=self.coverage_threshold_db,
            ut_loc=pooled_ut_loc_r0[None, :, :],
            distance_threshold_m=self.distance_threshold_m, overlap_margin_db=self.adaptive_legacy_overlap_margin_db,
        )
        self.coverage_drl[interval] = drl_kpis["coverage"]
        self.sinr_percentiles_drl[:, interval] = np.percentile(drl_kpis["sinr_db"], self.sinr_percentiles)
        self.drl_overshoot_history[interval] = drl_kpis["overshoot"]

        # The real reward each sector actually trained on this interval
        self.drl_reward_history[interval] = self._interval_last_reward
        self._interval_last_reward = np.full(self.num_bs, np.nan)

        if self._interval_step_losses:
            self.drl_loss_per_interval[interval] = np.mean(self._interval_step_losses)
        self._interval_step_losses = []

    def run_simulation(self) -> dict:
        """The per-interval loop: pool every measurement draw's contribution
        via step_environment() + kpi_manager, then execute_algorithms() once
        per interval on the pooled result.
        """
        global_slot = 0

        # For each tilt control interval
        for interval in range(self.num_tilt_control_intervals):
            is_last_interval = interval == self.num_tilt_control_intervals - 1
            drl_tilt_deg_this_interval = tilt_idx_to_deg(self.drl_controller.tilt_idx, self.downtilt_sweep_deg)

            self.measurement_pool.start_interval()
            ref_xy_snapshot = None

            # for each measurement slot
            for sub_step in range(self.measurement_slots_per_interval):
                # Mobility updates
                if sub_step == 0:
                    self.position_history[interval] = self.mobility_list[0].ut_loc.detach().cpu().numpy().copy()
                    if self.has_clusters:
                        ref_xy_snapshot = self.mobility_list[0].ref_xy.detach().cpu().numpy().copy()
                ut_loc_r0_this_draw = self.mobility_list[0].ut_loc.detach().cpu().numpy().copy()

                # Live per-sector tilt
                drl_tilt_deg_live = tilt_idx_to_deg(self.drl_controller.tilt_idx, self.downtilt_sweep_deg)

                is_last_step = is_last_interval and (sub_step == self.measurement_slots_per_interval - 1)

                # Step environment
                states = self.step_environment(is_last_step)
                for chunk_idx, state in enumerate(states):
                    # Pool measurements
                    drl_power_w_slot = self.measurement_pool.pool_measurement(
                        state, self.adaptive_legacy_controller.tilt_deg, drl_tilt_deg_live,
                        self.downtilt_sweep_deg, ut_loc_r0=ut_loc_r0_this_draw if chunk_idx == 0 else None,
                        needs_sweep=self.search_enabled,
                    )
                    # process DRL slot
                    if chunk_idx == 0:
                        self.drl_process_slot(drl_power_w_slot, ut_loc_r0_this_draw, global_slot, drl_tilt_deg_live)

                # Increment slot
                global_slot += 1

            if self.has_clusters:
                self.ref_xy_history[interval] = ref_xy_snapshot

            # Resolve pooled interval
            pooled = self.measurement_pool.pooled_interval()

            # Execute and evaluate algorithms
            self.execute_algorithms(pooled, interval, drl_tilt_deg_this_interval)

            # log results
            if interval % 10 == 0 or is_last_interval:
                fields = []
                if self.search_enabled:
                    fields.append(f"dynamic_local_oracle={self.coverage_dynamic_local_oracle[interval]:.3f}")
                    fields.append(f"dynamic_local_causal={self.coverage_dynamic_local_causal[interval]:.3f}")
                fields.append(f"adaptive_legacy={self.coverage_adaptive_legacy[interval]:.3f}")
                fields.append(f"no_tilt={self.coverage_no_tilt[interval]:.3f}")
                fields.append(f"drl={self.coverage_drl[interval]:.3f}")
                print(f"interval {interval:3d}/{self.num_tilt_control_intervals - 1}: " + ", ".join(fields))

        no_tilt_deg = np.full(self.num_bs, NO_TILT_DEG)

        # Return all recorded results
        return {
            "coverage_dynamic_local_oracle": self.coverage_dynamic_local_oracle,
            "coverage_dynamic_local_causal": self.coverage_dynamic_local_causal,
            "coverage_adaptive_legacy": self.coverage_adaptive_legacy,
            "coverage_no_tilt": self.coverage_no_tilt,
            "coverage_drl": self.coverage_drl,
            "sinr_percentiles_dynamic_local_oracle": self.sinr_percentiles_dynamic_local_oracle,
            "sinr_percentiles_dynamic_local_causal": self.sinr_percentiles_dynamic_local_causal,
            "sinr_percentiles_adaptive_legacy": self.sinr_percentiles_adaptive_legacy,
            "sinr_percentiles_no_tilt": self.sinr_percentiles_no_tilt,
            "sinr_percentiles_drl": self.sinr_percentiles_drl,
            "dynamic_local_oracle_tilt_deg_history": self.dynamic_local_oracle_tilt_deg_history,
            "dynamic_local_causal_tilt_deg_history": self.dynamic_local_causal_tilt_deg_history,
            "adaptive_legacy_tilt_deg_history": self.adaptive_legacy_tilt_deg_history,
            "drl_tilt_deg_history": self.drl_tilt_deg_history,
            "drl_reward_history": self.drl_reward_history,
            "drl_overshoot_history": self.drl_overshoot_history,
            "drl_loss_per_interval": self.drl_loss_per_interval,
            "no_tilt_deg": no_tilt_deg,
            "position_history": self.position_history,
            "ref_xy_history": self.ref_xy_history,
            "drl_policy": self.drl_policy,
        }
