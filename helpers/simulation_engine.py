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
from drl.interface import compute_region_state, compute_reward, num_features_for, tilt_idx_to_deg
from drl.scheduler import SectorTiltControlScheduler
from .cellular_topology import CellularTopology
from .electrical_downtilt import ElectricalDowntilt
from .kpi_manager import IntervalMeasurementPool, KpiManager
from .large_scale_channel import LargeScaleChannel
from .mobility import ReferencePointGroupMobility
from .regional_info_provider import RegionalInfoProvider
from .spatial_grid_estimator import SpatialGridEstimator
from .tilt_controller import AdaptiveLegacyTiltController, DynamicTiltController, LocalTiltSelector, \
    RLTiltController
from .ue_drop import UeDropper
from .utils import get_logger

logger = get_logger(__name__)

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

    def __init__(self, cfg, topology=None):
        """
        :param topology: optional pre-built CellularTopology -- if given,
            used as-is instead of building one from cfg["topology"] (e.g. a
            truncated single-site topology for a small-scale diagnostic).
        """
        # Load configs -- config.yaml is grouped by topic; pull each
        # section once, then read off of it below.
        topology_cfg = cfg["topology"]
        antenna_cfg = cfg["antenna"]
        downtilt_cfg = cfg["downtilt"]
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
        min_ut_speed = mobility_cfg["min_ut_speed"]
        max_ut_speed = mobility_cfg["max_ut_speed"]
        num_groups = mobility_cfg["num_groups"]
        deviation_radius_frac_area = mobility_cfg["deviation_radius_frac_area"]
        cluster_radius_max_m = mobility_cfg["cluster_radius_max_m"]
        member_jitter_speed = mobility_cfg["member_jitter_speed"]
        cluster_mobility_mode = mobility_cfg["cluster_mobility_mode"]
        transition = mobility_cfg["transition"]
        num_waypoints = mobility_cfg["num_waypoints"]
        waypoint_hold_intervals = mobility_cfg["waypoint_hold_steps"]
        waypoint_transition_matrix = mobility_cfg["waypoint_transition_matrix"]
        waypoint_type = mobility_cfg["waypoint_type"]
        intra_cluster_mobility = mobility_cfg["intra_cluster_mobility"]
        num_realizations_per_slot = simulation_cfg["num_realizations_per_slot"]
        max_realization_cuda = simulation_cfg["max_realization_cuda"]

        self.downtilt_sweep_deg = np.arange(
            downtilt_cfg["downtilt_sweep_start_deg"],
            downtilt_cfg["downtilt_sweep_end_deg"] + downtilt_cfg["tilt_step_deg"],
            downtilt_cfg["tilt_step_deg"],
        )
        self.coverage_threshold_db = kpi_cfg["coverage_threshold_db"]
        self.search_enabled = optimization_cfg["enabled"]
        self.run_causal = optimization_cfg["run_causal"]
        self.coordinate_ascent_max_rounds = optimization_cfg["coordinate_ascent_max_rounds"]
        self.sinr_percentiles = kpi_cfg["sinr_percentiles"]
        self.adaptive_legacy_overlap_margin_db = adaptive_legacy_cfg["overlap_margin_db"]
        self.adaptive_legacy_edge_percentile = adaptive_legacy_cfg["edge_percentile"]
        self.adaptive_legacy_tilt_step_deg = downtilt_cfg["tilt_step_deg"]
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
        self.coverage_fill = drl_cfg["coverage_fill"]
        self.state_padding = drl_cfg.get("state_padding", "variable")
        self.occupancy_ema_alpha = drl_cfg["occupancy_ema_alpha"]
        self.dqn_kwargs = dict(drl_cfg["dqn"])
        self.async_schedule = drl_cfg["async_schedule"]

        # Assertion check
        assert num_ut % num_groups == 0, \
            "num_ut must be divisible by num_groups for equal-sized RPGM groups"
        assert num_realizations_per_slot % max_realization_cuda == 0, \
            "num_realizations_per_slot must be a multiple of max_realization_cuda"
        assert abs(self.tilt_control_interval_s / self.measurement_interval_s
                  - round(self.tilt_control_interval_s / self.measurement_interval_s)) < 1e-9, \
            "tilt_control_interval_s must be an exact multiple of measurement_interval_s"
        self.measurement_slots_per_interval = round(self.tilt_control_interval_s / self.measurement_interval_s)
        members_per_group = num_ut // num_groups

        # waypoint_hold_steps is configured in TILT CONTROL INTERVALS (an
        # intuitive unit); mobility.py's ReferencePointGroupMobility counts
        # its own raw step() calls (one per measurement slot), so convert here.
        waypoint_hold_steps = waypoint_hold_intervals * self.measurement_slots_per_interval

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
        self.topology = topology if topology is not None else \
            CellularTopology(scenario_params, num_rings=num_rings, batch_size=max_realization_cuda)
        self.num_bs = self.topology.num_bs
        self.distance_threshold_m = float(self.topology.grid.cell_radius)

        # Per-sector rhombus grid + its 9 observation windows -- needed here
        # (ahead of the DRL section below, which also uses them) so
        # "central" waypoint mode can offer each cluster's home-sector
        # region centroids as hotspot candidates.
        spatial_grid_cell_size_m = drl_cfg["spatial_grid_cell_size_m"]
        self.grid_points, self.grid_n = self.topology.build_sector_rhombus_grid(spatial_grid_cell_size_m)
        self.num_grid_cells = self.grid_n * self.grid_n
        self.observation_windows = self.topology.compute_observation_windows(spatial_grid_cell_size_m)

        # region_aligned state: this sector's OWN rhombus grid pooled into
        # region_block_size x region_block_size blocks (num_own_regions,
        # fixed/identical for every sector -- see
        # helpers/regional_info_provider.py) + one coarse (occupancy,
        # coverage) pair per actual (2-4) side-sharing neighbor. Computed
        # once here regardless of state_type, since it's cheap relative to
        # everything else in __init__ and num_features_for needs it either way.
        self.region_block_size = drl_cfg["region_block_size"]
        self.num_own_regions = (self.grid_n // self.region_block_size) ** 2
        self.num_neighbors_per_sector = [int((ids >= 0).sum()) for ids in self.topology.neighbor_ids]
        # Cluster radius is capped at cluster_radius_max_m (asserted below),
        # so requiring every waypoint candidate -- cluster centers, "random"
        # auto-waypoints, and "central" region centroids alike -- to clear
        # min_bs_ut_dist + cluster_radius_max_m from the NEAREST site (not
        # just its own home site: a window can draw cells from a neighbor at
        # seams/corners) guarantees the WHOLE cluster disk stays outside the
        # exclusion zone, not just its center point. Not every window
        # qualifies (e.g. "corner_C", the site-center corner shared by
        # same-site siblings, sits at distance ~0 from its own BS by
        # construction) but WHICH ones do varies with a sector's real
        # geometry -- checked directly per window, not assumed by name.
        deviation_radius = deviation_radius_frac_area * self.topology.default_drop_radius
        assert deviation_radius <= cluster_radius_max_m, (
            f"deviation_radius ({deviation_radius:.2f}m, from deviation_radius_frac_area) exceeds "
            f"cluster_radius_max_m ({cluster_radius_max_m:.2f}m) -- the waypoint exclusion-zone buffer "
            "assumes clusters never grow past this cap")
        waypoint_exclusion_radius = float(self.topology.min_bs_ut_dist) + cluster_radius_max_m
        region_center_window_names = list(self.observation_windows[0].keys())
        site_loc = self.topology.site_loc.cpu().numpy()
        region_centers = np.zeros((self.num_bs, len(region_center_window_names), 2))
        region_centers_valid = np.zeros((self.num_bs, len(region_center_window_names)), dtype=bool)
        for s in range(self.num_bs):
            for w, wname in enumerate(region_center_window_names):
                members = self.observation_windows[s][wname]
                pts = np.array([self.grid_points[sec, cell] for sec, cell in members])
                centroid = pts.mean(axis=0)
                region_centers[s, w] = centroid
                nearest_site_dist = np.linalg.norm(site_loc - centroid[None, :], axis=1).min()
                region_centers_valid[s, w] = nearest_site_dist >= waypoint_exclusion_radius

        # Initial locations and mobility models
        self.mobility_list = []
        for i in range(num_realizations_per_slot):
            # Environment RNG
            environment_generator = torch.Generator(device=self.topology.bs_loc.device)
            environment_generator.manual_seed(environment_seed + i)

            sampler = UeDropper(self.topology, generator=environment_generator)
            start_xy_list = sampler.cluster_centers(num_groups, min_dist_from_site=waypoint_exclusion_radius)
            initial_ut_loc, member_group_idx = sampler.clustered(
                start_xy_list, members_per_group, deviation_radius, ut_height,
            )
            start_xy_arr = np.asarray(start_xy_list, dtype=float)
            home_sector, _ = self.topology.assign_to_sector_rhombus_grid(start_xy_arr)
            mobility = ReferencePointGroupMobility(
                initial_ut_loc, member_group_idx, start_xy_list,
                deviation_radius=deviation_radius, topo=self.topology,
                min_speed=min_ut_speed, max_speed=max_ut_speed,
                member_jitter_speed=member_jitter_speed,
                generator=environment_generator,
                cluster_mobility_mode=cluster_mobility_mode, transition=transition,
                num_waypoints=num_waypoints, waypoint_hold_steps=waypoint_hold_steps,
                waypoint_transition_matrix=waypoint_transition_matrix,
                intra_cluster_mobility=intra_cluster_mobility,
                waypoint_type=waypoint_type, region_centers=region_centers,
                region_centers_valid=region_centers_valid, cluster_home_sector=home_sector,
                waypoint_min_dist_from_site=waypoint_exclusion_radius,
                transition_duration_s=self.tilt_control_interval_s,
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

        # Per-sector rhombus spatial grid + its 9 observation windows were
        # already built earlier (ahead of mobility construction, for
        # "central" waypoint mode) -- see self.grid_points/observation_windows.
        self.tilt_deg_min = float(self.downtilt_sweep_deg.min())
        self.tilt_deg_max = float(self.downtilt_sweep_deg.max())

        # DRL
        num_features = num_features_for(
            self.state_type, state_padding=self.state_padding,
            num_own_regions=self.num_own_regions,
            num_neighbors_per_sector=self.num_neighbors_per_sector,
        )
        self.drl_policy = create_policy(
            self.policy_name, self.num_bs, num_features=num_features,
            num_actions=len(self.downtilt_sweep_deg), dqn_kwargs=self.dqn_kwargs,
            # CPU, not the pipeline default of cuda-if-available: measured
            # ~1.6x SLOWER on GPU for this workload (21 tiny 32-hidden-unit
            # networks, single-sample inference, batch_size=32 training) --
            # per-call kernel-launch/transfer overhead dominates at this
            # scale, leaving no compute for GPU parallelism to offset it.
            algorithm_seed=self.algorithm_seed, device="cpu",
        )
        variable_size = self.state_type == "region_aligned" and self.state_padding == "variable"
        self.drl_controller = RLTiltController(self.drl_policy, self.num_bs, initial_tilt_idx=0,
                                               variable_size=variable_size)

        # Per-sector tilt control decision schedule -- "async" colors
        # sector_adjacency so adjacent sectors never decide simultaneously
        # (their local reward/state would otherwise be unattributable to
        # either one's own action specifically).
        self.scheduler = SectorTiltControlScheduler(
            self.num_bs, self.measurement_slots_per_interval, self.async_schedule,
            sector_adjacency=self.topology.sector_adjacency,
        )

        self._interval_step_losses = []
        self._interval_last_reward = np.full(self.num_bs, np.nan)

        # region_aligned state + sector_neighbors reward, both over X_s =
        # sector + its actual neighbors -- see helpers/regional_info_provider.py.
        # state_type/reward_type are validated by num_features_for/
        # compute_reward -- region_aligned/sector_neighbors are the only
        # supported values.
        self.regional_info_provider = RegionalInfoProvider(
            self.topology.neighbor_ids, self.grid_n, self.region_block_size,
            occupancy_ema_alpha=self.occupancy_ema_alpha, coverage_fill=self.coverage_fill)

        # coverage_fill="kriged" only -- fine-grid per-cell coverage,
        # RSRP-based (Krige each sector's own RSRP map, reconstruct SINR
        # from ALL sectors' predictions -- see spatial_grid_estimator.py
        # module docstring), that regional_info_provider aggregates over
        # each region's own member cells.
        self.spatial_grid_estimator = None
        if self.coverage_fill == "kriged":
            self.spatial_grid_estimator = SpatialGridEstimator(
                self.num_bs, self.num_grid_cells, self.grid_points,
                length_scale=spatial_grid_cell_size_m, coverage_threshold_db=self.coverage_threshold_db,
                radio_map_method="rsrp_per_sector", noise_power_w=noise_power_w,
                occupancy_ema_alpha=self.occupancy_ema_alpha,
            )

        logger.info("SimulationEngine constructed: num_bs=%d num_ut=%d num_tilt_control_intervals=%d "
                   "policy_name=%s state_type=%s async_schedule=%s", self.num_bs, self.num_ut,
                   self.num_tilt_control_intervals, self.policy_name, self.state_type, self.async_schedule)

    def drl_process_slot(self, drl_power_w_this_slot, ut_loc_this_slot, global_slot, live_tilt_deg):
        """per-measurement-slot orchestration loop for the DRL controller

        :param drl_power_w_this_slot: [num_bs, num_ut] resolved power at
            live_tilt_deg.
        :param ut_loc_this_slot: [num_ut, 3] this slot's UE positions.
        :param global_slot: cumulative measurement-slot index since the run started.
        :param live_tilt_deg: [num_bs] float, the tilt drl_power_w_this_slot
            was resolved at -- this slot's own_tilt/neighbor_tilt state
            features reflect whatever's currently live.
        """
        logger.function("drl_process_slot start: global_slot=%d", global_slot)

        # Compute UE KPIs -- only sinr_db is needed here (no threshold_db/
        # ut_loc/overshoot machinery): coverage is a per-UE indicator
        # compared against coverage_threshold_db directly below, and reward
        # is entirely geographic-grid-based, not served-sector-based.
        kpis = self.kpi_manager.compute_ue_kpis(drl_power_w_this_slot[None, :, :])
        covered_ue = kpis["sinr_db"][0] > self.coverage_threshold_db

        # Assign UEs to spatial grids (geographic, independent of serving sector)
        sector_idx, cell_idx = self.topology.assign_to_sector_rhombus_grid(ut_loc_this_slot[:, :2])
        self.regional_info_provider.accumulate(sector_idx, cell_idx, covered_ue)
        if self.spatial_grid_estimator is not None:
            power_db_all_sectors = 10.0 * np.log10(drl_power_w_this_slot)
            self.spatial_grid_estimator.accumulate(sector_idx, cell_idx, covered_ue,
                                                   power_db_all_sectors=power_db_all_sectors)

        # Check if the window closes
        closes = self.scheduler.closes(global_slot)
        if not closes.any():
            logger.function("drl_process_slot end: no sector closed")
            return
        logger.debug("drl_process_slot: %d sector(s) closed", int(closes.sum()))

        coverage_per_cell = None
        if self.spatial_grid_estimator is not None:
            _, coverage_per_cell, _ = self.spatial_grid_estimator.compute(closes)
        own_occupancy, own_coverage, neighbor_occupancy, neighbor_coverage, reward_per_sector, state_ready = (
            self.regional_info_provider.compute(closes, coverage_per_cell=coverage_per_cell))
        # A sector's transition is only real training data once BOTH its
        # reward is defined (real UEs in its own X_s this window) AND its
        # state is fully defined (coverage_fill="kriged": every sector in
        # its X_s has a valid completed coverage snapshot -- see
        # RegionalInfoProvider.compute's own docstring for why a partially
        # ready region must be excluded rather than silently averaged over
        # fewer members).
        has_data = ~np.isnan(reward_per_sector) & state_ready
        reward = compute_reward(self.reward_type, reward_per_sector)
        self._interval_last_reward[closes] = reward[closes]

        own_tilt_norm = (live_tilt_deg - self.tilt_deg_min) / (self.tilt_deg_max - self.tilt_deg_min)
        neighbor_tilts_norm = [own_tilt_norm[ids[ids >= 0]] for ids in self.topology.neighbor_ids]
        observations = compute_region_state(
            own_tilt_norm=own_tilt_norm, neighbor_tilts_norm=neighbor_tilts_norm,
            own_occupancy=own_occupancy, own_coverage=own_coverage,
            neighbor_occupancy=neighbor_occupancy, neighbor_coverage=neighbor_coverage,
            state_padding=self.state_padding,
        )
        logger.debug("drl_process_slot: region_aligned observation widths=%s",
                    [len(o) for o in observations] if isinstance(observations, list) else observations.shape)

        # Compute loss
        losses_before = len(self.drl_policy.step_losses)

        # Update RL controller
        self.drl_controller.update(observations, reward, training=True, has_data=has_data, schedule=closes)

        if len(self.drl_policy.step_losses) > losses_before:
            self._interval_step_losses.extend(self.drl_policy.step_losses[losses_before:])
        logger.function("drl_process_slot end")

    def step_environment(self, is_last_step):
        """Advances every realization's mobility by measurement_interval_s,
        then draws a fresh large-scale channel state per realization-chunk
        (batched by max_realization_cuda) -- returns a list of
        LargeScaleState, one per chunk. Positions are read for THIS
        (pre-step) instant; mobility only advances afterward (never past
        the very last measurement draw of the run).

        :output: list of LargeScaleState, one per realization-chunk.
        """
        logger.function("step_environment start: is_last_step=%s", is_last_step)
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
        logger.debug("step_environment: %d chunk(s) processed", num_chunks)

        if not is_last_step:
            for mobility in self.mobility_list:
                mobility.step(self.measurement_interval_s)

        logger.function("step_environment end")
        return states

    def execute_algorithms(self, pooled, interval, drl_tilt_deg_this_interval):
        """Once per interval, given this interval's pooled KPI data: Oracle/
        Causal decide+score off the swept power table; Adaptive Legacy/DRL/
        No Tilt each score their own resolved power via KpiManager, Adaptive
        Legacy/DRL additionally getting overshoot/per-sector coverage back
        to drive their own decision update. Writes into self's pre-allocated
        history arrays at index `interval`.
        """
        logger.function("execute_algorithms start: interval=%d", interval)
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
            logger.debug("execute_algorithms: oracle coverage=%.4f", oracle_kpis["coverage"])

            # Causal: score the assignment decided from data through the previous interval against
            # this interval's actual pooled power table. Skipped by default (run_causal=false).
            if self.run_causal:
                # No prior decision yet only on the very first interval --
                # fall back to a neutral index-0 (0 deg) assignment chosen
                # BEFORE this interval, not to the oracle's assignment for
                # THIS interval (that would score causal noncausally on its
                # first reported interval, since it's derived from the same
                # data being scored against).
                causal_applied_assignment = (
                    self.dynamic_local_causal_controller.assignment
                    if self.dynamic_local_causal_controller.assignment is not None
                    else np.zeros(self.num_bs, dtype=int)
                )
                causal_kpis = self.kpi_manager.compute_ue_kpis(pooled_power_table, causal_applied_assignment,
                                                                self.coverage_threshold_db)
                self.coverage_dynamic_local_causal[interval] = causal_kpis["coverage"]
                self.sinr_percentiles_dynamic_local_causal[:, interval] = np.percentile(causal_kpis["sinr_db"],
                                                                                        self.sinr_percentiles)
                self.dynamic_local_causal_tilt_deg_history[interval] = self.downtilt_sweep_deg[causal_applied_assignment]
                self.dynamic_local_causal_controller.update(self.kpi_manager, pooled_power_table, self.coverage_threshold_db)
                logger.debug("execute_algorithms: causal coverage=%.4f", causal_kpis["coverage"])

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
        logger.debug("execute_algorithms: adaptive_legacy coverage=%.4f", adaptive_kpis["coverage"])

        # No Tilt: fixed 0 degrees, forever -- just scored, never decided.
        no_tilt_kpis = self.kpi_manager.compute_ue_kpis(pooled_no_tilt_power_w, threshold_db=self.coverage_threshold_db)
        self.coverage_no_tilt[interval] = no_tilt_kpis["coverage"]
        self.sinr_percentiles_no_tilt[:, interval] = np.percentile(no_tilt_kpis["sinr_db"], self.sinr_percentiles)
        logger.debug("execute_algorithms: no_tilt coverage=%.4f", no_tilt_kpis["coverage"])

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
        logger.debug("execute_algorithms: drl coverage=%.4f", drl_kpis["coverage"])

        # The real reward each sector actually trained on this interval
        self.drl_reward_history[interval] = self._interval_last_reward
        self._interval_last_reward = np.full(self.num_bs, np.nan)

        if self._interval_step_losses:
            self.drl_loss_per_interval[interval] = np.mean(self._interval_step_losses)
        self._interval_step_losses = []
        logger.function("execute_algorithms end")

    def run_simulation(self, live_plot=None, checkpoint_path=None, checkpoint_every: int = 1) -> dict:
        """The per-interval loop: pool every measurement draw's contribution
        via step_environment() + kpi_manager, then execute_algorithms() once
        per interval on the pooled result.

        :param live_plot: optional helpers.utils.LiveMetricsPlot --
            saves a reward/loss-vs-interval PNG every interval, for watching
            a long run's training progress without waiting for it to finish.
        :param checkpoint_path: optional -- if given, overwrites this path
            with every recorded array (see _collect_results/_save_checkpoint)
            every checkpoint_every intervals, so e.g. per-sector
            drl_reward_history can be inspected while the run is still going,
            not just once it completes. Cheap relative to a whole interval's
            own compute, but still real disk I/O -- widen checkpoint_every
            if that starts to matter for a particular run's array sizes.
        """
        logger.function("run_simulation start: num_tilt_control_intervals=%d", self.num_tilt_control_intervals)
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

            # Live reward/loss plot
            if live_plot is not None:
                live_plot.update(interval, self.drl_reward_history[interval], self.drl_loss_per_interval[interval])

            # log results
            if interval % 10 == 0 or is_last_interval:
                fields = []
                if self.search_enabled:
                    fields.append(f"dynamic_local_oracle={self.coverage_dynamic_local_oracle[interval]:.3f}")
                    if self.run_causal:
                        fields.append(f"dynamic_local_causal={self.coverage_dynamic_local_causal[interval]:.3f}")
                fields.append(f"adaptive_legacy={self.coverage_adaptive_legacy[interval]:.3f}")
                fields.append(f"no_tilt={self.coverage_no_tilt[interval]:.3f}")
                fields.append(f"drl={self.coverage_drl[interval]:.3f}")
                logger.info(f"interval {interval:3d}/{self.num_tilt_control_intervals - 1}: " + ", ".join(fields))

            if checkpoint_path is not None and (interval % checkpoint_every == 0 or is_last_interval):
                self._save_checkpoint(checkpoint_path)

        logger.function("run_simulation end")
        return self._collect_results()

    def _collect_results(self) -> dict:
        """Every recorded result array, as of however far the run has
        gotten -- used both for run_simulation's own return value and for
        _save_checkpoint's mid-run snapshots (see run_simulation's
        checkpoint_path/checkpoint_every)."""
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
            "no_tilt_deg": np.full(self.num_bs, NO_TILT_DEG),
            "position_history": self.position_history,
            "ref_xy_history": self.ref_xy_history,
            "drl_policy": self.drl_policy,
        }

    def _save_checkpoint(self, path) -> None:
        """Overwrites `path` with every array-valued result recorded SO
        FAR -- lets a still-running simulation be inspected (e.g. per-sector
        drl_reward_history) without waiting for the full run to finish.
        drl_policy is excluded (not an array; still only saved once, by the
        caller, at actual completion, e.g. main.py's own Dqn.save())."""
        arrays = {k: v for k, v in self._collect_results().items() if isinstance(v, np.ndarray)}
        np.savez(path, **arrays)
        logger.debug("_save_checkpoint: wrote %d arrays to %s", len(arrays), path)
