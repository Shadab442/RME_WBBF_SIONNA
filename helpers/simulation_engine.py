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
from .cellular_topology import CellularTopology
from .electrical_downtilt import ElectricalDowntilt
from .kpi_manager import KpiManager
from .large_scale_channel import LargeScaleChannel
from .mobility import RandomWalkMobility, ReferencePointGroupMobility
from .tilt_controller import AdaptiveLegacyTiltController, DynamicTiltController, LocalTiltSelector, \
    RLTiltController
from .ue_drop import sample_cluster_center_across_sites, sample_clustered_ut_loc, sample_uniform_ut_loc

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

            if mobility_model == "rpgm":
                start_xy_list = sample_cluster_center_across_sites(self.topology, num_groups,
                                                                    generator=environment_generator)
                deviation_radius = deviation_radius_frac_area * self.topology.default_drop_radius
                initial_ut_loc, member_group_idx = sample_clustered_ut_loc(
                    self.topology, start_xy_list, members_per_group, deviation_radius, ut_height,
                    dtype=self.topology.bs_loc.dtype, device=self.topology.bs_loc.device,
                    generator=environment_generator,
                )
                mobility = ReferencePointGroupMobility(
                    initial_ut_loc, member_group_idx, start_xy_list,
                    deviation_radius=deviation_radius, topo=self.topology,
                    min_speed=min_ut_speed, max_speed=max_ut_speed,
                    member_jitter_speed=member_jitter_speed,
                    generator=environment_generator,
                )
            else:
                initial_ut_loc = sample_uniform_ut_loc(
                    self.topology, num_ut, ut_height,
                    dtype=self.topology.bs_loc.dtype, device=self.topology.bs_loc.device,
                    generator=environment_generator,
                )
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

        # KPI manager
        self.kpi_manager = KpiManager(sector_etilts, bs_tx_power_w, noise_power_w, bs_xy)

        # UT orientation and mobility
        self.ut_orientations = torch.zeros(max_realization_cuda, num_ut, 3,
                                           dtype=self.topology.bs_loc.dtype, device=self.topology.bs_loc.device)
        self.ut_velocities = torch.zeros_like(self.ut_orientations)
        self.in_state = torch.zeros(max_realization_cuda, num_ut, dtype=torch.bool, device=self.topology.bs_loc.device)

        # Initialization of KPI history arrays
        n = self.num_tilt_control_intervals
        self.coverage_dynamic_local_oracle = np.zeros(n)
        self.coverage_dynamic_local_causal = np.zeros(n)
        self.coverage_adaptive_legacy = np.zeros(n)
        self.coverage_no_tilt = np.zeros(n)
        self.coverage_drl = np.zeros(n)
        self.sinr_percentiles_dynamic_local_oracle = np.zeros((len(self.sinr_percentiles), n))
        self.sinr_percentiles_dynamic_local_causal = np.zeros((len(self.sinr_percentiles), n))
        self.sinr_percentiles_adaptive_legacy = np.zeros((len(self.sinr_percentiles), n))
        self.sinr_percentiles_no_tilt = np.zeros((len(self.sinr_percentiles), n))
        self.sinr_percentiles_drl = np.zeros((len(self.sinr_percentiles), n))
        self.dynamic_local_oracle_tilt_deg_history = np.zeros((n, self.num_bs))
        self.dynamic_local_causal_tilt_deg_history = np.zeros((n, self.num_bs))
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

        # DRL: index 0 into downtilt_sweep_deg (the same "as deployed"
        # starting point Adaptive Legacy uses)
        self.drl_policy = create_policy(
            self.policy_name, self.num_bs, num_features=num_features_for(self.state_type),
            num_actions=len(self.downtilt_sweep_deg), dqn_kwargs=self.dqn_kwargs,
            algorithm_seed=self.algorithm_seed, device="cpu",
        )
        self.drl_controller = RLTiltController(self.drl_policy, self.num_bs, initial_tilt_idx=0)

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
            chunk_mobility = self.mobility_list[chunk * self.max_realization_cuda:(chunk + 1) * self.max_realization_cuda]
            ut_loc = torch.stack([mobility.ut_loc for mobility in chunk_mobility], dim=0)
            bs_virtual_loc = self.topology.mirror_bs_loc(ut_loc)
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
        pooled_power_table = pooled["power_table"]
        pooled_adaptive_power_w_r0 = pooled["adaptive_power_w_r0"]
        pooled_drl_power_w_r0 = pooled["drl_power_w_r0"]
        pooled_no_tilt_power_w = pooled["no_tilt_power_w"]
        pooled_ut_loc_r0 = pooled["ut_loc_r0"]

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
        self.adaptive_legacy_controller.update(adaptive_kpis["overshoot"], adaptive_kpis["per_sector_coverage"])

        # No Tilt: fixed 0 degrees, forever -- just scored, never decided.
        no_tilt_kpis = self.kpi_manager.compute_ue_kpis(pooled_no_tilt_power_w, threshold_db=self.coverage_threshold_db)
        self.coverage_no_tilt[interval] = no_tilt_kpis["coverage"]
        self.sinr_percentiles_no_tilt[:, interval] = np.percentile(no_tilt_kpis["sinr_db"], self.sinr_percentiles)

        # DRL: score at the tilt already in effect for the coverage/SINR comparison,
        # THEN compute its own per-sector (coverage, overshoot) state/reward
        self.drl_tilt_deg_history[interval] = drl_tilt_deg_this_interval
        drl_kpis = self.kpi_manager.compute_ue_kpis(
            pooled_drl_power_w_r0[None, :, :], threshold_db=self.coverage_threshold_db,
            ut_loc=pooled_ut_loc_r0[None, :, :],
            distance_threshold_m=self.distance_threshold_m, overlap_margin_db=self.adaptive_legacy_overlap_margin_db,
        )
        self.coverage_drl[interval] = drl_kpis["coverage"]
        self.sinr_percentiles_drl[:, interval] = np.percentile(drl_kpis["sinr_db"], self.sinr_percentiles)
        drl_coverage_per_sector = drl_kpis["per_sector_coverage"]
        drl_overshoot_per_sector = drl_kpis["overshoot"]
        self.drl_overshoot_history[interval] = drl_overshoot_per_sector
        self.drl_reward_history[interval] = compute_reward(
            self.reward_type, drl_coverage_per_sector, drl_overshoot_per_sector,
            self.reward_lambda_coverage, self.reward_lambda_overshoot,
        )
        drl_losses_before = len(self.drl_policy.step_losses)
        drl_observations = compute_state(self.state_type, drl_coverage_per_sector, drl_overshoot_per_sector)
        self.drl_controller.update(drl_observations, self.drl_reward_history[interval], training=True)
        if len(self.drl_policy.step_losses) > drl_losses_before:
            self.drl_loss_per_interval[interval] = np.mean(self.drl_policy.step_losses[drl_losses_before:])

    def run_simulation(self) -> dict:
        """The per-interval loop: pool every measurement draw's contribution
        via step_environment()+kpi_manager, then execute_algorithms() once
        per interval on the pooled result.
        """
        for interval in range(self.num_tilt_control_intervals):
            is_last_interval = interval == self.num_tilt_control_intervals - 1
            drl_tilt_deg_this_interval = tilt_idx_to_deg(self.drl_controller.tilt_idx, self.downtilt_sweep_deg)

            self.kpi_manager.start_interval()
            position_snapshot = None
            ref_xy_snapshot = None
            for sub_step in range(self.measurement_slots_per_interval):
                if sub_step == 0:
                    position_snapshot = self.mobility_list[0].ut_loc.detach().cpu().numpy().copy()
                    if self.has_clusters:
                        ref_xy_snapshot = self.mobility_list[0].ref_xy.detach().cpu().numpy().copy()
                ut_loc_r0_this_draw = self.mobility_list[0].ut_loc.detach().cpu().numpy().copy()

                is_last_step = is_last_interval and (sub_step == self.measurement_slots_per_interval - 1)
                states = self.step_environment(is_last_step)
                for chunk_idx, state in enumerate(states):
                    self.kpi_manager.pool_measurement(
                        state, self.adaptive_legacy_controller.tilt_deg, drl_tilt_deg_this_interval,
                        self.downtilt_sweep_deg, ut_loc_r0=ut_loc_r0_this_draw if chunk_idx == 0 else None,
                    )

            self.position_history[interval] = position_snapshot
            if self.has_clusters:
                self.ref_xy_history[interval] = ref_xy_snapshot

            pooled = self.kpi_manager.pooled_interval()
            self.execute_algorithms(pooled, interval, drl_tilt_deg_this_interval)

            if interval % 10 == 0 or is_last_interval:
                print(
                    f"interval {interval:3d}/{self.num_tilt_control_intervals - 1}: "
                    f"dynamic_local_oracle={self.coverage_dynamic_local_oracle[interval]:.3f}, "
                    f"dynamic_local_causal={self.coverage_dynamic_local_causal[interval]:.3f}, "
                    f"adaptive_legacy={self.coverage_adaptive_legacy[interval]:.3f}, "
                    f"no_tilt={self.coverage_no_tilt[interval]:.3f}, "
                    f"drl={self.coverage_drl[interval]:.3f}"
                )

        no_tilt_deg = np.full(self.num_bs, NO_TILT_DEG)

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
