"""The environment + algorithm side of the dynamic tilt-control comparison:
SimulationEngine owns everything -- topology/mobility/channel setup,
environment stepping, and both methods' (Adaptive Legacy, DRL -- DRL's own
policy_name switches between the trained DQN and the untrained
RandomPolicy baseline) per-interval decisions. Exactly these two exist as
of 2026-09-22 -- Dynamic Local Causal, No Tilt, and both the old
coordinate-ascent Oracle and its CP-SAT replacement are all deleted (CP-SAT
wasn't earning its own real-world cost relative to the baseline pipeline).
Nothing else is kept.

One measurement (mobility position + channel draw) per tilt-control
interval. DRL's async_schedule (see run_simulation) picks "sync" (every
sector decides at once, same resolved power) or "async" (sectors decide
in sequence, Gauss-Seidel -- each turn re-resolves power at the tilt
vector as it stands, already reflecting earlier turns this interval).

__init__(cfg) builds the whole world from a merged config dict;
run_simulation() is the per-interval loop -- everything it needs is
self.*, no nested functions. Mobility animation lives in
helpers.utils.save_mobility_animation. KpiManager (helpers/kpi_manager.py)
is a pure function of (state, tilt) to power/KPIs, reused every interval.
"""

import os

import numpy as np
import torch

from sionna.phy.channel.tr38901 import Antenna, AntennaArray, RMa, UMa, UMi
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from sionna.phy.constants import BOLTZMANN_CONSTANT

from drl.factory import create_policy
from drl.interface import cell_state_feature_names, compute_cell_state, compute_polar_regional_neighbor_state, \
    compute_region_state, compute_reward, num_features_for, polar_regional_neighbor_state_feature_names, \
    state_feature_names, tilt_idx_to_deg
from .cell_aligned_info_provider import CellAlignedInfoProvider
from .cellular_topology import CellularTopology
from .drl_diagnostics import DrlDiagnosticsLogger
from .electrical_downtilt import ElectricalDowntilt
from .kpi_manager import KpiManager
from .large_scale_channel import LargeScaleChannel
from .mobility import HotspotDemandMobility, ReferencePointGroupMobility, build_sector_biased_transition_matrix
from .occupancy_estimator import OccupancyEstimator
from .regional_info_provider import RegionalInfoProvider
from .spatial_grid_estimator import SpatialGridEstimator
from .sticky_los import StickyLosState
from .tilt_controller import AdaptiveLegacyTiltController, RLTiltController
from .ue_drop import UeDropper
from .utils import get_logger

logger = get_logger(__name__)

_SCENARIO_CHANNEL_MODEL = {"umi": UMi, "uma": UMa, "rma": RMa}

# hotspot_region_membership's calibration rule (2026-09-29): a FIXED
# RADIAL-DISTANCE classification -- near: dist_to_site < _HOTSPOT_NEAR_DIST_M,
# mid: between the two cuts, far: dist_to_site >= _HOTSPOT_FAR_DIST_M.
# Replaces the earlier SINR-VALUE rule (near/mid/far by real simulated SINR
# at a fixed 0-deg/all-sectors/forced-LOS/no-shadow reference condition,
# 5/0 dB cuts) -- that rule entangled RADIAL distance with AZIMUTHAL
# interference geometry (a cell's SINR depends on how close it sits to a
# shared sector boundary, not just distance from its own site), so its
# near/mid/far bands didn't line up with the two-population coverage
# reversal found empirically by sweeping tilt against PURE distance bins
# (scripts/radial_swap_plot.py, 2026-09-28/29): cells under ~900m
# genuinely prefer downtilt, cells from ~900-2300m genuinely prefer
# uptilt, roles reversing cleanly -- exactly the kind of state-dependent
# tilt tradeoff hotspot demand clustering needs to place UEs against, and
# something a purely radial split captures directly (tilt is an
# elevation-only control with NO azimuthal effect, so radial distance is
# the physically relevant axis, not SINR value or diagonal proximity).
# The earlier SINR-value rule also required a live radio simulation sweep
# every construction just to calibrate two thresholds; a pure distance
# rule is a one-time geometric computation, no radio pass needed, and
# survives ANY channel.force_los/disable_shadow_fading/antenna change
# without needing a separate frozen reference condition.
# Used ONLY for mobility placement (demand_states) as of 2026-09-22 -- the
# cell_aligned state_type (see CellAlignedInfoProvider) doesn't use
# near/mid/far at all; region_aligned's own state keeps using
# OccupancyEstimator's own SEPARATE, unaffected index-based rule.
#
# UMa recalibration (2026-10-07): the above 900m/2300m were RMa's own
# values (isd=5000m). After topology.scenario switched to uma (isd=500m),
# near/mid's crossover was re-measured directly (scripts/
# radial_swap_plot_scenario.py, seed-averaged distance-binned coverage
# swap) rather than naively isd-scaled -- the crossover is governed by
# antenna geometry (mechanical pitch), which does NOT scale with isd the
# way cell_radius does (UMa's own mechanical pitch is 9.36 deg vs RMa's
# 1.51 deg, a 6.2x difference, not the 10x isd ratio). Real measured
# crossover: ~174.2m. far_dist_m has no empirical crossover in either
# scenario's data (confirmed 2026-10-07) -- kept at the isd-ratio-scaled
# fraction of RMa's own 2300m (which itself was never independently
# derived either), not a second measured feature.
_HOTSPOT_NEAR_DIST_M = 174.2
_HOTSPOT_FAR_DIST_M = 230.0


class SimulationEngine:
    """Evaluate Adaptive Legacy and DRL -- against the same
    per-CONTROL-INTERVAL pooled data, every interval. __init__ does all
    setup (topology/mobility/channel,
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
        adaptive_legacy_cfg = cfg["algorithms"]["adaptive_legacy"]
        drl_cfg = cfg["algorithms"]["drl"]

        scenario = topology_cfg["scenario"]
        num_rings = topology_cfg["num_rings"]
        num_ut = topology_cfg["num_ut"]
        ut_height = topology_cfg["ut_height"]
        self.region_definition_method = topology_cfg.get("region_definition_method", "radial")
        assert self.region_definition_method in ("radial", "sinr"), \
            f"region_definition_method={self.region_definition_method!r} must be 'radial' or 'sinr'"
        m = antenna_cfg["m"]
        antenna_window = antenna_cfg["antenna_window"]
        azimuth_offset_deg = antenna_cfg.get("azimuth_offset_deg", 0.0)
        bs_tx_power_dbm = channel_cfg["bs_tx_power_dbm"]
        channel_bandwidth_prb = channel_cfg["channel_bandwidth_prb"]
        temperature = channel_cfg["temperature"]
        noise_figure_db = channel_cfg["noise_figure_db"]
        carrier_frequency = channel_cfg["carrier_frequency"]
        self.force_los = channel_cfg.get("force_los")
        disable_shadow_fading = channel_cfg.get("disable_shadow_fading", False)
        # See helpers/sticky_los.py's own module docstring -- entirely
        # optional, self-contained (this is its only construction site;
        # step_environment's own call is its only other touch point).
        self.sticky_los = StickyLosState() if channel_cfg.get("sticky_los_enabled", False) else None
        environment_seed = simulation_cfg["environment_seed"]
        mobility_common_cfg = mobility_cfg["common"]
        mobility_rpgm_cfg = mobility_cfg["rpgm"]
        num_clusters_per_sector = mobility_common_cfg["num_clusters_per_sector"]
        # num_ues_per_cluster: DERIVED below (2026-10-09), not a separate
        # config key -- num_ut, num_clusters_per_sector, and num_bs
        # (topology.num_rings) together already fully determine it; a
        # fourth independently-set value just invited the num_rings-changed-
        # but-num_ut-forgot-to-update bug this replaces (previously an
        # assert that num_ut == num_bs*num_clusters_per_sector*
        # num_ues_per_cluster, requiring all three kept in sync by hand).
        member_walk_step_frac = mobility_common_cfg["member_walk_step_frac"]
        # cluster_mobility_mode is "<family>-<submode>" (2026-10-01, e.g.
        # "rpgm-markov", "hotspot_demand-periodic") -- family picks which
        # mobility class gets built below, submode is that class's own
        # internal choice (ReferencePointGroupMobility's own
        # cluster_mobility_mode param for rpgm; HotspotDemandMobility's own
        # state_transition_mode param for hotspot_demand). Previously
        # hotspot_demand's own submode lived in a separate, differently-named
        # key (mobility.hotspot.state_transition_mode) not visible at this
        # top level at all -- an inconsistency with rpgm's own three modes
        # (random/periodic/markov), which WERE already top-level values here.
        cluster_mobility_mode = mobility_common_cfg["cluster_mobility_mode"]
        assert "-" in cluster_mobility_mode, \
            (f"cluster_mobility_mode must be '<family>-<submode>' (e.g. 'rpgm-random', "
            f"'hotspot_demand-markov'), got {cluster_mobility_mode!r} -- the old bare "
            f"'random'/'periodic'/'markov'/'hotspot_demand' values were replaced 2026-10-01")
        mobility_family, mobility_submode = cluster_mobility_mode.split("-", 1)
        assert mobility_family in ("rpgm", "hotspot_demand"), \
            f"cluster_mobility_mode must start with 'rpgm-' or 'hotspot_demand-', got {cluster_mobility_mode!r}"
        deviation_radius_frac_area = mobility_rpgm_cfg["deviation_radius_frac_area"]
        cluster_radius_max_m = mobility_rpgm_cfg["cluster_radius_max_m"]
        num_waypoints = mobility_rpgm_cfg["num_waypoints"]
        waypoint_hold_steps = mobility_rpgm_cfg["waypoint_hold_steps"]
        waypoint_transition_matrix = mobility_rpgm_cfg["waypoint_transition_matrix"]

        self.downtilt_sweep_deg = np.arange(
            downtilt_cfg["downtilt_sweep_start_deg"],
            downtilt_cfg["downtilt_sweep_end_deg"] + downtilt_cfg["tilt_step_deg"],
            downtilt_cfg["tilt_step_deg"],
        )
        self.coverage_threshold_db = kpi_cfg["coverage_threshold_db"]
        self.rsrp_threshold_dbm = kpi_cfg.get("rsrp_threshold_dbm")
        self.ho_mode = kpi_cfg.get("ho_mode", "a3_hysteresis")
        assert self.ho_mode in ("max_rsrp", "a3_hysteresis"), "kpi.ho_mode must be 'max_rsrp' or 'a3_hysteresis'"
        self.a3_hysteresis_db = kpi_cfg.get("a3_hysteresis_db", 3.0)
        self.sinr_percentiles = kpi_cfg["sinr_percentiles"]
        self.adaptive_legacy_overlap_margin_db = adaptive_legacy_cfg["overlap_margin_db"]
        self.adaptive_legacy_edge_percentile = adaptive_legacy_cfg["edge_percentile"]
        self.adaptive_legacy_tilt_step_deg = downtilt_cfg["tilt_step_deg"]
        self.adaptive_legacy_os_threshold = adaptive_legacy_cfg["os_threshold"]
        self.adaptive_legacy_bc_threshold = adaptive_legacy_cfg["bc_threshold"]
        self.num_ut = num_ut
        # kpi.ho_mode="a3_hysteresis" only: one persistent serving_idx
        # array per real per-interval scoring path (see config.yaml's own
        # comment and _compute_ue_kpis_tracked below) -- each algorithm
        # applies a different tilt vector over time, so each needs its OWN
        # handover trajectory, not one shared one. -1 = no prior attachment
        # yet (falls back to plain max-RSRP for that UE's very first
        # measurement, same as 3GPP initial cell selection).
        self._serving_idx = {name: np.full((1, num_ut), -1, dtype=int)
                             for name in ("adaptive_legacy", "drl")}
        self.num_tilt_control_intervals = simulation_cfg["num_tilt_control_intervals"]
        self.policy_name = drl_cfg["policy_name"]
        self.algorithm_seed = drl_cfg["algorithm_seed"]
        self.state_type = drl_cfg["state_type"]
        self.reward_type = drl_cfg["reward_type"]
        # cgr_cfg: kept for debug tooling's own standalone CGR estimator
        # (e.g. scripts/debug_pipeline_audit.py's phase 5 validation) --
        # production's own spatial_grid_estimator below is Kriging-only.
        self.cgr_cfg = drl_cfg.get("cgr", {})
        self.dqn_kwargs = dict(drl_cfg["dqn"])
        # tilt_quantization_deg builds drl_action_grid_deg below, not a Ppo
        # constructor kwarg -- excluded here rather than accepted-and-
        # ignored by Ppo.__init__.
        self.ppo_kwargs = {k: v for k, v in drl_cfg.get("ppo", {}).items() if k != "tilt_quantization_deg"}
        # drl_action_grid_deg: the grid the DRL controller's own tilt_idx
        # indexes into -- identical to downtilt_sweep_deg for dqn/random (no
        # behavior change there), but a separate, much finer grid for ppo
        # (continuous action, floor-quantized -- see drl/ppo.py). Adaptive
        # Legacy and the antenna init keep using downtilt_sweep_deg
        # directly, untouched by this.
        if self.policy_name == "ppo":
            ppo_cfg = drl_cfg.get("ppo", {})
            quantization_deg = ppo_cfg.get("tilt_quantization_deg", 0.1)
            theta_min, theta_max = self.downtilt_sweep_deg.min(), self.downtilt_sweep_deg.max()
            n_steps = int(round((theta_max - theta_min) / quantization_deg))
            self.drl_action_grid_deg = theta_min + quantization_deg * np.arange(n_steps + 1)
        else:
            self.drl_action_grid_deg = self.downtilt_sweep_deg
        self.async_schedule = drl_cfg["async_schedule"]
        diagnostics_cfg = drl_cfg.get("diagnostics", {})
        self.diagnostics_enabled = diagnostics_cfg.get("enabled", False)
        self.diagnostics_log_q_values = diagnostics_cfg.get("log_q_values", True)
        self.diagnostics_log_oracle_comparison = diagnostics_cfg.get("log_oracle_comparison", True)
        if self.policy_name == "ppo" and self.diagnostics_log_oracle_comparison:
            # _oracle_single_sector_step's power_table is built over
            # downtilt_sweep_deg (the coarse dqn/adaptive_legacy grid) and
            # indexes it with self.drl_controller.tilt_idx directly --
            # ppo's tilt_idx indexes its own much finer drl_action_grid_deg
            # instead, so the two aren't index-compatible. Forced off
            # rather than either rebuilding power_table at ppo's ~16x finer
            # resolution (diagnostics-only feature, not worth the extra
            # compute_tilt_sector_ue_rx_power cost every interval) or
            # silently producing wrong oracle comparisons.
            logger.warning("SimulationEngine: diagnostics.log_oracle_comparison disabled for policy_name=ppo "
                          "(oracle power_table uses the coarse downtilt_sweep_deg grid, incompatible with "
                          "ppo's own finer drl_action_grid_deg)")
            self.diagnostics_log_oracle_comparison = False
        self.diagnostics_out_subdir = diagnostics_cfg.get("out_subdir", "diagnostics")

        hotspot_cfg = mobility_cfg.get("hotspot", {})

        # Set scenario params
        min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, indoor_probability, _, _ = \
            set_3gpp_scenario_parameters(scenario)
        # topology.ut_height is a free config override (3GPP itself allows UE
        # height to vary, e.g. indoor floors up to max_ut_height), but must
        # stay within the scenario's own [min_ut_height, max_ut_height] --
        # catches a stale ut_height left over from a different scenario.
        assert min_ut_height.item() <= ut_height <= max_ut_height.item(), (
            f"topology.ut_height={ut_height} is outside scenario {scenario!r}'s own 3GPP height range "
            f"[{min_ut_height.item()}, {max_ut_height.item()}]"
        )
        outdoor_ue_ratio_override = topology_cfg.get("outdoor_ue_ratio_override")
        outdoor_ratio = (1.0 - float(indoor_probability.item())) if outdoor_ue_ratio_override is None \
            else float(outdoor_ue_ratio_override)
        outdoor_gnss_prob = topology_cfg["outdoor_gnss_prob"]
        indoor_gnss_prob = topology_cfg["indoor_gnss_prob"]
        scenario_params = {
            "isd": isd,
            "bs_height": bs_height,
            "min_bs_ut_dist": min_bs_ut_dist,
        }

        # Cellular topology -- batch_size=1, one measurement per interval.
        self.topology = topology if topology is not None else \
            CellularTopology(scenario_params, num_rings=num_rings, batch_size=1,
                            azimuth_offset_deg=azimuth_offset_deg)
        self.num_bs = self.topology.num_bs
        self.ut_height = ut_height

        # Population consistency -- SHARED by every cluster_mobility_mode
        # (hotspot_demand and the legacy random/periodic/markov RPGM modes
        # alike): exactly num_clusters_per_sector clusters per sector, every
        # sector, num_ut total members split as EVENLY as possible across
        # all clusters (2026-10-09 -- previously required num_ut to be an
        # EXACT multiple, rejecting anything else; relaxed since
        # UeDropper.clustered already natively accepts a per-cluster list
        # of member counts, not just a uniform scalar -- a non-exact
        # num_ut is a real, supported case, not an error). near-even split:
        # num_ut % clusters_total clusters get one extra member (ceil),
        # the rest get the floor -- reduces to the old uniform behavior
        # exactly when num_ut divides evenly (remainder 0).
        clusters_total = self.num_bs * num_clusters_per_sector
        assert num_ut >= clusters_total, (
            f"topology.num_ut ({num_ut}) must be at least num_bs * mobility.num_clusters_per_sector "
            f"({self.num_bs} * {num_clusters_per_sector} = {clusters_total}) -- every cluster needs "
            "at least 1 member.")
        base, remainder = divmod(num_ut, clusters_total)
        num_ues_per_cluster = [base + 1] * remainder + [base] * (clusters_total - remainder)
        # stored for post-construction introspection (e.g. logging) -- a
        # list even when uniform (remainder=0, every entry equals base),
        # for one consistent type regardless of whether num_ut divides evenly.
        self.num_ues_per_cluster = num_ues_per_cluster
        self.distance_threshold_m = float(self.topology.grid.cell_radius)

        # Per-sector rhombus grid -- needed here (ahead of the DRL section
        # below, which also uses it) so "periodic"/"markov" waypoint
        # auto-generation can place waypoints inside each cluster's home
        # sector's near/mid/far hotspot regions.
        spatial_grid_cell_size_m = topology_cfg["spatial_grid_cell_size_m"]
        self.grid_points, self.grid_n = self.topology.build_sector_rhombus_grid(spatial_grid_cell_size_m)
        self.num_grid_cells = self.grid_n * self.grid_n

        # region_aligned state: this sector's OWN rhombus grid partitioned
        # into the near/mid/far hotspot regions (num_own_regions, the ONE
        # region definition in this codebase -- see
        # helpers/regional_info_provider.py/OccupancyEstimator.region_names),
        # own SINR p25/p50/p75, + one coarse (occupancy, coverage) pair per
        # REAL cross-site neighbor (fixed at 2 neighbor slots, not one per
        # other sector -- see drl.interface.compute_region_state's own
        # docstring for the empirical justification, 2026-09-20).
        # Computed once here regardless of state_type, since it's cheap
        # relative to everything else in __init__ and num_features_for
        # needs it either way. Derived from OccupancyEstimator's own class
        # attribute (not hardcoded) so this and RegionalInfoProvider's own
        # num_own_regions can never drift apart.
        self.num_own_regions = len(OccupancyEstimator.region_names)
        # Cluster radius is capped at cluster_radius_max_m (asserted below),
        # so requiring every waypoint candidate -- cluster centers and
        # auto-generated waypoints alike -- to clear min_bs_ut_dist +
        # cluster_radius_max_m from its own site guarantees the WHOLE
        # cluster disk stays outside the exclusion zone, not just its
        # center point. deviation_radius_frac_area is rpgm-only (ignored
        # for hotspot_demand, whose own deviation radius is
        # mobility.hotspot.deviation_radius_m instead -- see config.yaml),
        # so only check it against the cap when an rpgm mode is actually
        # active; cluster_radius_max_m itself still sizes the exclusion
        # buffer for every mode, hotspot_demand included.
        if mobility_family != "hotspot_demand":
            deviation_radius = deviation_radius_frac_area * self.topology.default_drop_radius
            assert deviation_radius <= cluster_radius_max_m, (
                f"deviation_radius ({deviation_radius:.2f}m, from deviation_radius_frac_area) exceeds "
                f"cluster_radius_max_m ({cluster_radius_max_m:.2f}m) -- the waypoint exclusion-zone buffer "
                "assumes clusters never grow past this cap")
        waypoint_exclusion_radius = float(self.topology.min_bs_ut_dist) + cluster_radius_max_m

        # BS and UE Antenna array -- built here, ahead of mobility/UE
        # placement below, because calibrating hotspot_region_membership's
        # near/mid/far lookup table (see _calibrate_hotspot_region_lut)
        # needs a working radio model, and mobility's own construction is
        # the first real consumer of hotspot_region_membership (via
        # helpers/ue_drop.py's cluster/waypoint placement).
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
            direction="downlink", enable_pathloss=True, enable_shadow_fading=not disable_shadow_fading,
        )
        if scenario != "rma":
            channel_model_kwargs["o2i_model"] = "low"
        # Stashed for _hotspot_region_lut_sinr's own throwaway calibration
        # channel model (same scenario/antenna config, shadow fading forced
        # off there regardless of this run's own disable_shadow_fading).
        self._channel_model_cls = channel_model_cls
        self._channel_model_kwargs = channel_model_kwargs
        self.channel_model = channel_model_cls(**channel_model_kwargs)
        self.large_scale_channel = LargeScaleChannel(self.channel_model)

        # Tx/noise power
        bs_tx_power_w = 10 ** ((bs_tx_power_dbm - 30.0) / 10.0)
        channel_bandwidth_hz = channel_bandwidth_prb * 12 * 30e3
        noise_power_w = BOLTZMANN_CONSTANT * temperature * channel_bandwidth_hz * 10 ** (noise_figure_db / 10.0)

        # Sites/sectors share the same (x, y) geometry
        bs_xy = self.topology.bs_loc[0, :, :2].detach()

        # KPI manager -- adjacency is pure geometry, computed by CellularTopology
        self.kpi_manager = KpiManager(sector_etilts, bs_tx_power_w, noise_power_w, bs_xy,
                                      self.topology.neighbor_ids, self.topology.max_neighbors,
                                      self.topology.sector_adjacency)

        # Calibrate hotspot_region_membership's near/mid lookup table
        # (pure radial distance now -- see its own docstring), before any
        # real UE placement below needs it. No longer depends on the radio
        # model above (kept after it only for construction-order history).
        self._calibrate_hotspot_region_lut()

        # Initial locations and single mobility instance (one measurement/interval).
        environment_generator = torch.Generator(device=self.topology.bs_loc.device)
        environment_generator.manual_seed(environment_seed)

        if mobility_family == "hotspot_demand":
            hotspot_deviation_radius = hotspot_cfg["deviation_radius_m"]
            # sector_bias (config_3, 2026-09-25): OPTIONAL [num_bs] list of
            # demand_states row indices -- when given, builds a PER-SECTOR
            # transition matrix (each sector sticky toward its own assigned
            # state) instead of the shared uniform-default one, so
            # different sectors genuinely differ in their long-run demand
            # mix -- see build_sector_biased_transition_matrix's own
            # docstring. Mutually exclusive with hotspot_cfg's own
            # transition_matrix (shared 2D) -- sector_bias wins if both set.
            sector_bias = hotspot_cfg.get("sector_bias")
            if sector_bias is not None:
                hotspot_transition_matrix = build_sector_biased_transition_matrix(
                    num_states=len(hotspot_cfg["demand_states"]), sector_bias=sector_bias,
                    bias_strength=hotspot_cfg.get("sector_bias_strength", 0.7))
            else:
                hotspot_transition_matrix = hotspot_cfg.get("transition_matrix")
            self.mobility = HotspotDemandMobility(
                topo=self.topology, generator=environment_generator,
                num_clusters_per_sector=num_clusters_per_sector,
                num_ues_per_cluster=num_ues_per_cluster, ut_height=ut_height,
                deviation_radius=hotspot_deviation_radius,
                deviation_radius_tangential=hotspot_cfg.get("deviation_radius_tangential_m"),
                member_walk_step=member_walk_step_frac * hotspot_deviation_radius,
                hotspot_hold_steps=hotspot_cfg["hotspot_hold_steps"], demand_states=hotspot_cfg["demand_states"],
                state_transition_mode=mobility_submode,
                transition_matrix=hotspot_transition_matrix,
            )
        else:
            sampler = UeDropper(self.topology, generator=environment_generator)
            start_xy_list = sampler.cluster_centers(num_clusters_per_sector,
                                                    min_dist_from_site=waypoint_exclusion_radius)
            initial_ut_loc, member_group_idx = sampler.clustered(
                start_xy_list, num_ues_per_cluster, deviation_radius, ut_height,
            )
            start_xy_arr = np.asarray(start_xy_list, dtype=float)
            home_sector, _ = self.topology.assign_to_sector_rhombus_grid(start_xy_arr)
            self.mobility = ReferencePointGroupMobility(
                initial_ut_loc, member_group_idx, start_xy_list,
                deviation_radius=deviation_radius, member_walk_step=member_walk_step_frac * deviation_radius,
                topo=self.topology, generator=environment_generator,
                cluster_mobility_mode=mobility_submode,
                num_waypoints=num_waypoints, waypoint_hold_steps=waypoint_hold_steps,
                waypoint_transition_matrix=waypoint_transition_matrix,
                cluster_home_sector=home_sector,
                waypoint_min_dist_from_site=waypoint_exclusion_radius,
            )

        # UT orientation and mobility
        self.ut_orientations = torch.zeros(1, num_ut, 3,
                                           dtype=self.topology.bs_loc.dtype, device=self.topology.bs_loc.device)
        self.ut_velocities = torch.zeros_like(self.ut_orientations)
        # Indoor/outdoor split -- per CLUSTER (2026-10-10), not per UE: one
        # draw per cluster, broadcast to every member, since two UEs at the
        # SAME physical location can't be independently indoor/outdoor by
        # chance (member_group_idx is the same cluster-membership array
        # location_reporting_mask below uses). outdoor_ratio: DERIVED from
        # the scenario's own 3GPP indoor_probability (1 - indoor_probability)
        # unless topology.outdoor_ue_ratio_override overrides it (see its own
        # config.yaml comment). Drawn once, fixed for the whole run, a
        # DIFFERENT derived seed than mask_rng's own environment_seed so the
        # two draws can never coincidentally end up correlated.
        indoor_rng = np.random.default_rng(environment_seed + 1)
        member_group_idx_np = self.mobility.member_group_idx.detach().cpu().numpy()
        group_ids = np.unique(member_group_idx_np)
        cluster_is_indoor = indoor_rng.random(len(group_ids)) >= outdoor_ratio
        is_indoor = np.zeros(num_ut, dtype=bool)
        for gi, group_id in enumerate(group_ids):
            is_indoor[member_group_idx_np == group_id] = cluster_is_indoor[gi]
        self.in_state = torch.as_tensor(is_indoor, dtype=torch.bool,
                                        device=self.topology.bs_loc.device).unsqueeze(0)

        # Initialization of KPI history arrays.
        n = self.num_tilt_control_intervals
        self.coverage_adaptive_legacy = np.zeros(n)
        self.coverage_drl = np.zeros(n)
        self.sinr_percentiles_adaptive_legacy = np.zeros((len(self.sinr_percentiles), n))
        self.sinr_percentiles_drl = np.zeros((len(self.sinr_percentiles), n))
        self.adaptive_legacy_tilt_deg_history = np.zeros((n, self.num_bs))
        self.drl_tilt_deg_history = np.zeros((n, self.num_bs))

        # DRL-only
        self.drl_reward_history = np.zeros((n, self.num_bs))
        self.drl_overshoot_history = np.zeros((n, self.num_bs))
        self.drl_loss_per_interval = np.full(n, np.nan)
        self.position_history = np.zeros((n, self.num_ut, 3))

        self.has_clusters = hasattr(self.mobility, "ref_xy")
        self.ref_xy_history = (np.zeros((n, self.mobility.ref_xy.shape[0], 2))
                               if self.has_clusters else None)

        # Algorithms
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

        # DRL -- state is fixed-width for every sector: own regions
        # (region_aligned) or own fine-grid cells (cell_aligned, 2026-09-22
        # addition -- see helpers/cell_aligned_info_provider.py's own
        # module docstring), + coarse feedback from exactly 2 real
        # neighbors either way (see drl/interface.py). cell_aligned's own
        # provider is built here (not conditionally deferred) since
        # num_features needs its num_valid_cells regardless of whether
        # this run actually uses it below.
        assert self.state_type in ("region_aligned", "cell_aligned", "polar_whole_neighbor",
                                   "polar_region_neighbor"), \
            "algorithms.drl.state_type must be 'region_aligned', 'cell_aligned', 'polar_whole_neighbor', " \
            "or 'polar_region_neighbor'"
        self.cell_aligned_info_provider = None
        num_valid_cells = None
        if self.state_type == "cell_aligned":
            self.cell_aligned_info_provider = CellAlignedInfoProvider(
                self.num_bs, self.grid_n, self.grid_points, self.topology)
            num_valid_cells = self.cell_aligned_info_provider.num_valid_cells

        # polar_whole_neighbor/polar_region_neighbor (2026-10-15, explicit
        # request): own-sector state is a polar (radial x azimuthal) grid
        # instead of near/mid/far or per-cell -- see _calibrate_polar_lut's
        # own docstring. self.polar_info_provider computes ONLY the own
        # polar-binned occupancy/SINR (its own reward/all_occupancy outputs
        # are unused duplicates -- self.regional_info_provider, built
        # below, is still the REWARD source for every state_type).
        self.polar_info_provider = None
        self.num_polar_bins = None
        if self.state_type in ("polar_whole_neighbor", "polar_region_neighbor"):
            polar_num_azimuthal = drl_cfg.get("polar_num_azimuthal", 5)
            cell_to_polar_bin, polar_bin_names = self._calibrate_polar_lut(polar_num_azimuthal)
            self.polar_info_provider = RegionalInfoProvider.from_cell_to_region(
                self.topology.neighbor_ids, self.num_ut, cell_to_polar_bin, polar_bin_names)
            self.num_polar_bins = self.polar_info_provider.num_own_regions

        num_features = num_features_for(
            self.state_type, num_own_regions=self.num_own_regions, num_sectors=self.num_bs,
            num_valid_cells=num_valid_cells, num_polar_bins=self.num_polar_bins,
        )
        self.drl_policy = create_policy(
            self.policy_name, self.num_bs, num_features=num_features,
            num_actions=len(self.drl_action_grid_deg), action_grid_deg=self.drl_action_grid_deg,
            dqn_kwargs=self.dqn_kwargs, ppo_kwargs=self.ppo_kwargs,
            # CPU, not the pipeline default of cuda-if-available: measured
            # ~1.6x SLOWER on GPU for this workload (21 tiny 32-hidden-unit
            # networks, single-sample inference, batch_size=32 training) --
            # per-call kernel-launch/transfer overhead dominates at this
            # scale, leaving no compute for GPU parallelism to offset it.
            # That measurement assumed uncontended CPU, though -- DRL_DEVICE
            # env var overrides it (2026-09-30) for testing whether GPU
            # wins instead once the CPU itself is the scarce/contended
            # resource (e.g. a shared machine with other users' jobs).
            algorithm_seed=self.algorithm_seed, device=os.environ.get("DRL_DEVICE", "cpu"),
            expose_diagnostics=self.diagnostics_enabled and self.diagnostics_log_q_values,
        )
        self.drl_controller = RLTiltController(self.drl_policy, self.num_bs, initial_tilt_idx=0)
        assert self.async_schedule in ("sync", "async"), "async_schedule must be 'sync' or 'async'"

        # diagnostics_enabled=false (default): stays None, near-zero cost
        # (see run_simulation/drl_process_turn -- every diagnostics-only
        # branch is gated on this being non-None).
        self.diagnostics_logger = None
        if self.diagnostics_enabled:
            if self.state_type == "cell_aligned":
                feature_names = cell_state_feature_names(num_valid_cells)
            elif self.state_type == "polar_whole_neighbor":
                feature_names = state_feature_names(self.num_polar_bins, self.polar_info_provider.region_names)
            elif self.state_type == "polar_region_neighbor":
                feature_names = polar_regional_neighbor_state_feature_names(
                    self.num_polar_bins, self.polar_info_provider.region_names,
                    self.num_own_regions, OccupancyEstimator.region_names)
            else:
                feature_names = state_feature_names(self.num_own_regions, OccupancyEstimator.region_names)
            self.diagnostics_logger = DrlDiagnosticsLogger(
                self.num_bs, self.drl_action_grid_deg, self.policy_name, feature_names)

        self._interval_step_losses = []
        self._interval_last_reward = np.full(self.num_bs, np.nan)

        # region_aligned state (own regions + 2 real neighbors' coarse
        # pair) + a global reward (coverage ratio network-wide) -- see
        # helpers/regional_info_provider.py. state_type/reward_type are
        # validated by num_features_for/compute_reward -- region_aligned/
        # global are the only supported values.
        self.regional_info_provider = RegionalInfoProvider(
            self.topology.neighbor_ids, self.grid_n, self.num_ut,
            self._hotspot_near_mask, self._hotspot_mid_mask)

        # Anchor (location-known) status -- per CLUSTER (2026-10-10), same
        # reasoning as the indoor/outdoor split above: every member of a
        # cluster shares one anchor draw, conditioned on that SAME
        # cluster's own indoor/outdoor status (cluster_is_indoor, just
        # computed above) via topology.outdoor_gnss_prob/indoor_gnss_prob --
        # a simplistic 1-1 GNSS-capability model (outdoor UEs get GNSS,
        # i.e. anchor status, with probability outdoor_gnss_prob; indoor
        # UEs with indoor_gnss_prob). REPLACES the old, fully independent
        # location_reporting_fraction/location_reporting_mode mechanism
        # (see config.yaml's own comment on those removed keys) -- anchor
        # status is no longer uncorrelated with indoor/outdoor, since two
        # UEs at the same physical location can't plausibly differ in
        # GNSS availability by chance. Reward and whole-sector coarse
        # state (all_occupancy/all_sinr_p50) always count every UE fully
        # and exactly (serving sector is ordinary connectivity, never
        # actually unknown; SINR/coverage never depends on location
        # either). Own-region state (own_occupancy/own_sinr_*) is
        # different: a non-anchor UE's own region genuinely isn't known,
        # so non_anchor_region_handling (below, config-switchable) decides
        # what happens to it in drl_process_turn's own accumulate() call --
        # "exclude" (default): simply not counted there at all;
        # "rme": RME/Kriging-inferred via self.spatial_grid_estimator
        # (built below whenever any UE is non-anchor) -- NOT the default
        # (see that config key's own comment for why), but the estimator
        # stays fully available regardless of which mode is active, since
        # scripts/debug_pipeline_audit.py's own standalone Kriging/CGR
        # validation tooling always needs it.
        #
        # anchor_selection_mode (2026-10-15) -- "random" (default) or
        # "balanced": which clusters within a gnss-eligible population
        # become anchor. DYNAMIC for HotspotDemandMobility (both modes):
        # since a cluster's (sector, near/mid/far) slot is reshuffled every
        # hotspot transition (see helpers/mobility.py's own docstring --
        # cluster IDENTITY is permanent but its CURRENT region is not),
        # anchor status is re-derived from scratch at every transition too
        # (_redraw_location_reporting_mask, called here for the initial
        # draw and again from step_environment whenever
        # mobility.transitioned_this_step is True) rather than fixed once
        # for the whole run -- a static per-cluster draw would silently
        # stop meaning "this cluster's current region is anchored" the
        # moment mobility moves it elsewhere. "random": plain per-cluster
        # Bernoulli(cluster_gnss_prob) over the WHOLE population, redrawn
        # fresh every transition, ignoring region type entirely. "balanced":
        # stratified per (sector, region-type) -- within each currently-
        # occupied slot-group, max(1, round(p*n)) of its own n clusters are
        # anchor (never 0 when the slot is non-empty and p>0), explicitly
        # guaranteeing every region sector currently hosts stays
        # represented in the anchor-only state even at low global p -- a
        # deliberately harder "stress test" than random's unconstrained
        # draw, per explicit user request. For every other mobility mode
        # (no cluster_type/cluster_sector), the mask is simply drawn once
        # here and never redrawn (step_environment's hook is a no-op since
        # transitioned_this_step doesn't exist on those mobility objects).
        self.anchor_selection_mode = topology_cfg.get("anchor_selection_mode", "random")
        assert self.anchor_selection_mode in ("random", "balanced", "mid_prioritized_balanced"), \
            "topology.anchor_selection_mode must be 'random', 'balanced', or 'mid_prioritized_balanced'"
        if self.anchor_selection_mode in ("balanced", "mid_prioritized_balanced"):
            assert hasattr(self.mobility, "cluster_type") and hasattr(self.mobility, "cluster_sector"), \
                f"anchor_selection_mode={self.anchor_selection_mode!r} needs HotspotDemandMobility's " \
                "cluster_type/cluster_sector"
        self._anchor_rng = np.random.default_rng(environment_seed + 2)
        self._cluster_is_indoor = cluster_is_indoor
        self.indoor_gnss_prob = indoor_gnss_prob
        self.outdoor_gnss_prob = outdoor_gnss_prob
        self._redraw_location_reporting_mask()
        # Gaussian-kernel bandwidth [dB] for infer_region_probabilities'
        # own measured-vs-predicted-RSRP match -- calibrate empirically
        # against this scenario's own actual Kriging residual scale (see
        # its own docstring), not guessed.
        self.region_inference_sigma_db = drl_cfg.get("region_inference_sigma_db", 0.75)
        # "exclude" (default) or "rme" -- see config.yaml's own comment;
        # picks which of own_region_mask/region_idx_override
        # drl_process_turn actually passes to RegionalInfoProvider.accumulate.
        self.non_anchor_region_handling = drl_cfg.get("non_anchor_region_handling", "exclude")
        assert self.non_anchor_region_handling in ("exclude", "rme", "radio_map"), \
            "non_anchor_region_handling must be 'exclude', 'rme', or 'radio_map'"

        # Only built when non-anchor UEs can actually exist -- its ONLY job
        # is infer_region_probabilities' Kriging fit (never a coverage
        # map -- see RegionalInfoProvider's own module docstring), so it's
        # always Kriging (radio_map_estimator's own CGR option is
        # deliberately not offered here -- infer_region_probabilities is
        # Kriging-only by design, see its own docstring).
        self.spatial_grid_estimator = None
        if not self.location_reporting_mask.all():
            self.spatial_grid_estimator = SpatialGridEstimator(
                self.num_bs, self.num_grid_cells, self.grid_points,
                length_scale=spatial_grid_cell_size_m, coverage_threshold_db=self.coverage_threshold_db,
                radio_map_method="rsrp_per_sector", noise_power_w=noise_power_w,
                topology=self.topology,
            )

        # Reward timing: DRL always decides a_k from THIS interval's own
        # real, already-drawn channel/UE state (state = step_environment()
        # is called once per interval, before any per-sector turn -- see
        # run_simulation), resolved at the tilt vector as it stands right
        # before this turn's own decision (live_tilt_deg, already
        # reflecting every earlier-this-round turn's brand-new tilt under
        # async_schedule="async") -- never a forecast, never a stale
        # cross-interval cache. a_k is then immediately applied and
        # RE-measured, and THAT real outcome is what credits a_k -- no lag
        # between decision and evaluation, for either "sync" or "async"
        # scheduling (see drl_process_turn's own docstring for the exact
        # two-measurement mechanism). This replaced a next_state/
        # current_state config switch (2026-09-21) -- next_state (decide
        # from o_{k-1}, credited one interval late) is gone; the OLD
        # current_state (decide from a cached last-interval snapshot,
        # sync-only) is gone too, superseded by this per-turn, async-
        # capable version. Adaptive Legacy is unaffected -- it always
        # scores at the tilt already in effect, a separate mechanism
        # (AdaptiveLegacyTiltController.update).
        logger.info("SimulationEngine constructed: num_bs=%d num_ut=%d num_tilt_control_intervals=%d "
                   "policy_name=%s state_type=%s async_schedule=%s", self.num_bs, self.num_ut,
                   self.num_tilt_control_intervals, self.policy_name, self.state_type, self.async_schedule)

    def _calibrate_hotspot_region_lut(self) -> None:
        """Classifies every grid cell into near/mid/far and installs the
        result via topology.set_hotspot_region_lut, PLUS stores it
        (self._hotspot_near_mask/_hotspot_mid_mask) so RegionalInfoProvider's
        own OccupancyEstimator is built from the exact SAME partition --
        both PLACEMENT (hotspot_region_membership/hotspot_region_points,
        i.e. where demand-state clusters actually get sampled) and
        MEASUREMENT (OccupancyEstimator/RegionalInfoProvider, i.e. how
        existing UEs get classified into state/occupancy figures) read
        from this one calibrated table, so they can never silently diverge
        (2026-10-12 fix -- OccupancyEstimator used to compute its own,
        unrelated, long-stale diagonal-wedge rule instead; confirmed via
        region_occupancy_validation.png: ~80%+ "far" under every demand
        state, including near-heavy, once finally run through a real
        empirical check).

        Called once from __init__, before self.mobility (the first real
        consumer of hotspot_region_membership). topology.region_definition_method
        picks which rule: "radial" (default) or "sinr" (see this method's
        own two helpers below for each).
        """
        if self.region_definition_method == "radial":
            near_mask, mid_mask = self._hotspot_region_lut_radial()
        else:
            near_mask, mid_mask = self._hotspot_region_lut_sinr()
        self.topology.set_hotspot_region_lut(near_mask, mid_mask)
        self._hotspot_near_mask = near_mask
        self._hotspot_mid_mask = mid_mask

    def _hotspot_region_lut_radial(self) -> tuple:
        """RADIAL DISTANCE from each cell to its own sector's site, against
        _HOTSPOT_NEAR_DIST_M/_HOTSPOT_FAR_DIST_M (fixed cuts, 2026-09-29 --
        see their own module-level comment for why radial distance
        replaced the earlier real-SINR-value rule for PLACEMENT
        specifically). Pure geometry -- no radio simulation, no dependence
        on channel.force_los/disable_shadow_fading or any other propagation
        setting.

        Cross-sector consistency: every sector's own rhombus geometry is
        identical up to rotation (same site-relative basis vectors, only
        the boresight yaw differs -- see CellularTopology's own
        docstring), so sector 0's own distance-to-site grid is exact for
        every sector, not an approximation.

        :output: (near_mask, mid_mask), each [grid_n, grid_n] bool.
        """
        topo = self.topology
        n = self.grid_n
        bs_xy0, _, _ = topo.sector_basis(0)
        dist0 = np.linalg.norm(self.grid_points[0] - bs_xy0[None, :], axis=-1).reshape(n, n)
        excluded = dist0 < float(topo.min_bs_ut_dist)

        num_candidates = int((~excluded).sum())
        near_mask = (~excluded) & (dist0 < _HOTSPOT_NEAR_DIST_M)
        mid_mask = (~excluded) & (dist0 >= _HOTSPOT_NEAR_DIST_M) & (dist0 < _HOTSPOT_FAR_DIST_M)
        far_count = num_candidates - int(near_mask.sum()) - int(mid_mask.sum())
        logger.info("hotspot region LUT calibrated: n=%d near=%d mid=%d far=%d excluded=%d (radial, "
                   "%.0f/%.0f m cuts)", n, int(near_mask.sum()), int(mid_mask.sum()), far_count,
                   int(excluded.sum()), _HOTSPOT_NEAR_DIST_M, _HOTSPOT_FAR_DIST_M)
        return near_mask, mid_mask

    def _calibrate_polar_lut(self, num_azimuthal: int) -> tuple:
        """Builds a FLAT [grid_n*grid_n] cell_to_polar_bin lookup -- radial
        near/mid/far (the SAME _HOTSPOT_NEAR_DIST_M/_HOTSPOT_FAR_DIST_M
        cuts _hotspot_region_lut_radial uses) x num_azimuthal equal angular
        bins within sector 0's own wedge half-angle (centered on its
        boresight) = 3*num_azimuthal total bins -- for the polar_whole_
        neighbor/polar_region_neighbor state types (2026-10-15, explicit
        request). MEASUREMENT only (own-occupancy/SINR state features),
        NOT installed into topology.set_hotspot_region_lut -- UE PLACEMENT
        (hotspot_region_membership/hotspot_region_points, where demand-
        state clusters actually get sampled) stays on the near/mid/far
        rule regardless of this run's own state_type (see
        _calibrate_hotspot_region_lut's own docstring for why placement and
        measurement share one table there -- polar is measurement-only by
        design, there's no "polar demand state" to place clusters into).

        Always radial-cut-based (ignores topology.region_definition_method
        -- no SINR-based polar variant built) and always derived from
        sector 0's own grid, reused for every sector via the SAME cross-
        sector rotational-symmetry argument _hotspot_region_lut_radial
        already relies on.

        :output: (cell_to_polar_bin [grid_n*grid_n] int, bin_names tuple of
            len 3*num_azimuthal, e.g. "near_a0".."near_a4","mid_a0",...,
            "far_a4" for num_azimuthal=5).
        """
        topo = self.topology
        bs_xy0, e1, e2 = topo.sector_basis(0)
        xy = self.grid_points[0]  # [grid_n*grid_n, 2], already flat
        dist0 = np.linalg.norm(xy - bs_xy0[None, :], axis=-1)
        excluded = dist0 < float(topo.min_bs_ut_dist)
        # Same "excluded cells fall into the OUTER bin" convention
        # _hotspot_region_lut_radial's own near_mask/mid_mask booleans
        # produce (both False -> "far") -- replicated explicitly here since
        # there's no near_mask/mid_mask intermediate to inherit it from.
        radial_idx = np.where((~excluded) & (dist0 < _HOTSPOT_NEAR_DIST_M), 0,
                              np.where((~excluded) & (dist0 < _HOTSPOT_FAR_DIST_M), 1, 2))

        boresight = e1 + e2
        boresight = boresight / np.linalg.norm(boresight)
        half_angle = np.arccos(np.clip(np.dot(e1, boresight) / np.linalg.norm(e1), -1.0, 1.0))
        d = xy - bs_xy0[None, :]
        point_angle = np.arctan2(d[:, 1], d[:, 0])
        boresight_angle = np.arctan2(boresight[1], boresight[0])
        rel = np.arctan2(np.sin(point_angle - boresight_angle), np.cos(point_angle - boresight_angle))
        frac = np.clip(rel / half_angle, -1.0, 1.0)  # [-1, 1] across the wedge
        azimuthal_idx = np.clip(((frac + 1.0) / 2.0 * num_azimuthal).astype(int), 0, num_azimuthal - 1)

        cell_to_polar_bin = radial_idx * num_azimuthal + azimuthal_idx
        radial_names = ("near", "mid", "far")
        bin_names = tuple(f"{r}_a{a}" for r in radial_names for a in range(num_azimuthal))
        counts = np.bincount(cell_to_polar_bin, minlength=3 * num_azimuthal)
        logger.info("polar LUT calibrated: n=%d num_azimuthal=%d bins=%d counts=%s excluded=%d",
                   self.grid_n, num_azimuthal, 3 * num_azimuthal, counts.tolist(), int(excluded.sum()))
        return cell_to_polar_bin, bin_names

    def _redraw_location_reporting_mask(self) -> None:
        """(Re)derive self.location_reporting_mask (per-UE anchor status)
        from the CURRENT cluster population, per self.anchor_selection_mode
        -- see its own construction-time comment for the full rationale.
        Called once at construction and, for HotspotDemandMobility, again
        after every hotspot transition.
        """
        member_group_idx_np = self.mobility.member_group_idx.detach().cpu().numpy()
        group_ids = np.unique(member_group_idx_np)
        num_clusters = len(group_ids)
        cluster_gnss_prob = np.where(self._cluster_is_indoor, self.indoor_gnss_prob, self.outdoor_gnss_prob)

        if self.anchor_selection_mode in ("balanced", "mid_prioritized_balanced"):
            cluster_sector = self.mobility.cluster_sector
            cluster_type = self.mobility.cluster_type
            cluster_is_anchor = np.zeros(num_clusters, dtype=bool)
            for s in range(self.num_bs):
                region_idx = {rt: np.where((cluster_sector == s) & (cluster_type == rt))[0]
                             for rt in ("near", "mid", "far")}
                total_clusters_s = sum(idx.size for idx in region_idx.values())
                if total_clusters_s == 0:
                    continue
                p = float(np.mean(cluster_gnss_prob[np.concatenate(list(region_idx.values()))]))
                # Sector-wide target (2026-10-23, was per-region max(1, ...)):
                # the old rule guaranteed >=1 anchor cluster in EVERY region
                # regardless of p, which could badly overshoot the real
                # anchor fraction whenever a region was small (e.g. a
                # near-heavy state's 1-cluster "far" region still got its
                # own anchor, pushing a 20% target sector-wide up toward
                # 30%). This instead targets the sector's TRUE p fraction
                # of its total clusters, then water-fills that budget across
                # regions one cluster at a time -- always to whichever
                # region currently has the fewest anchors (ties broken by
                # dict order) among those with room left -- so it still
                # spreads as evenly as possible across near/mid/far, just
                # without inflating past the real percentage to guarantee
                # universal coverage.
                target_total = 0 if p <= 0.0 else min(total_clusters_s, max(1, int(round(p * total_clusters_s))))
                k = {rt: 0 for rt in region_idx}
                regions_to_fill = dict(region_idx)
                if self.anchor_selection_mode == "mid_prioritized_balanced":
                    # mid_prioritized_balanced (2026-10-23): mid clusters draw
                    # FIRST from the SAME sector-wide budget everyone else shares
                    # (not on top of it, and not forced regardless of budget) --
                    # with mid's cluster count this small (1/sector under the
                    # current demand states) and target_total this large (~4),
                    # mid ends up anchor almost every single time in practice,
                    # without ever pushing the overall anchor fraction above the
                    # real target the way an unconditional force would.
                    k["mid"] = min(region_idx["mid"].size, target_total)
                    target_total -= k["mid"]
                    regions_to_fill = {"near": region_idx["near"], "far": region_idx["far"]}
                remaining = target_total
                while remaining > 0:
                    candidates = [rt for rt, idx in regions_to_fill.items() if k[rt] < idx.size]
                    if not candidates:
                        break
                    min_k = min(k[rt] for rt in candidates)
                    tied = [rt for rt in candidates if k[rt] == min_k]
                    # Random tie-break (not just the first region in dict
                    # order) -- otherwise, whenever the budget runs out
                    # before every region gets a turn, the SAME region
                    # (near, by iteration order) always wins and the SAME
                    # region (far) always loses, every sector, every
                    # interval -- a structural bias, not an even spread.
                    rt = tied[int(self._anchor_rng.integers(len(tied)))] if len(tied) > 1 else tied[0]
                    k[rt] += 1
                    remaining -= 1
                for region_type, idx in region_idx.items():
                    if k[region_type] > 0:
                        chosen = self._anchor_rng.choice(idx, size=k[region_type], replace=False)
                        cluster_is_anchor[chosen] = True
        else:
            cluster_is_anchor = self._anchor_rng.random(num_clusters) < cluster_gnss_prob

        location_reporting_mask = np.zeros(self.num_ut, dtype=bool)
        for gi, group_id in enumerate(group_ids):
            location_reporting_mask[member_group_idx_np == group_id] = cluster_is_anchor[gi]
        self.location_reporting_mask = location_reporting_mask
        logger.debug("_redraw_location_reporting_mask: mode=%s anchor_clusters=%d/%d anchor_ue_frac=%.3f",
                    self.anchor_selection_mode, int(cluster_is_anchor.sum()), num_clusters,
                    float(location_reporting_mask.mean()))

    def _hotspot_region_lut_sinr(self) -> tuple:
        """SINR-measurement-based alternative (2026-10-12) -- classifies
        each grid cell by its own simulated SINR under a FROZEN reference
        condition: every sector's tilt fixed at 9 deg (the project's own
        established nominal tilt -- see scripts/debug_pipeline_audit.py's
        SINGLE_SECTOR_NOMINAL_TILT_DEG), forced LOS, shadow fading
        disabled, every virtual UE outdoor -- deterministic and
        INDEPENDENT of this run's own channel.force_los/
        disable_shadow_fading, same reasoning the project's original
        (pre-2026-09-29) SINR-value rule used: the region definition
        itself shouldn't shift depending on this run's own stochastic
        channel draw. near: SINR > 5 dB; mid: 0-5 dB; far: < 0 dB -- same
        0/5 dB convention scripts/debug_pipeline_audit.py's own
        COVERAGE_SWITCHING_REGION_EDGES already uses.

        A dedicated, throwaway channel model + LargeScaleChannel (NOT
        self.channel_model) does the calibration measurement -- so this
        never touches self.channel_model's own topology/shape state (no
        reset_topology() dance needed before real mobility-driven UEs
        start using it), and can force enable_shadow_fading=False
        independent of this run's own disable_shadow_fading setting (that
        flag is fixed at channel-model CONSTRUCTION time, not swappable
        per set_topology() call). self.kpi_manager itself IS reused, since
        it only consumes a LargeScaleState + a tilt vector -- it doesn't
        care which channel model produced that state.

        :output: (near_mask, mid_mask), each [grid_n, grid_n] bool.
        """
        REFERENCE_TILT_DEG = 9.0
        NEAR_SINR_DB, MID_SINR_DB = 5.0, 0.0

        topo = self.topology
        n = self.grid_n
        calib_channel_model = self._channel_model_cls(**{
            **self._channel_model_kwargs, "enable_shadow_fading": False})
        calib_large_scale_channel = LargeScaleChannel(calib_channel_model)

        # Sector 0's own grid ONLY (841 cells, not num_bs*841=17661) -- same
        # cross-sector-symmetry argument _hotspot_region_lut_radial already
        # relies on (every sector's rhombus geometry is identical up to
        # rotation). Interference from every OTHER real sector is still
        # fully resolved regardless (resolve_power_at_tilt sums over every
        # one of self.num_bs transmitting sectors for each virtual UE
        # queried, independent of how many virtual UEs there are) -- this
        # only reduces the QUERY-point count, not which sectors interfere.
        # Using the full num_bs*841 set instead blows up Sionna's own O(N^2)
        # LSP spatial-correlation matrix (confirmed: 24+ GiB CUDA OOM at
        # 17661 virtual UEs).
        xy = self.grid_points[0].reshape(-1, 2)
        num_virtual = xy.shape[0]
        z = np.full((num_virtual, 1), self.ut_height)
        virtual_loc = torch.tensor(np.concatenate([xy, z], axis=1),
                                   dtype=topo.bs_loc.dtype, device=topo.bs_loc.device).unsqueeze(0)
        bs_virtual_loc = topo.mirror_bs_loc(virtual_loc)
        zeros = torch.zeros(1, num_virtual, 3, dtype=topo.bs_loc.dtype, device=topo.bs_loc.device)
        in_state = torch.zeros(1, num_virtual, dtype=torch.bool, device=topo.bs_loc.device)  # all outdoor
        calib_channel_model.set_topology(virtual_loc, topo.bs_loc, zeros, topo.bs_orientations,
                                         zeros, in_state, True, bs_virtual_loc)  # True: forced LOS
        state = calib_large_scale_channel.generate_state()

        tilt_vec = np.full(self.num_bs, REFERENCE_TILT_DEG)
        power_w = self.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]
        kpis = self.kpi_manager.compute_ue_kpis(power_w[None, :, :])
        sinr_db = kpis["sinr_db"][0].reshape(n, n)  # already sector 0's own grid only (841 points)

        bs_xy0, _, _ = topo.sector_basis(0)
        dist0 = np.linalg.norm(self.grid_points[0] - bs_xy0[None, :], axis=-1).reshape(n, n)
        excluded = dist0 < float(topo.min_bs_ut_dist)
        num_candidates = int((~excluded).sum())
        near_mask = (~excluded) & (sinr_db > NEAR_SINR_DB)
        mid_mask = (~excluded) & (sinr_db > MID_SINR_DB) & (sinr_db <= NEAR_SINR_DB)
        far_count = num_candidates - int(near_mask.sum()) - int(mid_mask.sum())
        logger.info("hotspot region LUT calibrated: n=%d near=%d mid=%d far=%d excluded=%d (SINR-based, "
                   "tilt=%.1f deg, %.0f/%.0f dB cuts)", n, int(near_mask.sum()), int(mid_mask.sum()),
                   far_count, int(excluded.sum()), REFERENCE_TILT_DEG, NEAR_SINR_DB, MID_SINR_DB)
        return near_mask, mid_mask

    def _build_region_idx(self, sector_idx: np.ndarray, cell_idx: np.ndarray,
                          power_db_all_sectors: np.ndarray) -> np.ndarray:
        """non_anchor_region_handling="rme" only: per-UE own-region HARD
        assignment for RegionalInfoProvider.accumulate's region_idx_override
        -- its own real region for a UE whose location IS known (anchor);
        argmax(SpatialGridEstimator.infer_region_probabilities(...), from
        THIS SAME turn's anchor-only Kriging fit) for one whose location
        isn't (non-anchor). Every UE ends up as exactly one full unit in
        exactly one region, anchor and non-anchor alike -- no fractional/
        probabilistic splitting. NOT the default (see config.yaml's own
        non_anchor_region_handling comment for why) -- kept for after
        region_inference_sigma_db gets recalibrated.

        :param power_db_all_sectors: [num_bs, num_ue] every sector's own
            received power at every UE -- this turn's own
            drl_process_turn already computed it for the spatial
            estimator; reused here rather than recomputed.
        """
        cell_to_region = self.regional_info_provider.cell_to_region
        num_own_regions = self.regional_info_provider.num_own_regions
        region_idx = cell_to_region[np.clip(cell_idx, 0, len(cell_to_region) - 1)].copy()

        non_reporting = ~self.location_reporting_mask
        if non_reporting.any():
            safe_sector_idx = np.clip(sector_idx, 0, self.num_bs - 1)
            measured_rsrp_db = power_db_all_sectors[safe_sector_idx, np.arange(len(sector_idx))]
            region_probs = self.spatial_grid_estimator.infer_region_probabilities(
                safe_sector_idx[non_reporting], measured_rsrp_db[non_reporting],
                cell_to_region, num_own_regions, self.region_inference_sigma_db)
            region_idx[non_reporting] = region_probs.argmax(axis=1)
        return region_idx

    def _build_region_idx_radio_map(self, sector_idx: np.ndarray, cell_idx: np.ndarray,
                                    rsrp_db: np.ndarray, sinr_db: np.ndarray) -> np.ndarray:
        """non_anchor_region_handling="radio_map" only: per-UE own-region
        HARD assignment for RegionalInfoProvider.accumulate's
        region_idx_override -- its own real region for an anchor UE
        (location known). With anchor_selection_mode="mid_prioritized_balanced"
        (see _redraw_location_reporting_mask), mid clusters draw anchor
        status FIRST from each sector's budget, so a non-anchor UE is
        near-always near or far, essentially never mid in practice -- this
        function only ever guesses near-vs-far for a non-anchor UE (never
        mid), so on the rare sector-interval where mid's budget draw
        happens to fail, a true-mid non-anchor UE would simply be
        misclassified as near/far like any other near/far UE, not crash.
        For a non-anchor UE, each sector's near/far
        RSRP+SINR mean/std is built fresh THIS turn from EVERY UE in that
        sector (full simulator ground-truth access, not anchor-limited --
        a deliberately idealized "radio map given directly by the
        simulator", see config.yaml's own non_anchor_region_handling
        comment for why this differs from "rme"/Kriging's honest
        anchor-only constraint), then the UE is assigned to whichever of
        near/far minimizes standardized distance
        (((rsrp-mean)/std)^2 + ((sinr-mean)/std)^2). Validated offline at
        ~95-96% accuracy on this exact near/far split.
        """
        cell_to_region = self.regional_info_provider.cell_to_region
        region_idx = cell_to_region[np.clip(cell_idx, 0, len(cell_to_region) - 1)].copy()

        non_reporting = ~self.location_reporting_mask
        if not non_reporting.any():
            return region_idx

        region_names = self.regional_info_provider.region_names
        near_idx = region_names.index("near")
        far_idx = region_names.index("far")
        safe_sector_idx = np.clip(sector_idx, 0, self.num_bs - 1)

        for s in range(self.num_bs):
            own = safe_sector_idx == s
            own_nonreport = own & non_reporting
            if not own_nonreport.any():
                continue
            own_region = region_idx[own]
            own_rsrp, own_sinr = rsrp_db[own], sinr_db[own]
            near_mask, far_mask = own_region == near_idx, own_region == far_idx
            if near_mask.sum() < 2 or far_mask.sum() < 2:
                # Too few ground-truth points to build this sector's centroids
                # this turn -- fall back to the uninformed default (near),
                # same spirit as infer_region_probabilities' own uniform
                # fallback: never silently leak the true answer.
                region_idx[own_nonreport] = near_idx
                continue
            near_rsrp_mean, near_rsrp_std = own_rsrp[near_mask].mean(), max(own_rsrp[near_mask].std(), 1e-3)
            near_sinr_mean, near_sinr_std = own_sinr[near_mask].mean(), max(own_sinr[near_mask].std(), 1e-3)
            far_rsrp_mean, far_rsrp_std = own_rsrp[far_mask].mean(), max(own_rsrp[far_mask].std(), 1e-3)
            far_sinr_mean, far_sinr_std = own_sinr[far_mask].mean(), max(own_sinr[far_mask].std(), 1e-3)

            rsrp_q, sinr_q = rsrp_db[own_nonreport], sinr_db[own_nonreport]
            dist2_near = ((rsrp_q - near_rsrp_mean) / near_rsrp_std) ** 2 + \
                        ((sinr_q - near_sinr_mean) / near_sinr_std) ** 2
            dist2_far = ((rsrp_q - far_rsrp_mean) / far_rsrp_std) ** 2 + \
                       ((sinr_q - far_sinr_mean) / far_sinr_std) ** 2
            region_idx[own_nonreport] = np.where(dist2_near <= dist2_far, near_idx, far_idx)
        return region_idx

    def _oracle_single_sector_step(self, power_table: np.ndarray, current_tilt_idx: np.ndarray,
                                   sector: int, threshold_db: float, rsrp_threshold_dbm: float = None):
        """Diagnostics-only oracle/counterfactual: a single-sector marginal
        sweep, standalone and pure/stateless -- "holding every other
        sector's CURRENT tilt fixed, which candidate tilt for THIS sector
        alone maximizes network coverage right now, and what would
        coverage be for each candidate?" (see this project's own
        DRL-diagnostics implementation plan).

        :param power_table: [num_tilts, batch, num_sectors, num_ue], from
            kpi_manager.compute_tilt_sector_ue_rx_power(state,
            downtilt_sweep_deg) -- safe to reuse across every sector's own
            turn this interval since it only depends on `state` (unchanged
            within an interval), not on any sector's chosen tilt index.
        :param current_tilt_idx: [num_sectors] int -- every OTHER sector's
            assignment held fixed at this (typically self.drl_controller.
            tilt_idx, which stays index-for-index in sync with
            live_tilt_deg throughout the async round -- see
            run_simulation's own construction of both).
        :output: (best_tilt_idx: int, coverage_per_candidate: [num_tilts] float).
        """
        num_tilts = power_table.shape[0]
        coverage_per_candidate = np.empty(num_tilts)
        trial = np.asarray(current_tilt_idx, dtype=int).copy()
        for t in range(num_tilts):
            trial[sector] = t
            coverage_per_candidate[t] = self.kpi_manager.compute_ue_kpis(
                power_table, trial, threshold_db, rsrp_threshold_dbm)["coverage"]
        best_tilt_idx = int(np.argmax(coverage_per_candidate))
        return best_tilt_idx, coverage_per_candidate

    def _measure_and_pool(self, state, sector_idx, cell_idx, live_tilt_deg, closes) -> tuple:
        """One real measurement: resolves power at `live_tilt_deg`,
        accumulates it into region/reward state, returns
        RegionalInfoProvider.compute(closes)'s own tuple PLUS one extra
        trailing element, covered_count_total: float -- the network-wide
        RAW (unnormalized) covered-UE count this measurement, read from
        self.regional_info_provider.sector_covered_count.sum() BEFORE
        compute() resets it. reward_type="delta_covered" needs this raw
        total from BOTH calls (see drl_process_turn) to build its own
        pre/post delta -- RegionalInfoProvider.compute()'s own "reward"
        output is already divided by num_ut (see its own docstring), so
        it alone can't reconstruct an unnormalized delta.

        state_type="cell_aligned" ADDITIONALLY accumulates+computes
        self.cell_aligned_info_provider's own per-cell state this SAME
        measurement, reusing the already-computed sector_idx/cell_idx/
        covered_ue/sinr_db (no extra radio computation) -- its own 4
        outputs are appended as trailing tuple elements, always None when
        state_type="region_aligned" (the provider isn't even
        constructed then). REWARD is entirely unaffected either way --
        always from RegionalInfoProvider/covered_count_total above.

        Called TWICE per DRL turn (see drl_process_turn) -- once before
        this turn's own decision, once after -- each call is a self-
        contained "turn" in RegionalInfoProvider's (and, when active,
        CellAlignedInfoProvider's) own sense (accumulate() then compute(),
        counters reset at the end), reused here for the pre/post bracket.
        """
        power_w = self.kpi_manager.resolve_power_at_tilt(state, live_tilt_deg)[0]

        # Per-UE SINR only -- coverage/reward are geographic-grid-based, not served-sector-based.
        kpis = self.kpi_manager.compute_ue_kpis(power_w[None, :, :])
        sinr_db = kpis["sinr_db"][0]
        rsrp_db = kpis["rsrp_dbm"][0]
        covered_ue = sinr_db > self.coverage_threshold_db
        if self.rsrp_threshold_dbm is not None:
            covered_ue = covered_ue & (kpis["rsrp_dbm"][0] > self.rsrp_threshold_dbm)

        # non_anchor_region_handling picks how a non-anchor UE's own region
        # is decided for the OWN-REGION breakdown (own_occupancy/
        # own_sinr_*) -- config.yaml switchable, see its own comment.
        # Reward and whole-sector coarse (all_occupancy/all_sinr_p50)
        # always see the REAL sector_idx/cell_idx regardless (serving
        # sector is ordinary connectivity, never actually unknown) --
        # neither knob below touches those.
        own_region_mask, region_idx_override = None, None
        if self.non_anchor_region_handling == "exclude":
            own_region_mask = self.location_reporting_mask
        elif self.non_anchor_region_handling == "radio_map":
            region_idx_override = self._build_region_idx_radio_map(sector_idx, cell_idx, rsrp_db, sinr_db)
        elif self.spatial_grid_estimator is not None:  # "rme"
            power_db_all_sectors = 10.0 * np.log10(power_w)
            reporting_sector_idx = np.where(self.location_reporting_mask, sector_idx, -1)
            self.spatial_grid_estimator.accumulate(reporting_sector_idx, cell_idx, covered_ue,
                                                   power_db_all_sectors=power_db_all_sectors)
            if not self.location_reporting_mask.all():
                region_idx_override = self._build_region_idx(sector_idx, cell_idx, power_db_all_sectors)
            self.spatial_grid_estimator.reset()

        self.regional_info_provider.accumulate(sector_idx, cell_idx, covered_ue, sinr_db,
                                               own_region_mask=own_region_mask,
                                               region_idx_override=region_idx_override)
        covered_count_total = float(self.regional_info_provider.sector_covered_count.sum())
        region_result = self.regional_info_provider.compute(closes)

        cell_result = (None, None, None, None)
        if self.cell_aligned_info_provider is not None:
            # CellAlignedInfoProvider.accumulate takes TWO different masks now
            # (2026-10-24, see its own docstring/module docstring):
            #   own_region_mask/region_idx_override -- passed through UNCHANGED,
            #     same values RegionalInfoProvider gets -- feeds ONLY the
            #     far-ring-only neighbor coarse pair, which (like
            #     RegionalInfoProvider's own all_occupancy/all_sinr_p50) CAN use
            #     an inferred region, since it's a coarse near/mid/far question.
            #   cell_own_mask -- gates the per-FINE-CELL own-occupancy breakdown
            #     specifically, which can't use an inferred region at all (no
            #     per-cell mapping for a coarse guess). Falls back to the honest
            #     exclude-non-anchor mask whenever a region is being inferred
            #     (own_region_mask=None there would otherwise be read as "no
            #     filtering at all" by the per-cell logic -- found 2026-10-24
            #     when a 20%-anchor radio_map run and a 100%-anchor run produced
            #     byte-identical coverage over a full training run).
            cell_own_mask = self.location_reporting_mask if region_idx_override is not None \
                else own_region_mask
            self.cell_aligned_info_provider.accumulate(sector_idx, cell_idx, covered_ue, sinr_db,
                                                       own_region_mask=own_region_mask,
                                                       region_idx_override=region_idx_override,
                                                       cell_own_mask=cell_own_mask)
            cell_result = self.cell_aligned_info_provider.compute()

        # polar_whole_neighbor/polar_region_neighbor (2026-10-15): own
        # polar-binned occupancy/SINR, SAME anchor-gating (own_region_mask/
        # region_idx_override) as region_aligned/cell_aligned's own-state
        # above -- only the FIRST 4 of RegionalInfoProvider.compute()'s own
        # 8 outputs are kept (own_occupancy/own_sinr_p25/50/75); its own
        # all_occupancy/all_sinr_p50/reward/state_ready are unused
        # duplicates (reward always comes from self.regional_info_provider
        # above, regardless of state_type).
        polar_result = (None, None, None, None)
        if self.polar_info_provider is not None:
            self.polar_info_provider.accumulate(sector_idx, cell_idx, covered_ue, sinr_db,
                                                own_region_mask=own_region_mask,
                                                region_idx_override=region_idx_override)
            polar_result = self.polar_info_provider.compute(closes)[:4]

        return (*region_result, covered_count_total, *cell_result, *polar_result)

    def _build_state_observations(self, tilt_norm, own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75,
                                  all_occupancy, all_sinr_p50, cell_own_occupancy, cell_own_sinr_p50,
                                  cell_all_occupancy, cell_all_sinr_p50, polar_own_occupancy, polar_own_sinr_p25,
                                  polar_own_sinr_p50, polar_own_sinr_p75) -> np.ndarray:
        """Dispatches to the right compute_*_state builder for self.state_type
        (2026-10-15: factored out once a 3rd/4th state_type made inlining
        this at both the pre- and post-decision measurement sites in
        drl_process_turn repeat the same 4-way branch twice). All args are
        this SAME measurement's own already-computed outputs (from
        _measure_and_pool) -- this function does no measurement of its own.
        """
        if self.state_type == "cell_aligned":
            return compute_cell_state(
                own_tilt_norm=tilt_norm, own_occupancy=cell_own_occupancy, own_sinr_p50=cell_own_sinr_p50,
                all_occupancy=cell_all_occupancy, all_sinr_p50=cell_all_sinr_p50,
                neighbor_ids=self.topology.neighbor_ids,
            )
        if self.state_type == "polar_whole_neighbor":
            # compute_region_state's own "own" block is shape-generic (see
            # its own docstring) -- reused as-is with polar-shaped arrays;
            # the neighbor block (all_occupancy/all_sinr_p50, whole-sector
            # scalar) is identical to region_aligned's own.
            return compute_region_state(
                own_tilt_norm=tilt_norm, own_occupancy=polar_own_occupancy,
                own_sinr_p25=polar_own_sinr_p25, own_sinr_p50=polar_own_sinr_p50, own_sinr_p75=polar_own_sinr_p75,
                all_occupancy=all_occupancy, all_sinr_p50=all_sinr_p50,
                neighbor_ids=self.topology.neighbor_ids,
            )
        if self.state_type == "polar_region_neighbor":
            # own = polar bins; neighbor = each real neighbor's OWN
            # region-level (near/mid/far) breakdown (own_occupancy/
            # own_sinr_p50 from self.regional_info_provider, which always
            # runs regardless of state_type for reward) -- NOT the
            # whole-sector scalar.
            return compute_polar_regional_neighbor_state(
                own_tilt_norm=tilt_norm, own_occupancy=polar_own_occupancy,
                own_sinr_p25=polar_own_sinr_p25, own_sinr_p50=polar_own_sinr_p50, own_sinr_p75=polar_own_sinr_p75,
                neighbor_region_occupancy=own_occupancy, neighbor_region_sinr_p50=own_sinr_p50,
                neighbor_ids=self.topology.neighbor_ids,
            )
        return compute_region_state(
            own_tilt_norm=tilt_norm, own_occupancy=own_occupancy,
            own_sinr_p25=own_sinr_p25, own_sinr_p50=own_sinr_p50, own_sinr_p75=own_sinr_p75,
            all_occupancy=all_occupancy, all_sinr_p50=all_sinr_p50,
            neighbor_ids=self.topology.neighbor_ids,
        )

    def drl_process_turn(self, state, sector_idx, cell_idx, closes, live_tilt_deg, training: bool = True,
                        power_table: np.ndarray = None):
        """One DRL decision turn, no-lag: for whichever sector(s) `closes`
        marks, decides a NEW tilt from a REAL pre-decision measurement
        (this interval's own already-drawn channel state, resolved at
        `live_tilt_deg` as it stands right now -- under async_schedule=
        "async", already reflecting every earlier-this-round turn's own
        new decision), applies it, then takes a SECOND, post-decision
        measurement to credit that exact decision with its own real,
        immediate consequence -- no lag between decision and evaluation,
        for either "sync" or "async" scheduling. Doubles the per-turn
        resolve/accumulate cost (see _measure_and_pool) versus a single-
        measurement design, in exchange for a genuine same-turn (s, a, r,
        s') tuple with no forecasting and no stale cross-interval cache.

        :param state: this interval's single LargeScaleState -- shared and
            unchanged across every turn within the interval; only the
            resolved power differs per turn/measurement (live_tilt_deg differs).
        :param sector_idx, cell_idx: [num_ue] this interval's fixed
            geographic grid assignment (CellularTopology.
            assign_to_sector_rhombus_grid) -- UE positions don't move
            within an interval, so this is computed once and reused across
            every turn/measurement.
        :param closes: [num_bs] bool -- which sector(s)' decision this turn
            applies to: all-True for async_schedule="sync" (one turn covers
            the whole network); exactly one True for "async" (each of the
            num_bs sequential per-sector turns -- see run_simulation).
        :param live_tilt_deg: [num_bs] float -- MUTATED IN PLACE for the
            `closes` sector(s), to the newly-decided tilt, right after
            decide() -- the caller doesn't need its own post-call update
            anymore (contrast with the old single-measurement version).
        :param training: passed straight through to
            RLTiltController.decide/learn -- see helpers/tilt_controller.py.
        :param power_table: [num_tilts, batch, num_sectors, num_ue], optional
            -- diagnostics_logger's own per-decision oracle/counterfactual
            comparison (see _oracle_single_sector_step) only runs when this
            is given; None is a no-op there, same as diagnostics_logger
            itself being None.
        """
        logger.function("drl_process_turn start: closes=%d", int(closes.sum()))

        # === PRE-decision measurement: real, this-interval data, at the
        # tilt vector as it stands before this turn's own decision. ===
        (own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75,
         all_occupancy, all_sinr_p50, _pre_reward, pre_state_ready, pre_covered_total,
         cell_own_occupancy, cell_own_sinr_p50, cell_all_occupancy, cell_all_sinr_p50,
         polar_own_occupancy, polar_own_sinr_p25, polar_own_sinr_p50, polar_own_sinr_p75) = (
            self._measure_and_pool(state, sector_idx, cell_idx, live_tilt_deg, closes))
        pre_tilt_norm = (live_tilt_deg - self.tilt_deg_min) / (self.tilt_deg_max - self.tilt_deg_min)
        pre_observations = self._build_state_observations(
            pre_tilt_norm, own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75, all_occupancy, all_sinr_p50,
            cell_own_occupancy, cell_own_sinr_p50, cell_all_occupancy, cell_all_sinr_p50,
            polar_own_occupancy, polar_own_sinr_p25, polar_own_sinr_p50, polar_own_sinr_p75)
        logger.debug("drl_process_turn: pre-decision observation shape=%s", pre_observations.shape)

        # diagnostics: snapshot BEFORE decide() below overwrites tilt_idx --
        # this turn's oracle/counterfactual comparison needs the CURRENT
        # (pre-decision) assignment as the "other sectors held fixed"
        # vector (see _oracle_single_sector_step's own docstring).
        diag_pending = None
        if self.diagnostics_logger is not None:
            diag_pending = {}
            for s in np.where(closes)[0]:
                oracle_action_idx, oracle_coverage_per_candidate = None, None
                if power_table is not None and self.diagnostics_log_oracle_comparison:
                    oracle_action_idx, oracle_coverage_per_candidate = self._oracle_single_sector_step(
                        power_table, self.drl_controller.tilt_idx, s, self.coverage_threshold_db,
                        self.rsrp_threshold_dbm)
                diag_pending[s] = dict(
                    tilt_before_deg=float(live_tilt_deg[s]),
                    action_before_idx=int(self.drl_controller.tilt_idx[s]),
                    oracle_action_idx=oracle_action_idx,
                    oracle_coverage_per_candidate=oracle_coverage_per_candidate,
                )

        # === Decide: new tilt(s) for `closes`, from the real pre-decision
        # state; apply immediately so a later async turn this same round
        # sees it (genuine Gauss-Seidel propagation). ===
        self.drl_controller.decide(pre_observations, training=training, has_data=pre_state_ready, schedule=closes)
        for s in np.where(closes)[0]:
            live_tilt_deg[s] = self.drl_action_grid_deg[self.drl_controller.tilt_idx[s]]

        # === POST-decision measurement: the real, immediate consequence of
        # the decision just made. ===
        (own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75,
         all_occupancy, all_sinr_p50, reward_per_sector, post_state_ready, post_covered_total,
         cell_own_occupancy, cell_own_sinr_p50, cell_all_occupancy, cell_all_sinr_p50,
         polar_own_occupancy, polar_own_sinr_p25, polar_own_sinr_p50, polar_own_sinr_p75) = (
            self._measure_and_pool(state, sector_idx, cell_idx, live_tilt_deg, closes))
        if self.reward_type == "delta_covered":
            # RAW (unnormalized) change in network-wide covered-UE count
            # this turn's decision(s) caused -- reuses reward_per_sector's
            # own NaN mask (RegionalInfoProvider.compute()'s "nobody
            # visited anywhere this turn" gate) rather than recomputing it,
            # just substitutes the delta for the ratio value wherever it's
            # not NaN (see _measure_and_pool's own docstring for why the
            # ratio alone can't reconstruct this).
            delta_covered = post_covered_total - pre_covered_total
            reward_per_sector = np.where(np.isnan(reward_per_sector), np.nan, delta_covered)
        # Only real training data once reward AND state are both defined (see RegionalInfoProvider.compute).
        has_data = ~np.isnan(reward_per_sector) & post_state_ready
        reward = compute_reward(self.reward_type, reward_per_sector)
        self._interval_last_reward[closes] = reward[closes]
        post_tilt_norm = (live_tilt_deg - self.tilt_deg_min) / (self.tilt_deg_max - self.tilt_deg_min)
        post_observations = self._build_state_observations(
            post_tilt_norm, own_occupancy, own_sinr_p25, own_sinr_p50, own_sinr_p75, all_occupancy, all_sinr_p50,
            cell_own_occupancy, cell_own_sinr_p50, cell_all_occupancy, cell_all_sinr_p50,
            polar_own_occupancy, polar_own_sinr_p25, polar_own_sinr_p50, polar_own_sinr_p75)
        logger.debug("drl_process_turn: post-decision observation shape=%s", post_observations.shape)

        # === Learn: credits the decision just made -- decide() above
        # already set RLTiltController's own _prev_observations/
        # _prev_actions[closes] to (pre_observations, new action), so this
        # learn() call forms exactly that (s, a, r, s') tuple, same turn,
        # no lag. ===
        losses_before = len(self.drl_policy.step_losses)
        self.drl_controller.learn(post_observations, reward, training=training, has_data=has_data, schedule=closes)
        if len(self.drl_policy.step_losses) > losses_before:
            self._interval_step_losses.extend(self.drl_policy.step_losses[losses_before:])

        if diag_pending is not None:
            new_losses = self.drl_policy.step_losses[losses_before:]
            new_loss_sectors = self.drl_policy.step_loss_sectors[losses_before:]
            for s, pending in diag_pending.items():
                sector_losses = [loss for loss, sec in zip(new_losses, new_loss_sectors) if sec == s]
                decision_info = self.drl_policy.last_decision_info.get(s)
                self.diagnostics_logger.log_decision(
                    sector=s, training=training, state=pre_observations[s],
                    tilt_before_deg=pending["tilt_before_deg"],
                    action_before_idx=pending["action_before_idx"],
                    action_idx=int(self.drl_controller.tilt_idx[s]),
                    tilt_after_deg=float(self.drl_action_grid_deg[self.drl_controller.tilt_idx[s]]),
                    reward=(reward[s] if has_data[s] else None), has_data=bool(has_data[s]),
                    next_state=post_observations[s],
                    decision_info=decision_info,
                    oracle_action_idx=pending["oracle_action_idx"],
                    oracle_coverage_per_candidate=pending["oracle_coverage_per_candidate"],
                    td_losses=sector_losses,
                )
        logger.function("drl_process_turn end")

    def step_environment(self, is_last_interval):
        """This interval's single measurement: THIS (pre-step) instant's
        mobility position, plus a fresh large-scale channel draw at that
        position. Mobility only advances afterward (never past the very
        last measurement of the run), one step == one tilt-control interval.

        The anchor/location-reporting mask is redrawn HERE, before this
        interval's own measurement -- reacting to a transition flagged by
        the PREVIOUS call's mobility.step(), not right after that
        mobility.step() call itself (the old placement). A transition
        reassigns cluster_type/anchor eligibility for clusters whose
        members' positions only move as a result of that SAME step()
        call; redrawing immediately there paired the NEW anchor
        assignment with the OLD, not-yet-advanced positions THAT SAME
        call had just measured -- a one-interval mismatch between
        own_region_mask and the sector_idx/cell_idx/state it's applied
        to in _measure_and_pool, on every hotspot-hold transition
        interval (own_occupancy/own_sinr_* built from stale positions
        filtered by the new membership). Checking the flag here instead
        keeps the mask and the positions it filters always from the same
        instant: this call's own mobility.step() (below) is what makes
        the flag's positions real, so the redraw for THAT transition
        belongs at the top of the NEXT call, right before its
        generate_state().

        :output: this interval's LargeScaleState.
        """
        logger.function("step_environment start: is_last_interval=%s", is_last_interval)
        if getattr(self.mobility, "transitioned_this_step", False):
            self._redraw_location_reporting_mask()

        ut_loc = self.mobility.ut_loc.unsqueeze(0)  # [1, num_ut, 3]
        bs_virtual_loc = self.topology.mirror_bs_loc(ut_loc)

        self.channel_model.set_topology(
            ut_loc, self.topology.bs_loc, self.ut_orientations, self.topology.bs_orientations,
            self.ut_velocities, self.in_state, self.force_los, bs_virtual_loc,
        )
        if self.sticky_los is not None:
            self.sticky_los.apply(
                self.channel_model, self.mobility.member_group_idx.detach().cpu().numpy(),
                self.mobility.ref_xy.detach().cpu().numpy())
        state = self.large_scale_channel.generate_state()

        if not is_last_interval:
            self.mobility.step()

        logger.function("step_environment end")
        return state

    def _compute_ue_kpis_tracked(self, name: str, power_w, **kwargs) -> dict:
        """compute_ue_kpis, transparently threading algorithm `name`'s own
        persistent A3-hysteresis serving state through (self._serving_idx,
        see its own docstring and config.yaml's kpi.ho_mode comment) --
        so execute_algorithms's real per-interval calls don't each repeat
        the same read-state/call/write-state-back pattern. Under
        ho_mode="max_rsrp" this is a pure passthrough to compute_ue_kpis,
        byte-for-byte the original stateless behavior (ho_mode="a3_hysteresis"
        is the default as of 2026-09-22 -- see config.yaml's own comment).

        :param name: one of self._serving_idx's own keys ("adaptive_legacy", "drl").
        :param power_w, kwargs: forwarded to compute_ue_kpis as-is.
        """
        if self.ho_mode == "a3_hysteresis":
            kwargs = dict(kwargs, previous_serving_idx=self._serving_idx[name],
                          hysteresis_db=self.a3_hysteresis_db)
        kpis = self.kpi_manager.compute_ue_kpis(power_w, **kwargs)
        if self.ho_mode == "a3_hysteresis":
            self._serving_idx[name] = kpis["serving_idx"]
        return kpis

    def execute_algorithms(self, measurement, interval, drl_tilt_deg_this_interval):
        """Score every algorithm once against this interval's measurement,
        update history arrays at index `interval`. DRL scores at
        drl_tilt_deg_this_interval (the interval's STARTING tilt) always,
        regardless of async_schedule -- same single-tilt-per-interval
        convention as every other algorithm.
        """
        logger.function("execute_algorithms start: interval=%d", interval)
        adaptive_power_w = measurement["adaptive_power_w"]
        drl_power_w = measurement["drl_power_w"]
        ut_loc = measurement["ut_loc"]

        # Adaptive Legacy: score at the tilt already in effect, THEN update
        self.adaptive_legacy_tilt_deg_history[interval] = self.adaptive_legacy_controller.tilt_deg.copy()
        adaptive_kpis = self._compute_ue_kpis_tracked(
            "adaptive_legacy", adaptive_power_w[None, :, :], threshold_db=self.coverage_threshold_db,
            rsrp_threshold_dbm=self.rsrp_threshold_dbm,
            ut_loc=ut_loc[None, :, :], edge_percentile=self.adaptive_legacy_edge_percentile,
            distance_threshold_m=self.distance_threshold_m, overlap_margin_db=self.adaptive_legacy_overlap_margin_db,
        )
        self.coverage_adaptive_legacy[interval] = adaptive_kpis["coverage"]
        self.sinr_percentiles_adaptive_legacy[:, interval] = np.percentile(adaptive_kpis["sinr_db"],
                                                                           self.sinr_percentiles)
        adaptive_has_data = ~(np.isnan(adaptive_kpis["overshoot"]) | np.isnan(adaptive_kpis["per_sector_coverage"]))
        self.adaptive_legacy_controller.update(adaptive_kpis["overshoot"], adaptive_kpis["per_sector_coverage"],
                                               has_data=adaptive_has_data)
        logger.debug("execute_algorithms: adaptive_legacy coverage=%.4f", adaptive_kpis["coverage"])

        # DRL
        self.drl_tilt_deg_history[interval] = drl_tilt_deg_this_interval
        drl_kpis = self._compute_ue_kpis_tracked(
            "drl", drl_power_w[None, :, :], threshold_db=self.coverage_threshold_db,
            rsrp_threshold_dbm=self.rsrp_threshold_dbm,
            ut_loc=ut_loc[None, :, :],
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
        """The per-interval loop: draw one measurement, run DRL's turn(s)
        (drl_process_turn -- one "sync" turn or num_bs sequential "async"
        turns), then execute_algorithms() scores every algorithm.

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

        # For each tilt control interval
        for interval in range(self.num_tilt_control_intervals):
            is_last_interval = interval == self.num_tilt_control_intervals - 1

            self.position_history[interval] = self.mobility.ut_loc.detach().cpu().numpy().copy()
            if self.has_clusters:
                self.ref_xy_history[interval] = self.mobility.ref_xy.detach().cpu().numpy().copy()
            ut_loc_this_interval = self.mobility.ut_loc.detach().cpu().numpy().copy()

            # Draw this interval's single measurement.
            state = self.step_environment(is_last_interval)
            sector_idx, cell_idx = self.topology.assign_to_sector_rhombus_grid(ut_loc_this_interval[:, :2])

            # power_table: every sector's power at every candidate tilt --
            # needed by per-decision oracle/counterfactual diagnostics
            # (diagnostics_log_oracle_comparison) only. Built once per
            # interval, BEFORE the DRL turn(s) below (not just for
            # execute_algorithms() afterward), since per-turn diagnostics
            # need it DURING the async loop.
            power_table = (self.kpi_manager.compute_tilt_sector_ue_rx_power(state, self.downtilt_sweep_deg)
                          if self.diagnostics_enabled and self.diagnostics_log_oracle_comparison
                          else None)

            if self.diagnostics_logger is not None:
                self.diagnostics_logger.start_interval(interval)

            # DRL's own tilt vector for THIS interval -- mutated in place by
            # drl_process_turn's own decide() calls below (see its own
            # docstring for the no-lag, decide-then-remeasure mechanism),
            # so after the turn(s) complete it reflects the decision(s)
            # JUST MADE this interval. Read deliberately AFTER the turn(s),
            # not before: DRL's own scoring/history arrays (below) now
            # reflect the outcome of what was just decided this interval,
            # not what was carried over from last interval (the old
            # next_state convention).
            live_tilt_deg = tilt_idx_to_deg(self.drl_controller.tilt_idx, self.drl_action_grid_deg)
            if self.async_schedule == "sync":
                # One turn, whole network at once.
                closes = np.ones(self.num_bs, dtype=bool)
                self.drl_process_turn(state, sector_idx, cell_idx, closes, live_tilt_deg,
                                      training=True, power_table=power_table)
            else:
                # Gauss-Seidel: sequential per-sector turns, tilt propagates within the interval.
                for s in range(self.num_bs):
                    closes = np.zeros(self.num_bs, dtype=bool)
                    closes[s] = True
                    self.drl_process_turn(state, sector_idx, cell_idx, closes, live_tilt_deg,
                                          training=True, power_table=power_table)
            drl_tilt_deg_this_interval = live_tilt_deg

            # Score every algorithm once, DRL at drl_tilt_deg_this_interval (see execute_algorithms).
            measurement = {
                "adaptive_power_w": self.kpi_manager.resolve_power_at_tilt(
                    state, self.adaptive_legacy_controller.tilt_deg)[0],
                "drl_power_w": self.kpi_manager.resolve_power_at_tilt(state, drl_tilt_deg_this_interval)[0],
                "ut_loc": ut_loc_this_interval,
            }

            # Execute and evaluate algorithms
            self.execute_algorithms(measurement, interval, drl_tilt_deg_this_interval)

            # Live reward/loss plot
            if live_plot is not None:
                live_plot.update(interval, self.drl_reward_history[interval], self.drl_loss_per_interval[interval])

            # log results
            if interval % 10 == 0 or is_last_interval:
                fields = []
                fields.append(f"adaptive_legacy={self.coverage_adaptive_legacy[interval]:.3f}")
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
            "coverage_adaptive_legacy": self.coverage_adaptive_legacy,
            "coverage_drl": self.coverage_drl,
            "sinr_percentiles_adaptive_legacy": self.sinr_percentiles_adaptive_legacy,
            "sinr_percentiles_drl": self.sinr_percentiles_drl,
            "adaptive_legacy_tilt_deg_history": self.adaptive_legacy_tilt_deg_history,
            "drl_tilt_deg_history": self.drl_tilt_deg_history,
            "drl_reward_history": self.drl_reward_history,
            "drl_overshoot_history": self.drl_overshoot_history,
            "drl_loss_per_interval": self.drl_loss_per_interval,
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
