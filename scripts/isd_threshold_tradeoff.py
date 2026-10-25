"""Controlled, single-variable experiment, 2026-09-29: does shrinking ISD
(cell radius) compensate for a stricter coverage_threshold_db, holding
everything else (RMa's own channel model, bs_height, min_bs_ut_dist,
antenna, TX power) fixed? This is a deliberate ceteris paribus sweep to
isolate the ISD-vs-threshold tradeoff -- NOT a claim about a real 3GPP
deployment configuration (unlike the earlier ISD-without-matching-height
mistake, height/min_bs_ut_dist/channel model here all stay correctly
RMa's own values; only ISD itself is swept).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import debug_pipeline_audit as dpa
from sionna.phy.channel.utils import set_3gpp_scenario_parameters as _orig_scenario_params

ISD_VALUES_M = [1000.0, 2000.0, 3000.0, 5000.0]
THRESHOLDS_DB = [-3.0, 0.0, 3.0, 6.0]
TARGET_SECTOR = 0

_orig_load_config = dpa.load_config


def make_loader(threshold):
    def _patched_load_config():
        cfg = _orig_load_config()
        cfg["topology"]["scenario"] = "rma"
        cfg["antenna"]["m"] = 48
        cfg["channel"]["bs_tx_power_dbm"] = 40.0
        cfg["channel"]["force_los"] = None
        cfg["channel"]["disable_shadow_fading"] = False
        cfg["kpi"]["coverage_threshold_db"] = threshold
        cfg["downtilt"]["downtilt_sweep_start_deg"] = 0
        cfg["downtilt"]["downtilt_sweep_end_deg"] = 0
        cfg["downtilt"]["tilt_step_deg"] = 1
        # Mobility realism is irrelevant here (radio/grid layer only) --
        # just needs to not crash across a wide ISD range.
        cfg["mobility"]["rpgm"]["cluster_radius_max_m"] = 2000.0
        return cfg
    return _patched_load_config


def make_scenario_params_override(isd_m):
    def _patched(scenario):
        min_bs_ut_dist, isd, bs_height, min_ut_height, a, b, c, d = _orig_scenario_params(scenario)
        isd_override = isd.clone() if hasattr(isd, "clone") else isd
        import torch
        isd_override = torch.as_tensor(isd_m, dtype=isd.dtype, device=isd.device)
        return min_bs_ut_dist, isd_override, bs_height, min_ut_height, a, b, c, d
    return _patched


import helpers.simulation_engine as se_module

print(f"{'isd_m':>8}{'threshold':>11}{'coverage':>10}")
for isd_m in ISD_VALUES_M:
    for threshold in THRESHOLDS_DB:
        dpa.load_config = make_loader(threshold)
        se_module.set_3gpp_scenario_parameters = make_scenario_params_override(isd_m)
        sionna.phy.config.seed = dpa.DEBUG_SEED
        # cluster_mobility_mode="rpgm-random" sidesteps HotspotDemandMobility
        # entirely -- the fixed 900m/2300m near/mid/far distance cuts don't
        # scale with ISD, so a small ISD (e.g. 1000m) can have no valid
        # "far" region at all, crashing hotspot_region_points. This
        # experiment only needs the raw grid/radio layer, not demand-state
        # mobility.
        engine = dpa.build_debug_engine(cluster_mobility_mode="rpgm-random")
        topo = engine.topology
        xy = engine.grid_points[TARGET_SECTOR]
        tilt_values = np.array([0.0])
        pert = dpa.compute_single_sector_tilt_perturbation(TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
        num_grid_cells = engine.num_grid_cells
        sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
        sinr = pert["sinr_db"][0][sl]
        cov = (sinr > engine.coverage_threshold_db).mean()
        print(f"{isd_m:8.0f}{threshold:11.1f}{cov:10.3f}", flush=True)

se_module.set_3gpp_scenario_parameters = _orig_scenario_params
print("done")
