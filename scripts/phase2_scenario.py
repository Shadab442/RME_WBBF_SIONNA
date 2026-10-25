"""Runs debug_pipeline_audit.py's phase2_radio_environment_validation against
a non-default topology.scenario (UMa/UMi), with every absolute-meter
geometry knob tuned for RMa's isd=5000m (hotspot near/far cuts,
spatial_grid_cell_size_m, mobility deviation_radius_m/cluster_radius_max_m)
rescaled by the scenario's own isd ratio -- same scaling established in
gained_lost_fine_grid_scenario.py's own exploration (2026-10-05), which
found UMa gives a much cleaner near-vs-(mid+far) two-sided tilt tradeoff
than the committed RMa scenario. config.yaml itself is NOT modified --
this builds its own engine from an in-memory cfg override, per
main()'s own phase-2-variants-live-in-named-subfolders convention.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import debug_pipeline_audit as dpa
import helpers.simulation_engine as se_mod
from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine
from sionna.phy.channel.utils import set_3gpp_scenario_parameters

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "uma"
TILT_START = float(sys.argv[2]) if len(sys.argv) > 2 else -2.0
TILT_END = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
TILT_STEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
# "stochastic" (default) = config.yaml's own force_los=null/disable_shadow_fading=false.
# "los_with_shadowing" = force_los=True, disable_shadow_fading=False -- forces LOS
# (removing the stochastic LOS/NLOS draw, found 2026-10-07 to be the dominant noise
# source via radial_two_neighbor_factorial_scenario.py's deterministic/los_with_
# shadowing/stochastic breakdown) while keeping shadow fading's realistic per-link
# variability (found to be a much smaller contributor to signal compression).
CHANNEL_MODE = sys.argv[5] if len(sys.argv) > 5 else "stochastic"
assert CHANNEL_MODE in ("stochastic", "los_with_shadowing")
RMA_ISD_M = 5000.0
RMA_NEAR_DIST_M = 900.0
RMA_FAR_DIST_M = 2300.0
# Empirically-calibrated near/mid crossover (radial_swap_plot_scenario.py,
# seed-averaged distance-binned coverage swap, 2026-10-05) -- see
# gained_lost_fine_grid_scenario.py's own comment for why the naive
# isd-ratio rescale (90m for uma) is wrong: the crossover is governed by
# antenna geometry (mechanical pitch), which doesn't scale with isd the way
# cell_radius does. far_dist_m keeps the isd-ratio rescale -- it reproduces
# RMa's own near-to-cell_radius fraction, and there's no second empirical
# crossover in the data (same as RMa's own calibration).
_CALIBRATED_NEAR_M = {"uma": 174.2}

_, isd, _, _, _, _, _, _ = set_3gpp_scenario_parameters(SCENARIO)
isd_m = float(isd.item())
scale = isd_m / RMA_ISD_M
near_dist_m = _CALIBRATED_NEAR_M.get(SCENARIO, RMA_NEAR_DIST_M * scale)
far_dist_m = RMA_FAR_DIST_M * scale
spatial_grid_cell_size_m = 50.0 * scale
deviation_radius_m = 40.0 * scale
cluster_radius_max_m = 100.0 * scale
print(f"scenario={SCENARIO} isd={isd_m:.0f}m scale={scale:.4f} "
     f"near_dist={near_dist_m:.1f}m ({'empirical' if SCENARIO in _CALIBRATED_NEAR_M else 'isd-scaled guess'}) "
     f"far_dist={far_dist_m:.1f}m (isd-scaled) "
     f"spatial_grid_cell_size_m={spatial_grid_cell_size_m:.2f}m "
     f"deviation_radius_m={deviation_radius_m:.2f}m cluster_radius_max_m={cluster_radius_max_m:.2f}m "
     f"tilt_range=[{TILT_START},{TILT_END}] step={TILT_STEP}")

se_mod._HOTSPOT_NEAR_DIST_M = near_dist_m
se_mod._HOTSPOT_FAR_DIST_M = far_dist_m

def _scaled_load_config():
    cfg = load_config()
    cfg["topology"]["scenario"] = SCENARIO
    cfg["topology"]["spatial_grid_cell_size_m"] = spatial_grid_cell_size_m
    cfg["mobility"]["hotspot"]["deviation_radius_m"] = deviation_radius_m
    cfg["mobility"]["rpgm"]["cluster_radius_max_m"] = cluster_radius_max_m
    cfg["downtilt"]["downtilt_sweep_start_deg"] = TILT_START
    cfg["downtilt"]["downtilt_sweep_end_deg"] = TILT_END
    cfg["downtilt"]["tilt_step_deg"] = TILT_STEP
    if CHANNEL_MODE == "los_with_shadowing":
        cfg["channel"]["force_los"] = True
        cfg["channel"]["disable_shadow_fading"] = False
    return cfg

# compute_hotspot_ground_truth (called deep inside phase2) builds its OWN
# engine via build_debug_engine() -> module-global load_config(), ignoring
# the engine built below entirely -- patch dpa's own load_config binding so
# that internal rebuild also sees the scaled UMa/UMi cfg, not committed RMa.
dpa.load_config = _scaled_load_config

sionna.phy.config.seed = dpa.DEBUG_SEED
engine = SimulationEngine(_scaled_load_config())

out_tag = SCENARIO if CHANNEL_MODE == "stochastic" else f"{SCENARIO}_{CHANNEL_MODE}"
out_tag += f"_tilt{TILT_START:g}to{TILT_END:g}step{TILT_STEP:g}"
phase1_out_dir = os.path.join(dpa.OUT_ROOT, "phase1_static_geometry", out_tag)
dpa.phase1_static_geometry(engine, phase1_out_dir)

phase2_out_dir = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", out_tag)
dpa.phase2_radio_environment_validation(engine, None, phase2_out_dir)
