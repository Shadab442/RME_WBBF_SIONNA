"""Runs phase 2 (radio environment validation) for antenna.m in {4, 8, 10, 16, 32}
x antenna_window in {rectangular, hanning} (10 combos), ALL under the
los_noshadow channel condition (force_los=True, disable_shadow_fading=True
-- 2026-10-10, explicit request), using the COMMITTED config.yaml as-is
(UMa, downtilt 5/13/2 deg, etc.) otherwise. Saves each into its own named
subfolder under phase 2's own OUT_ROOT, same convention as
phase12_channel_conditions.py.

Phase 1 (static geometry) is deliberately NOT rerun here: antenna config
(m/antenna_window) has no effect on phase1_static_geometry's own output
(site/sector/grid-point geometry only -- see PHASE1_CONFIG_KEYS in
debug_pipeline_audit.py), so results/debug/phase1_static_geometry/default/
already covers every combo here.

Run: python scripts/phase12_antenna_conditions.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import torch
import debug_pipeline_audit as dpa
from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Phase 1 already done (antenna config doesn't affect static geometry) -- saved in "
     f"{os.path.join(dpa.OUT_ROOT, 'phase1_static_geometry', 'default')}/")

# name -> (m, antenna_window)
CONDITIONS = {
    "los_noshadow_m4_rectangular": (4, "rectangular"),
    "los_noshadow_m4_hanning": (4, "hanning"),
    "los_noshadow_m8_rectangular": (8, "rectangular"),
    "los_noshadow_m8_hanning": (8, "hanning"),
    "los_noshadow_m10_rectangular": (10, "rectangular"),
    "los_noshadow_m10_hanning": (10, "hanning"),
    "los_noshadow_m16_rectangular": (16, "rectangular"),
    "los_noshadow_m16_hanning": (16, "hanning"),
    "los_noshadow_m32_rectangular": (32, "rectangular"),
    "los_noshadow_m32_hanning": (32, "hanning"),
}

for name, (m, antenna_window) in CONDITIONS.items():
    print(f"\n=== {name}: m={m} antenna_window={antenna_window} (los_noshadow) ===")

    def _cond_load_config(m=m, antenna_window=antenna_window):
        cfg = load_config()
        cfg["antenna"]["m"] = m
        cfg["antenna"]["antenna_window"] = antenna_window
        cfg["channel"]["force_los"] = True
        cfg["channel"]["disable_shadow_fading"] = True
        return cfg
    dpa.load_config = _cond_load_config

    sionna.phy.config.seed = dpa.DEBUG_SEED
    engine = SimulationEngine(_cond_load_config())

    phase2_out_dir = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", name)
    dpa.phase2_radio_environment_validation(engine, None, phase2_out_dir)
    print(f"saved: {phase2_out_dir}")
