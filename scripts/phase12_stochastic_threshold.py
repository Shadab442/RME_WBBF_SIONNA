"""Reruns the same 4 stochastic-LOS (force_los: null) phase 2 experiments as
phase12_stochastic_sticky_los.py, but with kpi.coverage_threshold_db=-3.0
instead of config.yaml's own committed 0.0 (2026-10-11, explicit request),
and WITHOUT sticky LOS (channel.los_correlation_distance_m stays null --
default Sionna behavior, independent LOS redraw every interval).

Saves each into a new "_thrminus3" sibling directory under phase 2's own
OUT_ROOT, next to (not replacing) the 0 dB-threshold original:
  default                         -> default_thrminus3
  stochastic_shadow                -> stochastic_shadow_thrminus3
  m16_hanning_stochastic_noshadow  -> m16_hanning_stochastic_noshadow_thrminus3
  m16_hanning_stochastic_shadow    -> m16_hanning_stochastic_shadow_thrminus3

Phase 1 (static geometry) is deliberately NOT rerun here -- coverage
threshold has no effect on phase1_static_geometry's own output.

Run: python scripts/phase12_stochastic_threshold.py
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

COVERAGE_THRESHOLD_DB = -3.0

print(f"Phase 1 already done (coverage threshold has no effect on static geometry) -- saved in "
     f"{os.path.join(dpa.OUT_ROOT, 'phase1_static_geometry', 'default')}/")

# name -> (m, antenna_window, disable_shadow_fading) -- force_los is always
# None (stochastic), same 4 combos as phase12_stochastic_sticky_los.py.
CONDITIONS = {
    "default_thrminus3": (32, "rectangular", False),
    "stochastic_shadow_thrminus3": (32, "rectangular", False),
    "m16_hanning_stochastic_noshadow_thrminus3": (16, "hanning", True),
    "m16_hanning_stochastic_shadow_thrminus3": (16, "hanning", False),
}

for name, (m, antenna_window, disable_shadow_fading) in CONDITIONS.items():
    print(f"\n=== {name}: m={m} antenna_window={antenna_window} "
         f"disable_shadow_fading={disable_shadow_fading} "
         f"coverage_threshold_db={COVERAGE_THRESHOLD_DB} ===")

    def _cond_load_config(m=m, antenna_window=antenna_window, disable_shadow_fading=disable_shadow_fading):
        cfg = load_config()
        cfg["antenna"]["m"] = m
        cfg["antenna"]["antenna_window"] = antenna_window
        cfg["channel"]["force_los"] = None
        cfg["channel"]["disable_shadow_fading"] = disable_shadow_fading
        cfg["kpi"]["coverage_threshold_db"] = COVERAGE_THRESHOLD_DB
        return cfg
    dpa.load_config = _cond_load_config

    sionna.phy.config.seed = dpa.DEBUG_SEED
    engine = SimulationEngine(_cond_load_config())
    assert engine.sticky_los is None, "sticky LOS should be OFF for this sweep"

    phase2_out_dir = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", name)
    dpa.phase2_radio_environment_validation(engine, None, phase2_out_dir)
    print(f"saved: {phase2_out_dir}")
