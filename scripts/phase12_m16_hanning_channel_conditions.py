"""Runs phase 2 (radio environment validation) for all 4 LOS/shadow-fading
combinations (los_noshadow, los_shadow, stochastic_noshadow,
stochastic_shadow -- 2026-10-11), ALL with antenna.m=16,
antenna_window=hanning locked, using the COMMITTED config.yaml as-is
(UMa, downtilt 5/13/2 deg, etc.) otherwise. Saves each into its own named
subfolder under phase 2's own OUT_ROOT, same convention as
phase12_channel_conditions.py/phase12_antenna_conditions.py.

los_noshadow (m16_hanning_los_noshadow) was already produced by
phase12_antenna_conditions.py's own m16/hanning combo (identical config --
m=16, hanning, force_los=True, disable_shadow_fading=True) and copied here
rather than rerun; this script only computes the other 3.

Phase 1 (static geometry) is deliberately NOT rerun here -- see every
other phase12_*.py driver's own docstring for why (antenna/channel config
has no effect on phase1_static_geometry's own output).

Run: python scripts/phase12_m16_hanning_channel_conditions.py
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

print(f"Phase 1 already done (antenna/channel config doesn't affect static geometry) -- saved in "
     f"{os.path.join(dpa.OUT_ROOT, 'phase1_static_geometry', 'default')}/")
print(f"m16_hanning_los_noshadow already done (copied from phase12_antenna_conditions.py's own "
     f"los_noshadow_m16_hanning, identical config) -- saved in "
     f"{os.path.join(dpa.OUT_ROOT, 'phase2_radio_environment_validation', 'm16_hanning_los_noshadow')}/")

# name -> (force_los, disable_shadow_fading)
CONDITIONS = {
    "m16_hanning_los_shadow": (True, False),
    "m16_hanning_stochastic_noshadow": (None, True),
    "m16_hanning_stochastic_shadow": (None, False),
}

for name, (force_los, disable_shadow_fading) in CONDITIONS.items():
    print(f"\n=== {name}: force_los={force_los} disable_shadow_fading={disable_shadow_fading} "
         f"(m=16, hanning) ===")

    def _cond_load_config(force_los=force_los, disable_shadow_fading=disable_shadow_fading):
        cfg = load_config()
        cfg["antenna"]["m"] = 16
        cfg["antenna"]["antenna_window"] = "hanning"
        cfg["channel"]["force_los"] = force_los
        cfg["channel"]["disable_shadow_fading"] = disable_shadow_fading
        return cfg
    dpa.load_config = _cond_load_config

    sionna.phy.config.seed = dpa.DEBUG_SEED
    engine = SimulationEngine(_cond_load_config())

    phase2_out_dir = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", name)
    dpa.phase2_radio_environment_validation(engine, None, phase2_out_dir)
    print(f"saved: {phase2_out_dir}")
