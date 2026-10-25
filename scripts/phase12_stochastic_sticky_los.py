"""Reruns every phase 2 (radio environment validation) experiment that used
the stochastic LOS model (channel.force_los: null) with
channel.los_correlation_distance_m=40.0 (2026-10-11, explicit request) --
see helpers/sticky_los.py's own module docstring for what this does (holds
each UE's LOS/NLOS state fixed until it has moved more than 40m since its
own last redraw, approximating 3GPP TR 38.901 Sec 7.6.3's spatial-
consistency intent instead of Sionna's own independent-redraw-every-
interval default). force_los=True experiments (los_noshadow, los_shadow,
and every los_noshadow_m*/m16_hanning_los_* antenna combo) are NOT rerun --
sticky LOS is a no-op there (no stochastic draw to smooth).

Saves each into a new "_sticky_los" sibling directory under phase 2's own
OUT_ROOT, next to (not replacing) the original non-sticky result, so both
stay available for direct comparison:
  default                         -> default_sticky_los
  stochastic_shadow                -> stochastic_shadow_sticky_los
  m16_hanning_stochastic_noshadow  -> m16_hanning_stochastic_noshadow_sticky_los
  m16_hanning_stochastic_shadow    -> m16_hanning_stochastic_shadow_sticky_los

Phase 1 (static geometry) is deliberately NOT rerun here -- see every other
phase12_*.py driver's own docstring for why (LOS correlation has no effect
on phase1_static_geometry's own output).

Run: python scripts/phase12_stochastic_sticky_los.py
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

LOS_CORRELATION_DISTANCE_M = 40.0

print(f"Phase 1 already done (LOS correlation has no effect on static geometry) -- saved in "
     f"{os.path.join(dpa.OUT_ROOT, 'phase1_static_geometry', 'default')}/")

# name -> (m, antenna_window, disable_shadow_fading) -- force_los is always
# None (stochastic) for every entry here, that's the whole point of this sweep.
CONDITIONS = {
    "default_sticky_los": (32, "rectangular", False),
    "stochastic_shadow_sticky_los": (32, "rectangular", False),
    "m16_hanning_stochastic_noshadow_sticky_los": (16, "hanning", True),
    "m16_hanning_stochastic_shadow_sticky_los": (16, "hanning", False),
}

for name, (m, antenna_window, disable_shadow_fading) in CONDITIONS.items():
    print(f"\n=== {name}: m={m} antenna_window={antenna_window} "
         f"disable_shadow_fading={disable_shadow_fading} "
         f"los_correlation_distance_m={LOS_CORRELATION_DISTANCE_M} ===")

    def _cond_load_config(m=m, antenna_window=antenna_window, disable_shadow_fading=disable_shadow_fading):
        cfg = load_config()
        cfg["antenna"]["m"] = m
        cfg["antenna"]["antenna_window"] = antenna_window
        cfg["channel"]["force_los"] = None
        cfg["channel"]["disable_shadow_fading"] = disable_shadow_fading
        cfg["channel"]["los_correlation_distance_m"] = LOS_CORRELATION_DISTANCE_M
        return cfg
    dpa.load_config = _cond_load_config

    sionna.phy.config.seed = dpa.DEBUG_SEED
    engine = SimulationEngine(_cond_load_config())
    assert engine.sticky_los is not None, "sticky LOS should be enabled for this sweep"

    phase2_out_dir = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", name)
    dpa.phase2_radio_environment_validation(engine, None, phase2_out_dir)
    print(f"saved: {phase2_out_dir}")
