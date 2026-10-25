"""Runs phase 2 (radio environment validation) for 3 LOS/shadow-fading
conditions -- LOS, LOS+shadowing, stochastic LOS+shadowing (2026-10-10,
dropped the 4th "stochastic_noshadow" combo, not currently requested) --
using the COMMITTED config.yaml as-is (UMa, m=32, downtilt 5/13/2 deg,
etc.) except for channel.force_los/disable_shadow_fading, which this
script overrides per condition. Saves each into its own named subfolder
under phase 2's own OUT_ROOT, parallel to the existing "default"
(stochastic LOS + shadowing, i.e. config.yaml's own literal committed
values) subfolder convention.

Phase 1 (static geometry) is deliberately NOT rerun here: it depends only
on topology.scenario/num_rings/spatial_grid_cell_size_m and
antenna.azimuth_offset_deg (see PHASE1_CONFIG_KEYS in
debug_pipeline_audit.py) -- none of which this script touches -- so its
existing results/debug/phase1_static_geometry/default/ output already
covers every condition here; regenerating 3 identical copies would be
pure waste.

Run: python scripts/phase12_channel_conditions.py
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

print(f"Phase 1 already done (channel conditions don't affect static geometry) -- saved in "
     f"{os.path.join(dpa.OUT_ROOT, 'phase1_static_geometry', 'default')}/")

# name -> (force_los, disable_shadow_fading)
CONDITIONS = {
    "los_noshadow": (True, True),
    "los_shadow": (True, False),
    "stochastic_shadow": (None, False),
}

for name, (force_los, disable_shadow_fading) in CONDITIONS.items():
    print(f"\n=== {name}: force_los={force_los} disable_shadow_fading={disable_shadow_fading} ===")

    def _cond_load_config(force_los=force_los, disable_shadow_fading=disable_shadow_fading):
        cfg = load_config()
        cfg["channel"]["force_los"] = force_los
        cfg["channel"]["disable_shadow_fading"] = disable_shadow_fading
        return cfg
    dpa.load_config = _cond_load_config

    sionna.phy.config.seed = dpa.DEBUG_SEED
    engine = SimulationEngine(_cond_load_config())

    phase2_out_dir = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", name)
    dpa.phase2_radio_environment_validation(engine, None, phase2_out_dir)
    print(f"saved: {phase2_out_dir}")
