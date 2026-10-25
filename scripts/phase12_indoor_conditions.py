"""Runs phase 1 + phase 2 for all_indoor in {False, True}, using the
COMMITTED config.yaml as-is (UMa, stochastic LOS + shadowing, m=32
rectangular, downtilt 5-13/2 deg) -- phase 1/2's own virtual-grid
diagnostic path (debug_pipeline_audit._virtual_grid_state) previously
IGNORED SimulationEngine's own derived anchor_ue_ratio entirely, always treating
every virtual UE as outdoor; all_indoor=True (2026-10-08 addition) forces
every virtual UE indoor instead -- which, per Sionna's own set_topology
(los is ANDed with ~in_state), ALSO forces NLOS regardless of force_los,
plus whatever O2I penetration-loss model the channel model applies for
UMa (o2i_model="low", see helpers/simulation_engine.py).

"outdoor" here is the normal diagnostic baseline (matches every other
phase1/2 run so far in this project), "indoor" is the new all-indoor
condition, for direct comparison.

Run: python scripts/phase12_indoor_conditions.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import debug_pipeline_audit as dpa
from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine

# name -> all_indoor
CONDITIONS = {
    "outdoor": False,
    "indoor": True,
}

for name, all_indoor in CONDITIONS.items():
    print(f"\n=== {name}: all_indoor={all_indoor} ===")

    sionna.phy.config.seed = dpa.DEBUG_SEED
    engine = SimulationEngine(load_config())

    phase1_out_dir = os.path.join(dpa.OUT_ROOT, "phase1_static_geometry", name)
    dpa.phase1_static_geometry(engine, phase1_out_dir)

    phase2_out_dir = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", name)
    dpa.phase2_radio_environment_validation(engine, None, phase2_out_dir, all_indoor=all_indoor)
    print(f"saved: {phase1_out_dir}  and  {phase2_out_dir}")
