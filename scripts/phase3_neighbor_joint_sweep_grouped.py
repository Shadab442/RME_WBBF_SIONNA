"""Runs phase 3's opt-in GROUPED neighbor-joint-sweep sub-experiment
(2026-10-12, explicit request) -- combines two changes in one run:
  1. Demand state + tilt synced across each sector's own full phase group
     (not just sector 0/N1/N2 individually) -- see
     compute_neighbor_joint_tilt_sweep_grouped's own docstring. Eliminates
     the global-coverage dilution the ungrouped sweep had (target_sector/
     n1/n2 were only ~15% of the network; now every sector is covered by
     one of the 3 swept phase groups).
  2. demand_states' own near-heavy/far-heavy rows sharpened to 8:1:1 (was
     6:2:2) in config.yaml -- isolates each named region's own tilt
     behavior more cleanly from its own minority near/mid/far blend.

Uses the COMMITTED config.yaml as-is otherwise (UMa, m=16 hanning, LOS no
shadowing, hotspot_demand-periodic mobility).

Run: python scripts/phase3_neighbor_joint_sweep_grouped.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import torch
import debug_pipeline_audit as dpa

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
sionna.phy.config.seed = dpa.DEBUG_SEED

engine = dpa.build_debug_engine()
scenario = dpa.load_or_run_scenario(engine, use_cache=False)

out_dir = os.path.join(dpa.OUT_ROOT, "phase3_raw_ue_behavior")
dpa.phase3_raw_ue_behavior(engine, scenario, out_dir, run_neighbor_joint_sweep_grouped=True)
print(f"saved: {out_dir}")
