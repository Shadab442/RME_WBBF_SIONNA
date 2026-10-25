"""Reruns phase 3's GROUPED neighbor-joint-sweep sub-experiment (see
scripts/phase3_neighbor_joint_sweep_grouped.py's own module docstring for
the sync/sharpening design) now that config.yaml's committed antenna
default has been changed to m=16/rectangular (2026-10-12) -- the earlier
grouped run was under m=16/hanning, which leaves far UEs structurally
uncovered at every tilt and confounds the own-group-tilt-direction finding.

Uses the COMMITTED config.yaml as-is (m=16 rectangular, demand_states
sharpened to 8:1:1, LOS no shadowing, hotspot_demand-periodic mobility,
phase-group-synced demand+tilt).

Saves to phase3_raw_ue_behavior/grouped_rect/ -- a sibling of the
m16/hanning grouped run's own output (neighbor_joint_sweep_grouped/), so
both stay available side by side for direct comparison.

Run: python scripts/phase3_neighbor_joint_sweep_grouped_rect.py
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

out_dir = os.path.join(dpa.OUT_ROOT, "phase3_raw_ue_behavior", "grouped_rect")
dpa.phase3_raw_ue_behavior(engine, scenario, out_dir, run_neighbor_joint_sweep_grouped=True)
print(f"saved: {out_dir}")
