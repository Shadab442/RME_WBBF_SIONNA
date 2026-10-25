"""Reruns the GROUPED neighbor-joint-sweep sub-experiment (see
scripts/phase3_neighbor_joint_sweep_grouped_rect.py's own module docstring)
with demand_states' near-heavy/far-heavy rows sharpened further to
9:1:0 / 0:1:9 (was 8:1:1) -- 2026-10-12, explicit request -- the purest
possible near-only/far-only population split still consistent with
mobility.common.num_clusters_per_sector=10.

Uses the COMMITTED config.yaml as-is otherwise (m=16 rectangular,
LOS no shadowing, hotspot_demand-periodic mobility, phase-group-synced
demand+tilt).

Saves to phase3_raw_ue_behavior/grouped_rect_910/ -- a sibling of the
8:1:1 grouped_rect/ run's own output, so both stay available side by side
for direct comparison.

Run: python scripts/phase3_neighbor_joint_sweep_grouped_rect_910.py
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

out_dir = os.path.join(dpa.OUT_ROOT, "phase3_raw_ue_behavior", "grouped_rect_910")
dpa.phase3_raw_ue_behavior(engine, scenario, out_dir, run_neighbor_joint_sweep_grouped=True)
print(f"saved: {out_dir}")
