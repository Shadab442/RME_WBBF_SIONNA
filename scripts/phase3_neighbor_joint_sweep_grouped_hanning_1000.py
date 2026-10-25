"""Reruns the GROUPED neighbor-joint-sweep sub-experiment (see
scripts/phase3_neighbor_joint_sweep_grouped_rect.py's own module docstring)
with config.yaml set to m=16/hanning and demand_states near/far sharpened
to the purest possible split, 10:0:0 / 0:0:10 -- 2026-10-12, explicit
request. Note m=16/hanning has the known structural bug (far UEs
permanently uncovered at every tilt, see config.yaml's own comment) --
this run deliberately tests the pure-split case under that same bug.

Uses the COMMITTED config.yaml as-is (m=16 hanning, demand_states
10:0:0/0:0:10, LOS no shadowing, hotspot_demand-periodic mobility,
phase-group-synced demand+tilt).

Saves to phase3_raw_ue_behavior/grouped_hanning_1000/.

Run: python scripts/phase3_neighbor_joint_sweep_grouped_hanning_1000.py
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

out_dir = os.path.join(dpa.OUT_ROOT, "phase3_raw_ue_behavior", "grouped_hanning_1000")
dpa.phase3_raw_ue_behavior(engine, scenario, out_dir, run_neighbor_joint_sweep_grouped=True)
print(f"saved: {out_dir}")
