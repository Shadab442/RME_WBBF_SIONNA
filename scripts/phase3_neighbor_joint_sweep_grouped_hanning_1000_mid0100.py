"""Reruns the GROUPED neighbor-joint-sweep sub-experiment with config.yaml's
mid-heavy demand_states row changed from [2,6,2] to pure [0,10,0] -- 2026-
10-12, explicit request. Near/far-heavy stay at the pure 10:0:0/0:0:10
split from the prior run (m=16/hanning).

Note: mid-heavy is unused by NEIGHBOR_JOINT_CASES (only "L"/"H" --
near/far-heavy -- ever get forced), so the 6 case heatmaps themselves are
unaffected by this change; only the OTHER phase3 figures in this same run
(region_occupancy_validation, real_ue_tilt_sensitivity's state-0 panel)
reflect the new pure mid-heavy population.

Saves to phase3_raw_ue_behavior/grouped_hanning_1000_mid0100/ -- a sibling
of grouped_hanning_1000/'s own output.

Run: python scripts/phase3_neighbor_joint_sweep_grouped_hanning_1000_mid0100.py
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

out_dir = os.path.join(dpa.OUT_ROOT, "phase3_raw_ue_behavior", "grouped_hanning_1000_mid0100")
dpa.phase3_raw_ue_behavior(engine, scenario, out_dir, run_neighbor_joint_sweep_grouped=True)
print(f"saved: {out_dir}")
