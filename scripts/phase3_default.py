"""Reruns phase 3 (raw UE behavior) with the committed config.yaml as-is --
2026-10-13, writes to the new default/ subfolder convention (matching
phase4_default.py and phase 1/2's existing convention) now that phase 3's
own output directory also accumulates many named experiment variants
(grouped_rect_910/, outdoor_ratio_null/, m16_rectangular/, ...).

Saves to phase3_raw_ue_behavior/default/ -- no opt-in neighbor-joint-sweep
sub-experiment (that's what the named variant scripts are for).

Run: python scripts/phase3_default.py
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

out_dir = os.path.join(dpa.OUT_ROOT, "phase3_raw_ue_behavior", "default")
dpa.phase3_raw_ue_behavior(engine, scenario, out_dir)
print(f"saved: {out_dir}")
