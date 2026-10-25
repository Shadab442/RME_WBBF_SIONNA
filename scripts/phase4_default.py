"""Reruns phase 4 (measurement aggregation) with the committed config.yaml
as-is -- 2026-10-13, picks up the new tilt-decision-accuracy figure, the
whole-area column on every spatial-state figure, and the coverage_by_region
rename, none of which existed in whatever phase4_measurement_aggregation/
output is currently on disk.

Saves to phase4_measurement_aggregation/ (default, outdoor_ue_ratio_override
inherited from config.yaml, currently 1.0).

Run: python scripts/phase4_default.py
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
out_dir = os.path.join(dpa.OUT_ROOT, "phase4_measurement_aggregation", "default")
dpa.phase4_measurement_aggregation(engine, None, out_dir)
print(f"saved: {out_dir}")
