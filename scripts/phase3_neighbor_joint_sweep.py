"""Runs phase 3 (raw UE behavior) with its opt-in neighbor-joint-sweep
sub-experiment enabled (2026-10-12, explicit request) -- see
scripts/debug_pipeline_audit.py's own NEIGHBOR_JOINT_CASES/
compute_neighbor_joint_tilt_sweep/_neighbor_joint_sweep_figures for the
full design: sector 0 + its 2 real neighbors (N1, N2) each pinned to
near-heavy ("L") or far-heavy ("H") demand across 6 cases, every OTHER
sector fixed at mid-heavy; for each case, ONE population snapshot swept
exhaustively across all 5x5x5=125 (sector 0, N1, N2) own-tilt combos
(every other sector fixed at 9 deg) -- global (whole-network) coverage
fraction at every combo, plotted as a 1x5 row of heatmaps per case (one
per N2 tilt value), x=sector 0's own tilt, y=N1's own tilt.

Uses the COMMITTED config.yaml as-is otherwise (UMa, m=16 hanning, LOS no
shadowing, hotspot_demand-periodic mobility).

Run: python scripts/phase3_neighbor_joint_sweep.py
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
dpa.phase3_raw_ue_behavior(engine, scenario, out_dir, run_neighbor_joint_sweep=True)
print(f"saved: {out_dir}")
