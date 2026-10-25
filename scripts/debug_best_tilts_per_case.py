"""One-off (2026-10-12): for the just-completed grouped neighbor-joint-sweep
(9:1:0/0:1:9 demand split, m=16/rectangular), report the (own, N1, N2) tilt
combo that maximizes GLOBAL coverage for each of the 6 cases -- i.e. the
best tilt for sector 0 (own group), sector 8 (N1 group), sector 10 (N2
group) in every demand-state case.

Uses the COMMITTED config.yaml as-is (same config the grouped_rect_910 run
used). Re-runs only the 125-combo sweep itself (fast, single snapshot per
case), not the full phase3 pipeline.

Run: python scripts/debug_best_tilts_per_case.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import torch
import debug_pipeline_audit as dpa

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
sionna.phy.config.seed = dpa.DEBUG_SEED

engine = dpa.build_debug_engine()
scenario = dpa.load_or_run_scenario(engine, use_cache=False)

target_sector = 0
nbr = engine.topology.neighbor_ids[target_sector]
nbr = nbr[nbr >= 0]
n1, n2 = int(nbr[0]), int(nbr[1])

base_state_idx = engine.mobility._state_idx.copy()
own_group = np.flatnonzero(base_state_idx == base_state_idx[target_sector])
n1_group = np.flatnonzero(base_state_idx == base_state_idx[n1])
n2_group = np.flatnonzero(base_state_idx == base_state_idx[n2])
print(f"target_sector={target_sector} n1={n1} n2={n2} group_sizes=own:{len(own_group)}/n1:{len(n1_group)}/n2:{len(n2_group)}")

for case_name, (own_s, n1_s, n2_s) in dpa.NEIGHBOR_JOINT_CASES.items():
    res = dpa.compute_neighbor_joint_tilt_sweep_grouped(
        engine, target_sector, n1, n2, own_s, n1_s, n2_s, own_group, n1_group, n2_group)
    tilt_values = res["tilt_values"]
    gc = res["global_coverage"]
    oi, ni, nj = np.unravel_index(np.argmax(gc), gc.shape)
    best_cov = gc[oi, ni, nj]
    print(f"  {case_name} (own={own_s}, N1={n1_s}, N2={n2_s}): "
         f"best tilt -- sector0(own)={tilt_values[oi]:.0f}deg sector{n1}(N1)={tilt_values[ni]:.0f}deg "
         f"sector{n2}(N2)={tilt_values[nj]:.0f}deg -> global coverage={best_cov:.4f}")
