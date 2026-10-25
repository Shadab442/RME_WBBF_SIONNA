"""One-off (2026-10-12): same best-tilt-per-case report as
debug_best_tilts_per_case.py, but parameterized over antenna_window and
demand_states so it can reproduce the table for EACH of today's 3 grouped
neighbor-joint-sweep runs, not just the final (committed-config) one:
  1. m=16/hanning, demand_states near/far = 8:1:1 / 1:1:8 (first grouped run)
  2. m=16/rectangular, demand_states 8:1:1 / 1:1:8 (after the antenna fix)
  3. m=16/rectangular, demand_states 9:1:0 / 0:1:9 (current committed config
     -- already reported via debug_best_tilts_per_case.py, not rerun here)

Run: python scripts/debug_best_tilts_per_case2.py <window> <near_a>,<near_b>,<near_c> <far_a>,<far_b>,<far_c>
e.g. python scripts/debug_best_tilts_per_case2.py hanning 8,1,1 1,1,8
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import torch
import debug_pipeline_audit as dpa
from helpers.utils import load_config

window = sys.argv[1]
near_row = [int(x) for x in sys.argv[2].split(",")]
far_row = [int(x) for x in sys.argv[3].split(",")]
# Optional 4th arg: "<start>,<end>,<step>" degrees -- overrides config.yaml's
# downtilt sweep range/resolution (default 5-13 step 2).
tilt_override = [float(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None


def _cond_load_config():
    cfg = load_config()
    cfg["antenna"]["antenna_window"] = window
    cfg["mobility"]["hotspot"]["demand_states"][1] = near_row
    cfg["mobility"]["hotspot"]["demand_states"][2] = far_row
    if tilt_override is not None:
        start, end, step = tilt_override
        cfg["downtilt"]["downtilt_sweep_start_deg"] = start
        cfg["downtilt"]["downtilt_sweep_end_deg"] = end
        cfg["downtilt"]["tilt_step_deg"] = step
    return cfg
dpa.load_config = _cond_load_config

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
print(f"CONFIG: window={window} near={near_row} far={far_row}")
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
