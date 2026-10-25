"""Runs phase 7 (state representation comparison) with the mixed
demand_states split (2026-10-13, matches the phase4 mixed_111/mixed_zero_heavy
convention -- pure splits can't stress-test anchor sparsity meaningfully,
see that discussion), m=16/hanning and everything else inherited from the
committed config.yaml as-is.

demand_states override (row order mid/near/far, all sum to 10):
  mid-heavy  [2,6,2]
  near-heavy [6,3,1]
  far-heavy  [1,3,6]

Run: python scripts/phase7_default.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import torch
import debug_pipeline_audit as dpa
from helpers.utils import load_config


def _cond_load_config():
    cfg = load_config()
    cfg["mobility"]["hotspot"]["demand_states"] = [[2, 6, 2], [6, 3, 1], [1, 3, 6]]
    return cfg
dpa.load_config = _cond_load_config

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
sionna.phy.config.seed = dpa.DEBUG_SEED

engine = dpa.build_debug_engine()
out_dir = os.path.join(dpa.OUT_ROOT, "phase7_state_representation")
dpa.phase7_state_representation(engine, None, out_dir)
print(f"saved: {out_dir}")
