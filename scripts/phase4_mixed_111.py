"""Phase 4 variant (2026-10-13, explicit request): the structural
near/mid/far-leakage finding showed PURE demand splits (the committed
config.yaml's current [10,0,0]/[0,0,10]) can't actually stress-test the
anchor-unavailability question -- the dominant region always has so many
UEs that enough survive random 80% non-anchor dropout regardless. This
variant uses a MIXED demand_states split (every region genuinely present
in every state) specifically so a region can have few enough clusters that
losing its anchors is a real, plausible outcome, not just noisier
statistics.

demand_states override (row order mid/near/far, all sum to 10):
  mid-heavy  [2,6,2]
  near-heavy [6,3,1]
  far-heavy  [1,3,6]

Anchor rule: exactly 1 anchor CLUSTER (all its own UEs, not a random
Bernoulli UE fraction) per sector per region, uniformly -- see
_force_fixed_anchors_per_sector_region's own docstring. This is the
"[1,1,1]" condition -- 3 of 10 clusters/sector report (30%), same count
regardless of which region is that state's own dominant one.

Saves to phase4_measurement_aggregation/mixed_111/.

Run: python scripts/phase4_mixed_111.py
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


def anchor_override_fn(engine, state_idx, trial):
    dpa._force_fixed_anchors_per_sector_region(engine, anchors_per_region=1, seed=trial)


out_dir = os.path.join(dpa.OUT_ROOT, "phase4_measurement_aggregation", "mixed_111")
dpa.phase4_measurement_aggregation(engine, None, out_dir, anchor_override_fn=anchor_override_fn)
print(f"saved: {out_dir}")
