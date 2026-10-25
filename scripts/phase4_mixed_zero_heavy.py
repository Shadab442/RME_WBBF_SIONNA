"""Phase 4 variant (2026-10-13, explicit request): same mixed demand_states
as scripts/phase4_mixed_111.py (see its own docstring for why a mixed
split is needed at all), but a SHARPER anchor condition than the uniform
"[1,1,1]" case -- for each demand state, the region that state is itself
DOMINANT in gets ZERO anchor clusters, the other two get 1 each:

  mid-heavy  [2,6,2] -> anchors [near=1, mid=0, far=1]
  near-heavy [6,3,1] -> anchors [near=0, mid=1, far=1]
  far-heavy  [1,3,6] -> anchors [near=1, mid=1, far=0]

This totals 2 of 10 clusters/sector reporting (20%) in every state --
matching the standard non_anchor_fraction=0.8 anchor RATE used everywhere
else in phase 4, just restructured as "the densely-populated region goes
completely dark" instead of a random 20% Bernoulli draw -- the worst
realistic case (e.g. a local GNSS blackout exactly where the crowd is),
not merely "fewer" anchors.

Saves to phase4_measurement_aggregation/mixed_zero_heavy/.

Run: python scripts/phase4_mixed_zero_heavy.py
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

# state_idx -> anchors_per_region [near, mid, far] -- zero out that state's
# own dominant (heavy) region, 1 cluster/sector in each of the other two.
_ZERO_HEAVY_ANCHORS = {
    0: [1, 0, 1],  # mid-heavy -- zero mid
    1: [0, 1, 1],  # near-heavy -- zero near
    2: [1, 1, 0],  # far-heavy -- zero far
}


def anchor_override_fn(engine, state_idx, trial):
    dpa._force_fixed_anchors_per_sector_region(engine, anchors_per_region=_ZERO_HEAVY_ANCHORS[state_idx], seed=trial)


out_dir = os.path.join(dpa.OUT_ROOT, "phase4_measurement_aggregation", "mixed_zero_heavy")
dpa.phase4_measurement_aggregation(engine, None, out_dir, anchor_override_fn=anchor_override_fn)
print(f"saved: {out_dir}")
