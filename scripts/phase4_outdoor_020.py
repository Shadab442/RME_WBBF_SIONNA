"""Reruns phase 4 (measurement aggregation) with topology.outdoor_ue_ratio_
override overridden to 0.2 (80% indoor UEs) -- 2026-10-13, explicit request.
Note: channel.force_los=true is committed, so indoor UEs here stay LOS, not
NLoS (force_los overrides any indoor-dependent stochastic LOS draw) -- this
experiment isolates the O2I pathloss-penalty effect on anchor-only's own
measurement accuracy, not an LOS-state change. Everything else (antenna,
demand_states) inherited from the committed config.yaml as-is.

Saves to phase4_measurement_aggregation/outdoor_020/ -- a sibling of the
default run's own output, same convention as phase 3's variant subfolders.

Run: python scripts/phase4_outdoor_020.py
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
    cfg["topology"]["outdoor_ue_ratio_override"] = 0.2
    return cfg
dpa.load_config = _cond_load_config

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
sionna.phy.config.seed = dpa.DEBUG_SEED

engine = dpa.build_debug_engine()
out_dir = os.path.join(dpa.OUT_ROOT, "phase4_measurement_aggregation", "outdoor_020")
dpa.phase4_measurement_aggregation(engine, None, out_dir)
print(f"saved: {out_dir}")
