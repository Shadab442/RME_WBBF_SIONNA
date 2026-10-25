"""Reruns phase 3's opt-in neighbor-joint-sweep sub-experiment (see
scripts/phase3_neighbor_joint_sweep.py's own module docstring for the full
design) with antenna.antenna_window overridden to "rectangular" instead of
the committed config.yaml default ("hanning") -- 2026-10-12, explicit
request, following the discovery that m=16/hanning leaves sector 0's own
far UEs at structurally ZERO coverage across the ENTIRE tilt sweep in the
HLL case (main lobe too wide at this element count under Hanning tapering
-- confirmed m=16/rectangular and m=32/hanning BOTH fix it, pointing at
main-lobe width specifically, not sidelobe interference). Running the full
6-case sweep under m=16/rectangular to see how much that changes today's
other neighbor-joint-sweep conclusions.

Saves to phase3_raw_ue_behavior/m16_rectangular/ -- a sibling of the
default (m16/hanning) run's own output, so both stay available side by
side for direct comparison.

Run: python scripts/phase3_neighbor_joint_sweep_m16_rect.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import torch
import debug_pipeline_audit as dpa
from helpers.utils import load_config

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _cond_load_config():
    cfg = load_config()
    cfg["antenna"]["antenna_window"] = "rectangular"
    return cfg
dpa.load_config = _cond_load_config

sionna.phy.config.seed = dpa.DEBUG_SEED
engine = dpa.build_debug_engine()
scenario = dpa.load_or_run_scenario(engine, use_cache=False)

out_dir = os.path.join(dpa.OUT_ROOT, "phase3_raw_ue_behavior", "m16_rectangular")
dpa.phase3_raw_ue_behavior(engine, scenario, out_dir, run_neighbor_joint_sweep=True)
print(f"saved: {out_dir}")
