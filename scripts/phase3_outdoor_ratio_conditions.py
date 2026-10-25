"""Runs phase 3 (raw UE behavior) with topology.outdoor_ue_ratio_override
set to null (2026-10-11, explicit request) -- uses the scenario's own
3GPP-derived indoor_probability instead of config.yaml's own committed 1.0
(all outdoor) default (UMa: 0.8 indoor, i.e. 0.2 outdoor). Everything else
as COMMITTED config.yaml (UMa, m=16 hanning, LOS no shadowing,
hotspot_demand-periodic mobility, near/mid/far-heavy demand_states).

Saves to phase3_raw_ue_behavior/outdoor_ratio_null -- a sibling of the
PLAIN `python scripts/debug_pipeline_audit.py --phase 3` run's own default
output (results/debug/phase3_raw_ue_behavior/, config.yaml's literal
committed outdoor_ue_ratio_override=1.0), which this script does NOT
re-run (identical config, would be pure waste -- run that separately/first
if it isn't already there).

Run: python scripts/phase3_outdoor_ratio_conditions.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import torch
import debug_pipeline_audit as dpa
from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"default (outdoor_ue_ratio_override=1.0) run separately -- see "
     f"{os.path.join(dpa.OUT_ROOT, 'phase3_raw_ue_behavior')}/")

# name -> outdoor_ue_ratio_override
CONDITIONS = {
    "outdoor_ratio_null": None,
}

for name, outdoor_ue_ratio_override in CONDITIONS.items():
    print(f"\n=== {name}: topology.outdoor_ue_ratio_override={outdoor_ue_ratio_override} ===")

    def _cond_load_config(outdoor_ue_ratio_override=outdoor_ue_ratio_override):
        cfg = load_config()
        cfg["topology"]["outdoor_ue_ratio_override"] = outdoor_ue_ratio_override
        return cfg
    dpa.load_config = _cond_load_config

    sionna.phy.config.seed = dpa.DEBUG_SEED
    engine = SimulationEngine(_cond_load_config())
    scenario = dpa.load_or_run_scenario(engine, use_cache=False)

    out_dir = os.path.join(dpa.OUT_ROOT, "phase3_raw_ue_behavior", name)
    dpa.phase3_raw_ue_behavior(engine, scenario, out_dir)
    print(f"saved: {out_dir}")
