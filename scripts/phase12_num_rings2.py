"""Runs phase 1 + phase 2 with topology.num_rings=2 (19 sites, 57 sectors,
vs the committed num_rings=1's 7 sites/21 sectors) -- everything else as
committed (UMa, stochastic LOS + shadowing, m=32 rectangular, downtilt
5-13/2 deg, outdoor UEs). topology.num_ut is overridden to match (57*5*5=
1425, same per-sector UE density as the committed 21*5*5=525). Requested
2026-10-08 to check how much a bigger (but still finite, non-wraparound)
topology changes coverage, following the num_rings=1-vs-2 sector-0-only
spot check earlier, which showed little change for ONE already-interior-ish
sector -- this runs the full phase1/2 diagnostic suite instead.

Run: python scripts/phase12_num_rings2.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import debug_pipeline_audit as dpa
from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine

NUM_RINGS = 2
NAME = "num_rings2"

def _cond_load_config():
    cfg = load_config()
    cfg["topology"]["num_rings"] = NUM_RINGS
    num_bs = 3 * (1 + 3 * NUM_RINGS * (NUM_RINGS + 1))  # 3 sectors/site x num sites in NUM_RINGS-ring hex layout
    # num_ues_per_cluster is now DERIVED (SimulationEngine.__init__,
    # 2026-10-09) from num_ut // (num_bs * num_clusters_per_sector) -- only
    # num_ut itself needs overriding here to preserve the SAME per-cluster
    # density (5 UEs/cluster) as the committed num_rings=1 config.
    cfg["topology"]["num_ut"] = num_bs * cfg["mobility"]["common"]["num_clusters_per_sector"] * 5
    return cfg
dpa.load_config = _cond_load_config

print(f"=== {NAME}: num_rings={NUM_RINGS} ===")
sionna.phy.config.seed = dpa.DEBUG_SEED
engine = SimulationEngine(_cond_load_config())
print(f"num_bs={engine.num_bs} num_ut={engine.num_ut}")

phase1_out_dir = os.path.join(dpa.OUT_ROOT, "phase1_static_geometry", NAME)
dpa.phase1_static_geometry(engine, phase1_out_dir)

phase2_out_dir = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", NAME)
dpa.phase2_radio_environment_validation(engine, None, phase2_out_dir)
print(f"saved: {phase1_out_dir}  and  {phase2_out_dir}")
