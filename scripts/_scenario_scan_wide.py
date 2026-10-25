"""Scratch: wide coarse tilt scan for UMa to find where the near-vs-(mid+far)
tradeoff is strongest/most two-sided before committing a final fine-sweep
range -- the committed -1..+1/0.1 range (copied from RMa) wasn't re-derived
for UMa's own geometry (mechanical pitch ~9.4 deg for UMa vs ~1.5 deg for
RMa -- see sector_center geometry in CellularTopology.__init__), and the
phase2 tilt_sensitivity curves at +-1.0 deg hadn't saturated yet at either
edge, suggesting a wider range might show more. Requested 2026-10-05.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import debug_pipeline_audit as dpa
import helpers.simulation_engine as se_mod
from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine
from sionna.phy.channel.utils import set_3gpp_scenario_parameters

SCENARIO = "uma"
TARGET_SECTOR = 0
REFERENCE_TILT = 0.0
RMA_ISD_M, RMA_NEAR_DIST_M, RMA_FAR_DIST_M = 5000.0, 900.0, 2300.0

_, isd, _, _, _, _, _, _ = set_3gpp_scenario_parameters(SCENARIO)
isd_m = float(isd.item())
scale = isd_m / RMA_ISD_M
near_dist_m, far_dist_m = RMA_NEAR_DIST_M * scale, RMA_FAR_DIST_M * scale
spatial_grid_cell_size_m, deviation_radius_m, cluster_radius_max_m = 50.0 * scale, 40.0 * scale, 100.0 * scale

se_mod._HOTSPOT_NEAR_DIST_M = near_dist_m
se_mod._HOTSPOT_FAR_DIST_M = far_dist_m

def _scaled_load_config():
    cfg = load_config()
    cfg["topology"]["scenario"] = SCENARIO
    cfg["topology"]["spatial_grid_cell_size_m"] = spatial_grid_cell_size_m
    cfg["mobility"]["hotspot"]["deviation_radius_m"] = deviation_radius_m
    cfg["mobility"]["rpgm"]["cluster_radius_max_m"] = cluster_radius_max_m
    return cfg

dpa.load_config = _scaled_load_config
sionna.phy.config.seed = dpa.DEBUG_SEED
engine = SimulationEngine(_scaled_load_config())
topo = engine.topology

print(f"mechanical pitch (deg) = {np.degrees(float(topo.bs_orientations[0, TARGET_SECTOR, 1])):.3f}")

site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()
xy = engine.grid_points[TARGET_SECTOR]
dist = np.linalg.norm(xy - site_xy[None, :], axis=1)
region_of = np.full(dist.shape, "far", dtype=object)
region_of[dist < near_dist_m] = "near"
region_of[(dist >= near_dist_m) & (dist < far_dist_m)] = "mid"
regions = ["near", "mid", "far"]

wide_tilts = np.round(np.arange(-2.5, 2.0 + 1e-9, 0.1), 2)
tilt_values = np.unique(np.concatenate([wide_tilts, [REFERENCE_TILT]]))

pert = dpa.compute_single_sector_tilt_perturbation(
    TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0,
    reference_tilt_deg=REFERENCE_TILT, engine=engine)
num_grid_cells = engine.num_grid_cells
sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
cov_all = pert["sinr_db"][:, sl] > engine.coverage_threshold_db
ref_idx = int(np.argmin(np.abs(tilt_values - REFERENCE_TILT)))
cov_ref = cov_all[ref_idx]

print(f"{'tilt':>6}" + "".join(f"{r+'_net':>10}" for r in regions))
for tilt in wide_tilts:
    if abs(tilt - REFERENCE_TILT) < 1e-9:
        continue
    t = int(np.argmin(np.abs(tilt_values - tilt)))
    cov_cand = cov_all[t]
    gained = (~cov_ref) & cov_cand
    lost = cov_ref & (~cov_cand)
    row = []
    for r in regions:
        rm = region_of == r
        net = int((gained & rm).sum()) - int((lost & rm).sum())
        row.append(net)
    print(f"{tilt:6.1f}" + "".join(f"{v:10d}" for v in row))
