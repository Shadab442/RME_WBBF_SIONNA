"""Single grouped bar chart: x-axis = tilt, grouped bars = near/mid/far
NET (gained - lost) cell count relative to tilt=0 deg, for every tilt in
the configured sweep. Reformats gained_lost_region_bars.py's per-tilt
figures into one combined view. Requested 2026-10-05.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sionna
import debug_pipeline_audit as dpa

TARGET_SECTOR = 0
HOTSPOT_NEAR_DIST_M = 900.0
HOTSPOT_FAR_DIST_M = 2300.0
REFERENCE_TILT = 0.0

sionna.phy.config.seed = dpa.DEBUG_SEED
engine = dpa.build_debug_engine()
topo = engine.topology

tilt_values = engine.downtilt_sweep_deg
site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()
xy = engine.grid_points[TARGET_SECTOR]
dist = np.linalg.norm(xy - site_xy[None, :], axis=1)

region_of = np.full(dist.shape, "far", dtype=object)
region_of[dist < HOTSPOT_NEAR_DIST_M] = "near"
region_of[(dist >= HOTSPOT_NEAR_DIST_M) & (dist < HOTSPOT_FAR_DIST_M)] = "mid"
regions = ["near", "mid", "far"]
colors = {"near": "tab:red", "mid": "tab:orange", "far": "tab:blue"}

pert = dpa.compute_single_sector_tilt_perturbation(
    TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
num_grid_cells = engine.num_grid_cells
sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
cov_all = pert["sinr_db"][:, sl] > engine.coverage_threshold_db
ref_idx = int(np.argmin(np.abs(tilt_values - REFERENCE_TILT)))
cov_ref = cov_all[ref_idx]

non_zero_tilts = [t for t in tilt_values if abs(t - REFERENCE_TILT) > 1e-9]
net_by_region = {r: [] for r in regions}
print(f"{'tilt':>6}" + "".join(f"{r+'_net':>10}" for r in regions))
for tilt in non_zero_tilts:
    t = int(np.argmin(np.abs(tilt_values - tilt)))
    cov_cand = cov_all[t]
    gained = (~cov_ref) & cov_cand
    lost = cov_ref & (~cov_cand)
    row = []
    for r in regions:
        rm = region_of == r
        net = int((gained & rm).sum()) - int((lost & rm).sum())
        net_by_region[r].append(net)
        row.append(net)
    print(f"{tilt:6.1f}" + "".join(f"{v:10d}" for v in row))

x = np.arange(len(non_zero_tilts))
width = 0.25
fig, ax = plt.subplots(figsize=(14, 6))
for i, r in enumerate(regions):
    offset = (i - 1) * width
    bars = ax.bar(x + offset, net_by_region[r], width, label=r, color=colors[r])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{t:+.1f}" for t in non_zero_tilts])
ax.set_xlabel("tilt (deg)")
ax.set_ylabel(f"net cells (gained - lost) vs tilt={REFERENCE_TILT:.1f} deg")
ax.set_title(f"Sector {TARGET_SECTOR}: net coverage change by region, across the tilt sweep")
ax.legend()
fig.tight_layout()
out_path = "results/debug/phase2_radio_environment_validation/gained_lost_net_by_tilt.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
