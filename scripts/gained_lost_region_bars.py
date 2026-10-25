"""Simple grouped bar chart per tilt: for near/mid/far (current distance
cuts), the NUMBER of cells that gained or lost coverage relative to
tilt=0 deg. One figure per non-zero tilt in the configured sweep.
Requested 2026-10-05 (azimuth dropped per explicit instruction -- not
wanted).
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

pert = dpa.compute_single_sector_tilt_perturbation(
    TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
num_grid_cells = engine.num_grid_cells
sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
cov_all = pert["sinr_db"][:, sl] > engine.coverage_threshold_db
ref_idx = int(np.argmin(np.abs(tilt_values - REFERENCE_TILT)))
cov_ref = cov_all[ref_idx]

print(f"{'tilt':>6}{'region':>7}{'gained':>8}{'lost':>8}")
for t, tilt in enumerate(tilt_values):
    if abs(tilt - REFERENCE_TILT) < 1e-9:
        continue
    cov_cand = cov_all[t]
    gained = (~cov_ref) & cov_cand
    lost = cov_ref & (~cov_cand)

    gained_counts = [int((gained & (region_of == r)).sum()) for r in regions]
    lost_counts = [int((lost & (region_of == r)).sum()) for r in regions]
    for r, g, l in zip(regions, gained_counts, lost_counts):
        print(f"{tilt:6.1f}{r:>7}{g:8d}{l:8d}")

    x = np.arange(len(regions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(x - width/2, gained_counts, width, color="tab:green", label="gained")
    ax.bar(x + width/2, lost_counts, width, color="tab:red", label="lost")
    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("number of cells")
    ax.set_title(f"Sector {TARGET_SECTOR}: cells gained/lost coverage, tilt {REFERENCE_TILT:.1f} -> {tilt:.1f} deg")
    ax.legend()
    for i, (g, l) in enumerate(zip(gained_counts, lost_counts)):
        ax.text(i - width/2, g, str(g), ha="center", va="bottom", fontsize=9)
        ax.text(i + width/2, l, str(l), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path = f"results/debug/phase2_radio_environment_validation/gained_lost_bars_tilt{tilt:+.1f}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")
