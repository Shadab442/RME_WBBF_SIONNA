"""Fine-grained tilt sweep, -1.0 to +1.0 deg every 0.1 deg (20 non-zero
tilts, excluding the 0.0 deg reference itself), one subplot per tilt
(4 rows x 5 cols), each showing near/mid/far NET (gained-lost) cell
count relative to tilt=0 deg. Requested 2026-10-05 to find exactly where
the tradeoff region actually starts/ends, finer than the committed
sweep's own 0.5 deg steps.
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
OTHER_SECTORS_TILT_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
# FIXED at sector0's own neutral point (2026-10-05 fix) -- NOT tracking
# other_sectors_tilt_deg. An earlier version used reference=other_sectors_
# tilt_deg (homogeneous baseline), which confounds two things at once:
# sector 0's OWN tilt-preference shift (near prefers more downtilt
# regardless of neighbors) gets mixed into what should be an isolated
# neighbor-interference effect. Keeping the reference fixed at 0.0
# isolates "how does neighbor tilt change sector 0's own sweep outcome",
# comparable across all neighbor conditions on the same baseline.
REFERENCE_TILT = 0.0

sionna.phy.config.seed = dpa.DEBUG_SEED
engine = dpa.build_debug_engine()
topo = engine.topology

# -1.0 .. 1.0 every 0.1, plus the 0.0 reference itself (included once for
# the comparison baseline, excluded from the 20 plotted panels).
fine_tilts = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.1), 1)
tilt_values = np.unique(np.concatenate([fine_tilts, [REFERENCE_TILT]]))

site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()
xy = engine.grid_points[TARGET_SECTOR]
dist = np.linalg.norm(xy - site_xy[None, :], axis=1)

region_of = np.full(dist.shape, "far", dtype=object)
region_of[dist < HOTSPOT_NEAR_DIST_M] = "near"
region_of[(dist >= HOTSPOT_NEAR_DIST_M) & (dist < HOTSPOT_FAR_DIST_M)] = "mid"
regions = ["near", "mid", "far"]
colors = {"near": "tab:red", "mid": "tab:orange", "far": "tab:blue"}

pert = dpa.compute_single_sector_tilt_perturbation(
    TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=OTHER_SECTORS_TILT_DEG,
    reference_tilt_deg=REFERENCE_TILT, engine=engine)
num_grid_cells = engine.num_grid_cells
sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
cov_all = pert["sinr_db"][:, sl] > engine.coverage_threshold_db
ref_idx = int(np.argmin(np.abs(tilt_values - REFERENCE_TILT)))
cov_ref = cov_all[ref_idx]

plot_tilts = [t for t in fine_tilts if abs(t - REFERENCE_TILT) > 1e-9]
print(f"{'tilt':>6}" + "".join(f"{r+'_net':>10}" for r in regions))

fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharey=True)
for i, tilt in enumerate(plot_tilts):
    ax = axes.flat[i]
    t = int(np.argmin(np.abs(tilt_values - tilt)))
    cov_cand = cov_all[t]
    gained = (~cov_ref) & cov_cand
    lost = cov_ref & (~cov_cand)
    nets = []
    for r in regions:
        rm = region_of == r
        net = int((gained & rm).sum()) - int((lost & rm).sum())
        nets.append(net)
    print(f"{tilt:6.1f}" + "".join(f"{v:10d}" for v in nets))

    bar_colors = [colors[r] for r in regions]
    ax.bar(regions, nets, color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_title(f"tilt={tilt:+.1f} deg", fontsize=10)
    for j, v in enumerate(nets):
        ax.text(j, v, str(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

fig.suptitle(f"Sector {TARGET_SECTOR}: net coverage change (gained-lost) by region, fine tilt sweep\n"
            f"neighbors fixed at {OTHER_SECTORS_TILT_DEG:+.1f} deg, reference = sector 0 also at {REFERENCE_TILT:+.1f} deg")
fig.tight_layout()
out_path = f"results/debug/phase2_radio_environment_validation/gained_lost_fine_grid_neighbors{OTHER_SECTORS_TILT_DEG:+.1f}.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
