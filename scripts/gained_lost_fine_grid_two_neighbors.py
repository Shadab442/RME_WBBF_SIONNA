"""Same fine-grid gained/lost analysis as gained_lost_fine_grid.py, but
instead of moving all 20 other sectors together, move ONLY the two
empirically-identified top interferers to sector 0's own mid cells
(sector 2, a same-site sibling, and sector 8, the cross-site neighbor) --
everyone else, including sector 0's own reference, stays at 0 deg. Much
cleaner signal than the all-20-sectors version. Requested 2026-10-05.
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
from debug_pipeline_audit import _virtual_grid_state

TARGET_SECTOR = 0
TOP_NEIGHBOR_SECTORS = [2, 8]
HOTSPOT_NEAR_DIST_M = 900.0
HOTSPOT_FAR_DIST_M = 2300.0
REFERENCE_TILT = 0.0
NEIGHBORS_TILT_DEG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0

sionna.phy.config.seed = dpa.DEBUG_SEED
engine = dpa.build_debug_engine()
topo = engine.topology

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

num_virtual = xy.shape[0]
batch_size = 800
sinr_by_tilt = np.zeros((len(tilt_values), num_virtual))
with __import__("torch").no_grad():
    for start in range(0, num_virtual, batch_size):
        end = min(start + batch_size, num_virtual)
        chunk_state = _virtual_grid_state(engine, xy=xy[start:end])
        for t, tilt in enumerate(tilt_values):
            tilt_vec = np.zeros(engine.num_bs)
            tilt_vec[TARGET_SECTOR] = tilt
            for s in TOP_NEIGHBOR_SECTORS:
                tilt_vec[s] = NEIGHBORS_TILT_DEG
            power = engine.kpi_manager.resolve_power_at_tilt(chunk_state, tilt_vec)[0]
            kpis = engine.kpi_manager.compute_ue_kpis(power[None, :, :])
            sinr_by_tilt[t, start:end] = kpis["sinr_db"][0]

cov_all = sinr_by_tilt > engine.coverage_threshold_db
ref_idx = int(np.argmin(np.abs(tilt_values - REFERENCE_TILT)))
cov_ref = cov_all[ref_idx]

plot_tilts = [t for t in fine_tilts if abs(t - REFERENCE_TILT) > 1e-9]
print(f"Two-neighbor tilt={NEIGHBORS_TILT_DEG:+.1f} (sectors {TOP_NEIGHBOR_SECTORS}), sector0 ref={REFERENCE_TILT:.1f}")
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

fig.suptitle(f"Sector {TARGET_SECTOR}: net coverage change by region, fine tilt sweep\n"
            f"ONLY sectors {TOP_NEIGHBOR_SECTORS} (top-2 interferers) fixed at {NEIGHBORS_TILT_DEG:+.1f} deg, "
            f"all others incl. reference at 0.0 deg")
fig.tight_layout()
out_path = f"results/debug/phase2_radio_environment_validation/gained_lost_two_neighbors{NEIGHBORS_TILT_DEG:+.1f}.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
