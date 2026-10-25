"""Same fine-grid gained/lost analysis as gained_lost_fine_grid.py (-1.0 to
+1.0 deg every 0.1 deg, near/mid/far net coverage change vs tilt=0 deg), but
parameterized over topology.scenario -- checks whether UMa/UMi show a
stronger two-sided tradeoff than the committed RMa result (m=32,
threshold=0dB). Requested 2026-10-05.

_HOTSPOT_NEAR_DIST_M/_HOTSPOT_FAR_DIST_M (900/2300m) were calibrated for
RMa's isd=5000m. UMa (isd=500m) and UMi (isd=200m) have proportionally
smaller cells (cell_radius scales linearly with isd -- confirmed: rma
2886.75m, uma 288.68m, umi 115.47m, each exactly isd/sqrt(3)), so the
near/far cuts are rescaled by the SAME isd ratio rather than reused as
absolute meters (which would classify almost every UMa/UMi cell as "far").

config.yaml's committed scenario/antenna/channel/kpi settings (rma, m=32,
3.5GHz, threshold=0dB) are NOT modified -- this script builds its own
engine from an in-memory cfg override, same pattern as the earlier m=64/
2.12GHz scratch explorations.
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
import helpers.simulation_engine as se_mod
from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "uma"
TARGET_SECTOR = 0
REFERENCE_TILT = 0.0
RMA_ISD_M = 5000.0
RMA_NEAR_DIST_M = 900.0
RMA_FAR_DIST_M = 2300.0

# Empirically-calibrated near/mid crossover (radial_swap_plot_scenario.py,
# seed-averaged distance-binned coverage swap, 2026-10-05) -- NOT the naive
# isd-ratio rescale of RMa's 900m (that gives 90m for uma, but the real
# crossover is governed by antenna geometry/mechanical pitch, which doesn't
# scale with isd the way cell_radius does; verified crossover ~174.2m).
# far_dist_m DOES still use the isd-ratio rescale -- RMa's own 2300m sits at
# ~80% of ITS cell_radius, same fraction the isd scaling already reproduces,
# and there's no second empirical crossover in the data (same as RMa's own
# calibration, which didn't independently derive 2300m either).
_CALIBRATED_NEAR_M = {"uma": 174.2}

from sionna.phy.channel.utils import set_3gpp_scenario_parameters
_, isd, _, _, _, _, _, _ = set_3gpp_scenario_parameters(SCENARIO)
isd_m = float(isd.item())
scale = isd_m / RMA_ISD_M
near_dist_m = _CALIBRATED_NEAR_M.get(SCENARIO, RMA_NEAR_DIST_M * scale)
far_dist_m = RMA_FAR_DIST_M * scale
print(f"scenario={SCENARIO} isd={isd_m:.0f}m scale={scale:.4f} "
     f"near_dist={near_dist_m:.1f}m ({'empirical' if SCENARIO in _CALIBRATED_NEAR_M else 'isd-scaled guess'}) "
     f"far_dist={far_dist_m:.1f}m (isd-scaled)")

# Monkeypatch the near/mid/far calibration constants BEFORE building the
# engine -- _calibrate_hotspot_region_lut (called inside SimulationEngine.
# __init__) reads the module-level globals at call time.
se_mod._HOTSPOT_NEAR_DIST_M = near_dist_m
se_mod._HOTSPOT_FAR_DIST_M = far_dist_m

# cell_radius scales linearly with isd (confirmed: rma/uma/umi all exactly
# isd/sqrt(3)), so every OTHER absolute-meter geometry knob tuned for RMa's
# isd=5000m needs the SAME scale factor, not just the near/far cuts --
# otherwise e.g. spatial_grid_cell_size_m=50m (RMa) would leave UMa's
# ~289m-radius sector with only a 6x6=36-cell grid (vs RMa's 58x58=3364),
# too coarse for a comparable gained/lost cell count.
spatial_grid_cell_size_m = 50.0 * scale
deviation_radius_m = 40.0 * scale
cluster_radius_max_m = 100.0 * scale
print(f"scaled: spatial_grid_cell_size_m={spatial_grid_cell_size_m:.2f}m "
     f"deviation_radius_m={deviation_radius_m:.2f}m cluster_radius_max_m={cluster_radius_max_m:.2f}m")

cfg = load_config()
cfg["topology"]["scenario"] = SCENARIO
cfg["simulation"]["num_tilt_control_intervals"] = dpa.NUM_NO_TILT + dpa.NUM_RANDOM
cfg["topology"]["spatial_grid_cell_size_m"] = spatial_grid_cell_size_m
cfg["mobility"]["hotspot"]["deviation_radius_m"] = deviation_radius_m
cfg["mobility"]["rpgm"]["cluster_radius_max_m"] = cluster_radius_max_m

sionna.phy.config.seed = dpa.DEBUG_SEED
engine = SimulationEngine(cfg)
topo = engine.topology

fine_tilts = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.1), 1)
tilt_values = np.unique(np.concatenate([fine_tilts, [REFERENCE_TILT]]))

site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()
xy = engine.grid_points[TARGET_SECTOR]
dist = np.linalg.norm(xy - site_xy[None, :], axis=1)

region_of = np.full(dist.shape, "far", dtype=object)
region_of[dist < near_dist_m] = "near"
region_of[(dist >= near_dist_m) & (dist < far_dist_m)] = "mid"
regions = ["near", "mid", "far"]
colors = {"near": "tab:red", "mid": "tab:orange", "far": "tab:blue"}
pop = {r: int((region_of == r).sum()) for r in regions}
print(f"populations: {pop}")

pert = dpa.compute_single_sector_tilt_perturbation(
    TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0,
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

fig.suptitle(f"Sector {TARGET_SECTOR} ({SCENARIO}): net coverage change (gained-lost) by region, fine tilt sweep\n"
            f"neighbors fixed at 0.0 deg, reference = sector 0 also at {REFERENCE_TILT:+.1f} deg "
            f"(near<{near_dist_m:.0f}m, mid<{far_dist_m:.0f}m, far>={far_dist_m:.0f}m)")
fig.tight_layout()
out_path = f"results/debug/phase2_radio_environment_validation/gained_lost_fine_grid_{SCENARIO}.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
