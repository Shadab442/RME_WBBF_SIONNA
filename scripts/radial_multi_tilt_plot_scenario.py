"""Same seed-averaged, distance-binned coverage methodology as
radial_swap_plot_scenario.py, but for an arbitrary LIST of tilts (not just
two endpoints) plotted as separate curves on one figure -- requested
2026-10-07 to check whether a genuine THIRD regime (a real mid/far
crossover) shows up at smaller tilt magnitudes, after radial_swap_plot.py/
radial_swap_plot_scenario.py's own two-endpoint (-2.0 vs +2.0) check found
only ONE real crossover (near/mid, ~900m RMa / ~174m UMa) and no second
one anywhere out to the cell edge.

Run: python scripts/radial_multi_tilt_plot_scenario.py <scenario> <tilt1,tilt2,...>
e.g. python scripts/radial_multi_tilt_plot_scenario.py uma -0.5,0.0,0.5
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sionna
import debug_pipeline_audit as dpa
import helpers.simulation_engine as se_mod
from helpers.utils import load_config
from sionna.phy.channel.utils import set_3gpp_scenario_parameters

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "uma"
TILTS = [float(t) for t in (sys.argv[2] if len(sys.argv) > 2 else "-0.5,0.0,0.5").split(",")]
# "stochastic" (default) = config.yaml's own force_los=null/disable_shadow_fading=false
# (real per-link LOS/NLOS draw + shadow fading). "deterministic" = force_los=True,
# disable_shadow_fading=True -- pure distance-based pathloss only, no randomness at
# all, so every grid cell's RSRP/SINR at a given tilt is single-valued (no seed
# dependence) -- isolates whether core's own small tilt-swing is a real geometric
# property or just noise-masked in the stochastic version (requested 2026-10-07).
CHANNEL_MODE = sys.argv[3] if len(sys.argv) > 3 else "stochastic"
assert CHANNEL_MODE in ("stochastic", "deterministic")
SEEDS = [1, 2, 3] if CHANNEL_MODE == "stochastic" else [1]
TARGET_SECTOR = 0
RMA_ISD_M, RMA_NEAR_DIST_M, RMA_FAR_DIST_M = 5000.0, 900.0, 2300.0
# Empirically-calibrated near/mid crossover (radial_swap_plot_scenario.py,
# 2026-10-05/06) -- see that script's own docstring. far_dist_m has NO
# empirical basis (confirmed 2026-10-07: neither RMa's nor UMa's data shows
# a second crossover), kept here only for the hotspot mobility calibration
# machinery, not treated as meaningful in this script's own analysis.
_CALIBRATED_NEAR_M = {"uma": 174.2}

_, isd, _, _, _, _, _, _ = set_3gpp_scenario_parameters(SCENARIO)
isd_m = float(isd.item())
scale = isd_m / RMA_ISD_M
near_dist_m = _CALIBRATED_NEAR_M.get(SCENARIO, RMA_NEAR_DIST_M * scale)
far_dist_m = RMA_FAR_DIST_M * scale
spatial_grid_cell_size_m = 50.0 * scale
deviation_radius_m = 40.0 * scale
cluster_radius_max_m = 100.0 * scale

se_mod._HOTSPOT_NEAR_DIST_M = near_dist_m
se_mod._HOTSPOT_FAR_DIST_M = far_dist_m

def _scaled_load_config():
    cfg = load_config()
    cfg["topology"]["scenario"] = SCENARIO
    cfg["topology"]["spatial_grid_cell_size_m"] = spatial_grid_cell_size_m
    cfg["mobility"]["hotspot"]["deviation_radius_m"] = deviation_radius_m
    cfg["mobility"]["rpgm"]["cluster_radius_max_m"] = cluster_radius_max_m
    cfg["downtilt"]["downtilt_sweep_start_deg"] = min(TILTS) - 1
    cfg["downtilt"]["downtilt_sweep_end_deg"] = max(TILTS) + 1
    cfg["downtilt"]["tilt_step_deg"] = 1
    if CHANNEL_MODE == "deterministic":
        cfg["channel"]["force_los"] = True
        cfg["channel"]["disable_shadow_fading"] = True
    return cfg
dpa.load_config = _scaled_load_config

sionna.phy.config.seed = SEEDS[0]
_probe_engine = dpa.build_debug_engine()
cell_radius_m = float(_probe_engine.topology.grid.cell_radius)
print(f"scenario={SCENARIO} isd={isd_m:.0f}m cell_radius={cell_radius_m:.1f}m tilts={TILTS} channel={CHANNEL_MODE}")

bins = np.linspace(0, cell_radius_m, 30)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
all_cov = {t: [] for t in TILTS}

for seed in SEEDS:
    sionna.phy.config.seed = seed
    engine = dpa.build_debug_engine()
    topo = engine.topology
    tilt_values = np.array(TILTS)
    pert = dpa.compute_single_sector_tilt_perturbation(
        TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
    num_grid_cells = engine.num_grid_cells
    sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
    xy = engine.grid_points[TARGET_SECTOR]
    site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()
    dist = np.linalg.norm(xy - site_xy[None, :], axis=1)
    idx = np.digitize(dist, bins) - 1

    for t_i, t in enumerate(TILTS):
        cov = pert["sinr_db"][t_i][sl] > engine.coverage_threshold_db
        cov_by_bin = np.full(len(bin_centers), np.nan)
        for b in range(len(bin_centers)):
            mask = idx == b
            if mask.sum() > 0:
                cov_by_bin[b] = cov[mask].mean()
        all_cov[t].append(cov_by_bin)

means = {t: np.nanmean(np.array(all_cov[t]), axis=0) for t in TILTS}
stds = {t: np.nanstd(np.array(all_cov[t]), axis=0) for t in TILTS}

print(f"{'dist(m)':>8}" + "".join(f"tilt={t:+.1f}".rjust(12) for t in TILTS))
for i, bc in enumerate(bin_centers):
    print(f"{bc:8.0f}" + "".join(f"{means[t][i]:12.3f}" for t in TILTS))

fig, ax = plt.subplots(figsize=(10, 6))
colors = cm.coolwarm(np.linspace(0, 1, len(TILTS)))
for t, color in zip(TILTS, colors):
    ax.errorbar(bin_centers, means[t], yerr=stds[t], marker="o", label=f"tilt={t:+.1f} deg",
               color=color, capsize=3)
ax.axvline(near_dist_m, color="gray", linestyle=":", alpha=0.6)
ax.annotate(f"near/mid cut ~{near_dist_m:.0f}m\n(empirical crossover)", xy=(near_dist_m, 0.5),
           fontsize=8, color="gray")
ax.set_xlabel("distance from site (m)")
ax.set_ylabel(f"coverage fraction (mean +/- std across {len(SEEDS)} seeds)")
ax.set_title(f"{SCENARIO.upper()} sector 0, {CHANNEL_MODE} channel (force_los={CHANNEL_MODE=='deterministic'}, "
            f"no_shadow={CHANNEL_MODE=='deterministic'})\n"
            f"Coverage vs distance, multiple tilts, averaged over {len(SEEDS)} seeds")
ax.legend()
fig.tight_layout()
out_path = f"results/debug/phase2_radio_environment_validation/radial_multi_tilt_plot_{SCENARIO}_{CHANNEL_MODE}.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
