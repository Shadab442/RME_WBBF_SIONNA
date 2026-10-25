import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sionna
import debug_pipeline_audit as dpa

SCENARIO = "rma"
M = 48
THRESHOLD = 3.0
SEEDS = [1, 2, 3]
TILT_A, TILT_B = -1.0, 2.0
TARGET_SECTOR = 0

_orig_load_config = dpa.load_config
def _patched_load_config():
    cfg = _orig_load_config()
    cfg["topology"]["scenario"] = SCENARIO
    cfg["channel"]["bs_tx_power_dbm"] = 40.0
    cfg["channel"]["force_los"] = None
    cfg["channel"]["disable_shadow_fading"] = False
    cfg["antenna"]["m"] = M
    cfg["antenna"]["antenna_window"] = "rectangular"
    cfg["kpi"]["coverage_threshold_db"] = THRESHOLD
    cfg["downtilt"]["downtilt_sweep_start_deg"] = -5
    cfg["downtilt"]["downtilt_sweep_end_deg"] = 5
    cfg["downtilt"]["tilt_step_deg"] = 1
    return cfg
dpa.load_config = _patched_load_config

bins = np.linspace(0, 2862, 15)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
all_cov_a, all_cov_b = [], []

for seed in SEEDS:
    sionna.phy.config.seed = seed
    engine = dpa.build_debug_engine()
    topo = engine.topology
    tilt_values = np.array([TILT_A, TILT_B])
    pert = dpa.compute_single_sector_tilt_perturbation(TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
    num_grid_cells = engine.num_grid_cells
    sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
    xy = engine.grid_points[TARGET_SECTOR]
    site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()
    dist = np.linalg.norm(xy - site_xy[None, :], axis=1)
    cov_a = pert["sinr_db"][0][sl] > engine.coverage_threshold_db
    cov_b = pert["sinr_db"][1][sl] > engine.coverage_threshold_db
    idx = np.digitize(dist, bins) - 1

    cov_a_by_bin = np.full(len(bin_centers), np.nan)
    cov_b_by_bin = np.full(len(bin_centers), np.nan)
    for b in range(len(bin_centers)):
        mask = idx == b
        if mask.sum() > 0:
            cov_a_by_bin[b] = cov_a[mask].mean()
            cov_b_by_bin[b] = cov_b[mask].mean()
    all_cov_a.append(cov_a_by_bin)
    all_cov_b.append(cov_b_by_bin)

all_cov_a = np.array(all_cov_a)
all_cov_b = np.array(all_cov_b)
mean_a, mean_b = np.nanmean(all_cov_a, axis=0), np.nanmean(all_cov_b, axis=0)
std_a, std_b = np.nanstd(all_cov_a, axis=0), np.nanstd(all_cov_b, axis=0)

fig, ax = plt.subplots(figsize=(9, 6))
ax.errorbar(bin_centers, mean_a, yerr=std_a, marker="o", label=f"tilt={TILT_A} deg (uptilt)", color="tab:blue", capsize=3)
ax.errorbar(bin_centers, mean_b, yerr=std_b, marker="s", label=f"tilt={TILT_B} deg (downtilt)", color="tab:red", capsize=3)
ax.set_xlabel("distance from site (m)")
ax.set_ylabel("coverage fraction (mean +/- std across 3 seeds)")
ax.set_title(f"RMa m={M} thr={THRESHOLD}dB, sector 0, stochastic LOS/shadow\n"
            f"Coverage vs distance at the two swap-endpoint tilts (averaged over 3 seeds)")
ax.legend()
ax.axvline(900, color="gray", linestyle=":", alpha=0.6)
ax.annotate("crossover ~900m", xy=(900, 0.75), fontsize=9, color="gray")
fig.tight_layout()
out_path = "results/debug/phase2_radio_environment_validation/radial_swap_plot.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
for i, bc in enumerate(bin_centers):
    print(f"{bc:7.0f}m  tilt{TILT_A}={mean_a[i]:.3f}  tilt{TILT_B}={mean_b[i]:.3f}  diff={mean_b[i]-mean_a[i]:+.3f}")
