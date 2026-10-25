import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sionna
import debug_pipeline_audit as dpa

SCENARIO = "rma"
TX_POWER_DBM = 40.0
M = 48
THRESHOLD = 3.0
SEED = 3
TILT_A, TILT_B = -1.0, 2.0
TARGET_SECTOR = 0

_orig_load_config = dpa.load_config
def _patched_load_config():
    cfg = _orig_load_config()
    cfg["topology"]["scenario"] = SCENARIO
    cfg["channel"]["bs_tx_power_dbm"] = TX_POWER_DBM
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

sionna.phy.config.seed = SEED
engine = dpa.build_debug_engine()
topo = engine.topology

tilt_values = np.array([TILT_A, TILT_B])
pert = dpa.compute_single_sector_tilt_perturbation(TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
num_grid_cells = engine.num_grid_cells
sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
xy = engine.grid_points[TARGET_SECTOR]
site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()

cov_a = pert["sinr_db"][0][sl] > engine.coverage_threshold_db
cov_b = pert["sinr_db"][1][sl] > engine.coverage_threshold_db

losers = cov_a & ~cov_b     # covered at tilt_a, lost at tilt_b
gainers = ~cov_a & cov_b    # uncovered at tilt_a, gained at tilt_b
still_cov = cov_a & cov_b
still_unc = ~cov_a & ~cov_b

category = np.zeros(cov_a.size, dtype=int)
category[still_unc] = 0
category[gainers] = 1
category[losers] = 2
category[still_cov] = 3
cmap = ListedColormap(["#a6bddb", "#1a9850", "#d73027", "#67000d"])
labels = ["still uncovered", f"gained ({TILT_A}->{TILT_B})", f"lost ({TILT_A}->{TILT_B})", "still covered"]

fig, ax = plt.subplots(figsize=(8, 7))
sc = ax.scatter(xy[:, 0], xy[:, 1], c=category, cmap=cmap, vmin=-0.5, vmax=3.5, s=6, linewidths=0)
ax.scatter(*site_xy, marker="^", color="black", s=100, zorder=5, label="site")
ax.set_aspect("equal")
ax.set_title(f"RMa m={M} thr={THRESHOLD}dB stochastic seed={SEED}\n"
            f"Sector 0 coverage swap: tilt {TILT_A} deg -> {TILT_B} deg\n"
            f"losers={losers.mean():.3f}  gainers={gainers.mean():.3f}", fontsize=11)
cbar = fig.colorbar(sc, ax=ax, ticks=[0, 1, 2, 3], fraction=0.04)
cbar.ax.set_yticklabels(labels)
fig.tight_layout()
out_path = "results/debug/phase2_radio_environment_validation/best_swap_visualize.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
print(f"losers={losers.mean():.3f} gainers={gainers.mean():.3f} n={cov_a.size}")
