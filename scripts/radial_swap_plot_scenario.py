"""Same methodology as scripts/radial_swap_plot.py (the script that
originally calibrated RMa's near/mid crossover at ~900m, 2026-09-28/29:
seed-averaged, distance-binned coverage fraction compared at two tilt
endpoints), but parameterized over topology.scenario -- used to find
UMa's OWN empirical near/mid crossover rather than assuming it scales
linearly with isd the way cell_radius does.

Why this was needed (2026-10-05): gained_lost_fine_grid_scenario.py's
near/far cuts (90m/230m) were a naive isd-ratio rescale of RMa's own
empirically-derived 900m/2300m. But checking phase2's own
radial_crossover_single_sector.png diagnostic (already generated for UMa)
showed the REAL downtilt-favored/uptilt-favored crossover sits around
170-200m, not 90m -- because the crossover is governed by antenna
geometry (mechanical pitch, HPBW), and mechanical pitch does NOT scale by
the isd ratio (UMa's own mechanical pitch is 9.36 deg vs RMa's 1.51 deg,
a 6.2x difference, not the 10x isd ratio -- bs_height differs too, 25m vs
35m, not isd-scaled). This script re-derives the crossover directly from
real seed-averaged coverage data instead of assuming the scaling holds.
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
from sionna.phy.channel.utils import set_3gpp_scenario_parameters

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "uma"
TILT_A = float(sys.argv[2]) if len(sys.argv) > 2 else -2.0   # uptilt endpoint
TILT_B = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0    # downtilt endpoint
SEEDS = [1, 2, 3]
TARGET_SECTOR = 0
RMA_ISD_M, RMA_NEAR_DIST_M, RMA_FAR_DIST_M = 5000.0, 900.0, 2300.0

_, isd, _, _, _, _, _, _ = set_3gpp_scenario_parameters(SCENARIO)
isd_m = float(isd.item())
scale = isd_m / RMA_ISD_M
# Starting guess only -- used for spatial_grid_cell_size_m/mobility
# geometry (which DO scale with isd, since they're tied to cell_radius),
# NOT assumed correct for the near/mid cut itself, which this script
# re-derives empirically below.
near_dist_guess_m = RMA_NEAR_DIST_M * scale
far_dist_guess_m = RMA_FAR_DIST_M * scale
spatial_grid_cell_size_m = 50.0 * scale
deviation_radius_m = 40.0 * scale
cluster_radius_max_m = 100.0 * scale

se_mod._HOTSPOT_NEAR_DIST_M = near_dist_guess_m
se_mod._HOTSPOT_FAR_DIST_M = far_dist_guess_m

def _scaled_load_config():
    cfg = load_config()
    cfg["topology"]["scenario"] = SCENARIO
    cfg["topology"]["spatial_grid_cell_size_m"] = spatial_grid_cell_size_m
    cfg["mobility"]["hotspot"]["deviation_radius_m"] = deviation_radius_m
    cfg["mobility"]["rpgm"]["cluster_radius_max_m"] = cluster_radius_max_m
    cfg["downtilt"]["downtilt_sweep_start_deg"] = min(TILT_A, TILT_B) - 1
    cfg["downtilt"]["downtilt_sweep_end_deg"] = max(TILT_A, TILT_B) + 1
    cfg["downtilt"]["tilt_step_deg"] = 1
    return cfg
dpa.load_config = _scaled_load_config

# Probe cell_radius once to size the distance bins correctly for this scenario.
sionna.phy.config.seed = SEEDS[0]
_probe_engine = dpa.build_debug_engine()
cell_radius_m = float(_probe_engine.topology.grid.cell_radius)
print(f"scenario={SCENARIO} isd={isd_m:.0f}m cell_radius={cell_radius_m:.1f}m "
     f"tilt_a={TILT_A} (uptilt) tilt_b={TILT_B} (downtilt)")

bins = np.linspace(0, cell_radius_m, 30)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
all_cov_a, all_cov_b = [], []

for seed in SEEDS:
    sionna.phy.config.seed = seed
    engine = dpa.build_debug_engine()
    topo = engine.topology
    tilt_values = np.array([TILT_A, TILT_B])
    pert = dpa.compute_single_sector_tilt_perturbation(
        TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
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

all_cov_a, all_cov_b = np.array(all_cov_a), np.array(all_cov_b)
mean_a, mean_b = np.nanmean(all_cov_a, axis=0), np.nanmean(all_cov_b, axis=0)
std_a, std_b = np.nanstd(all_cov_a, axis=0), np.nanstd(all_cov_b, axis=0)
diff = mean_b - mean_a   # >0: downtilt(B) favored, <0: uptilt(A) favored

# First bin (from the near side) where diff goes negative and STAYS
# negative for the rest of the profile -- the near/mid crossover.
crossover_m = None
for i in range(len(bin_centers)):
    if diff[i] < 0 and np.all(diff[i:] < 0.02):  # small tolerance for noise
        crossover_m = bin_centers[i]
        break
print(f"{'dist(m)':>8}{'cov_a(uptilt)':>16}{'cov_b(downtilt)':>16}{'diff':>10}")
for i, bc in enumerate(bin_centers):
    print(f"{bc:8.0f}{mean_a[i]:16.3f}{mean_b[i]:16.3f}{diff[i]:+10.3f}")
print(f"empirical near/mid crossover ~= {crossover_m:.1f}m" if crossover_m is not None
     else "no stable crossover found")

fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(bin_centers, mean_a, yerr=std_a, marker="o", label=f"tilt={TILT_A} deg (uptilt)", color="tab:blue", capsize=3)
ax.errorbar(bin_centers, mean_b, yerr=std_b, marker="s", label=f"tilt={TILT_B} deg (downtilt)", color="tab:red", capsize=3)
ax.set_xlabel("distance from site (m)")
ax.set_ylabel("coverage fraction (mean +/- std across 3 seeds)")
ax.set_title(f"{SCENARIO.upper()} sector 0, stochastic LOS/shadow\n"
            f"Coverage vs distance at tilt={TILT_A} (uptilt) vs {TILT_B} (downtilt), averaged over {len(SEEDS)} seeds")
ax.legend()
if crossover_m is not None:
    ax.axvline(crossover_m, color="gray", linestyle=":", alpha=0.6)
    ax.annotate(f"crossover ~{crossover_m:.0f}m", xy=(crossover_m, 0.75), fontsize=9, color="gray")
fig.tight_layout()
out_path = f"results/debug/phase2_radio_environment_validation/radial_swap_plot_{SCENARIO}.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
