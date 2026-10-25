"""Single grouped-bar tradeoff curve (x-axis=tilt, near/mid/far net
gained-lost bars), parameterized over topology.scenario AND the tilt sweep
range -- same chart style as gained_lost_net_by_tilt.py, but for a
non-default scenario with its own best-tradeoff range (NOT necessarily the
committed RMa -1..+1/0.1 window).

Range-finding (2026-10-05, UMa): a wide coarse scan (-6..+6/0.5) then a
fine scan (-2.5..+2.0/0.1) of sector 0's own tilt (neighbors fixed at 0 deg)
showed mid/far's net magnitude roughly DOUBLES by extending from +-1.0 to
+-2.0 deg (far: +43/-36 at +-1.0 -> +65/-59 at +-2.0; mid grows even more
on the downtilt side, -32 at +1.0 -> -103 at +2.0) while staying two-sided
(near opposes mid+far) and without the outright sign-reversal collapse
seen beyond -2.5 deg uptilt (mid's own net starts flipping sign
non-monotonically there) or the accelerating runaway past +2.0 deg
downtilt. near itself stays weak (roughly +-10 cells) across the WHOLE
range tested, -6 to +6 deg -- it doesn't have a wider-range payoff the way
mid/far do, so widening is purely a mid/far gain, not a near gain.
Landed on -2.0..+2.0 deg, 0.1 step as the better range: same round,
symmetric convention as RMa's committed sweep, captures most of both
regions' clean-monotonic growth before the edges degrade.
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
from sionna.phy.channel.utils import set_3gpp_scenario_parameters

SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "uma"
TILT_START = float(sys.argv[2]) if len(sys.argv) > 2 else -2.0
TILT_END = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0
TILT_STEP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1
ANTENNA_M = int(sys.argv[5]) if len(sys.argv) > 5 else 32
ANTENNA_WINDOW = sys.argv[6] if len(sys.argv) > 6 else "rectangular"
# Empirically-calibrated overrides (radial_swap_plot_scenario.py, 2026-10-05)
# -- None falls back to the naive isd-ratio rescale of RMa's 900m/2300m,
# which radial_swap_plot_scenario.py's own seed-averaged crossover check
# showed is WRONG for UMa (real crossover ~174m, not 90m) because the
# crossover depends on antenna geometry (mechanical pitch), which doesn't
# scale with isd the way cell_radius does.
NEAR_DIST_OVERRIDE_M = float(sys.argv[7]) if len(sys.argv) > 7 else None
FAR_DIST_OVERRIDE_M = float(sys.argv[8]) if len(sys.argv) > 8 else None
TARGET_SECTOR = 0
REFERENCE_TILT = 0.0
RMA_ISD_M, RMA_NEAR_DIST_M, RMA_FAR_DIST_M = 5000.0, 900.0, 2300.0
# See gained_lost_fine_grid_scenario.py's own comment -- empirical near/mid
# crossover from radial_swap_plot_scenario.py, NOT isd-ratio scaling.
_CALIBRATED_NEAR_M = {"uma": 174.2}

_, isd, _, _, _, _, _, _ = set_3gpp_scenario_parameters(SCENARIO)
isd_m = float(isd.item())
scale = isd_m / RMA_ISD_M
near_dist_m = NEAR_DIST_OVERRIDE_M if NEAR_DIST_OVERRIDE_M is not None \
    else _CALIBRATED_NEAR_M.get(SCENARIO, RMA_NEAR_DIST_M * scale)
far_dist_m = FAR_DIST_OVERRIDE_M if FAR_DIST_OVERRIDE_M is not None else RMA_FAR_DIST_M * scale
spatial_grid_cell_size_m, deviation_radius_m, cluster_radius_max_m = 50.0 * scale, 40.0 * scale, 100.0 * scale
print(f"scenario={SCENARIO} m={ANTENNA_M} window={ANTENNA_WINDOW} isd={isd_m:.0f}m scale={scale:.4f} "
     f"near_dist={near_dist_m:.1f}m far_dist={far_dist_m:.1f}m "
     f"({'empirical override' if NEAR_DIST_OVERRIDE_M is not None else 'isd-scaled guess'}) "
     f"tilt_range=[{TILT_START},{TILT_END}] step={TILT_STEP}")

se_mod._HOTSPOT_NEAR_DIST_M = near_dist_m
se_mod._HOTSPOT_FAR_DIST_M = far_dist_m

def _scaled_load_config():
    cfg = load_config()
    cfg["topology"]["scenario"] = SCENARIO
    cfg["antenna"]["m"] = ANTENNA_M
    cfg["antenna"]["antenna_window"] = ANTENNA_WINDOW
    cfg["topology"]["spatial_grid_cell_size_m"] = spatial_grid_cell_size_m
    cfg["mobility"]["hotspot"]["deviation_radius_m"] = deviation_radius_m
    cfg["mobility"]["rpgm"]["cluster_radius_max_m"] = cluster_radius_max_m
    return cfg

dpa.load_config = _scaled_load_config
sionna.phy.config.seed = dpa.DEBUG_SEED
engine = SimulationEngine(_scaled_load_config())
topo = engine.topology

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

sweep_tilts = np.round(np.arange(TILT_START, TILT_END + 1e-9, TILT_STEP), 2)
tilt_values = np.unique(np.concatenate([sweep_tilts, [REFERENCE_TILT]]))

pert = dpa.compute_single_sector_tilt_perturbation(
    TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0,
    reference_tilt_deg=REFERENCE_TILT, engine=engine)
num_grid_cells = engine.num_grid_cells
sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
cov_all = pert["sinr_db"][:, sl] > engine.coverage_threshold_db
ref_idx = int(np.argmin(np.abs(tilt_values - REFERENCE_TILT)))
cov_ref = cov_all[ref_idx]

non_zero_tilts = [t for t in sweep_tilts if abs(t - REFERENCE_TILT) > 1e-9]
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
    print(f"{tilt:6.2f}" + "".join(f"{v:10d}" for v in row))

x = np.arange(len(non_zero_tilts))
width = 0.25
fig, ax = plt.subplots(figsize=(18, 6))
for i, r in enumerate(regions):
    offset = (i - 1) * width
    ax.bar(x + offset, net_by_region[r], width, label=r, color=colors[r])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"{t:+.1f}" for t in non_zero_tilts], rotation=90, fontsize=7)
ax.set_xlabel("tilt (deg)")
ax.set_ylabel(f"net cells (gained - lost) vs tilt={REFERENCE_TILT:.1f} deg")
ax.set_title(f"Sector 0 ({SCENARIO}): net coverage change by region, across the tilt sweep "
            f"(near<{near_dist_m:.0f}m, mid<{far_dist_m:.0f}m, far>={far_dist_m:.0f}m)")
ax.legend()
fig.tight_layout()
cal_tag = "empirical" if NEAR_DIST_OVERRIDE_M is not None else "isdscaled"
out_path = f"results/debug/phase2_radio_environment_validation/gained_lost_net_by_tilt_{SCENARIO}_m{ANTENNA_M}_{ANTENNA_WINDOW}_{cal_tag}.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
