"""Factorial design: sector 0 plus its two REAL geometric front-neighbors
(topo.neighbor_ids[0] == [8, 10], confirmed directly from CellularTopology
-- sector 8 sits at site (0,5000) yaw=300 deg (facing back toward sector
0's site), sector 10 sits at site (4330,2500) yaw=180 deg (also facing
back toward sector 0's site); sector 0 itself is at the center site,
yaw=60 deg, pointing between them). For each of the 3x3=9 (sector8_tilt,
sector10_tilt) combinations in {-2,0,2} deg, sweeps sector 0's OWN tilt
across {-2,0,2} deg and plots its own coverage-vs-distance curve (every
other sector held at 0 deg) -- one 3x3 grid of 9 panels, 3 curves each.

2026-10-07 correction: an earlier version of this script used [2, 8] --
sector 2 is a SAME-SITE SIBLING (co-located with sector 0, different
azimuth), not a real front-neighbor by the topology's own adjacency rule.
That pair was inherited unquestioningly from an earlier RMa-specific
script (gained_lost_fine_grid_two_neighbors.py) that picked its two
sectors by measured dB-drop ranking in that specific scenario, not from
neighbor_ids. Re-run with the CORRECT pair after the user caught the
mismatch by checking neighbor_ids directly.

Defaults to the DETERMINISTIC channel (forced LOS, no shadow fading) --
same reasoning as radial_multi_tilt_plot_scenario.py's own deterministic
mode: removes shadow-fading noise so the real underlying structure isn't
compressed/masked by per-draw randomness. Pass "stochastic" as a 3rd arg
to use config.yaml's own real per-link LOS/NLOS + shadow fading instead.

Run: python scripts/radial_two_neighbor_factorial_scenario.py uma deterministic
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
CHANNEL_MODE = sys.argv[2] if len(sys.argv) > 2 else "deterministic"
assert CHANNEL_MODE in ("stochastic", "deterministic", "los_with_shadowing")

TARGET_SECTOR = 0
NEIGHBOR_SECTORS = [8, 10]  # real topo.neighbor_ids[0], confirmed 2026-10-07
OWN_TILTS = [-2.0, 0.0, 2.0]
NEIGHBOR_TILTS = [-2.0, 0.0, 2.0]
SEEDS = [1] if CHANNEL_MODE == "deterministic" else [1, 2, 3]
RMA_ISD_M, RMA_NEAR_DIST_M, RMA_FAR_DIST_M = 5000.0, 900.0, 2300.0
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
    cfg["downtilt"]["downtilt_sweep_start_deg"] = -3
    cfg["downtilt"]["downtilt_sweep_end_deg"] = 3
    cfg["downtilt"]["tilt_step_deg"] = 1
    if CHANNEL_MODE == "deterministic":
        cfg["channel"]["force_los"] = True
        cfg["channel"]["disable_shadow_fading"] = True
    elif CHANNEL_MODE == "los_with_shadowing":
        cfg["channel"]["force_los"] = True
        cfg["channel"]["disable_shadow_fading"] = False
    return cfg
dpa.load_config = _scaled_load_config

sionna.phy.config.seed = SEEDS[0]
_probe_engine = dpa.build_debug_engine()
cell_radius_m = float(_probe_engine.topology.grid.cell_radius)
print(f"scenario={SCENARIO} channel={CHANNEL_MODE} cell_radius={cell_radius_m:.1f}m "
     f"target={TARGET_SECTOR} neighbors={NEIGHBOR_SECTORS}")

NUM_BINS = 20
bins = np.linspace(0, cell_radius_m, NUM_BINS + 1)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
colors = {-2.0: "tab:blue", 0.0: "tab:gray", 2.0: "tab:red"}

# result[tilt_sector8][tilt_sector10][own_tilt] -> [NUM_BINS] mean coverage fraction
result = {n8: {n10: {t: [] for t in OWN_TILTS} for n10 in NEIGHBOR_TILTS} for n8 in NEIGHBOR_TILTS}

for seed in SEEDS:
    sionna.phy.config.seed = seed
    engine = dpa.build_debug_engine()
    topo = engine.topology
    num_bs = engine.num_bs
    xy = engine.grid_points[TARGET_SECTOR]
    site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()
    dist = np.linalg.norm(xy - site_xy[None, :], axis=1)
    idx = np.digitize(dist, bins) - 1

    import torch
    # Chunked (batch_size=400) -- the channel model's own LSP spatial-
    # correlation matrix is O(N^2) in the virtual-UE count (same reasoning
    # as compute_hotspot_ground_truth's own chunking); 3364 points at once
    # OOM'd on this shared GPU. Only sector 0's OWN grid points are used as
    # virtual UEs (not all 21 sectors') -- resolve_power_at_tilt still sees
    # every TX sector's own contribution at each point regardless.
    batch_size = 400
    num_pts = xy.shape[0]
    cov_accum = {n8: {n10: {t: np.full(num_pts, False) for t in OWN_TILTS} for n10 in NEIGHBOR_TILTS}
                for n8 in NEIGHBOR_TILTS}
    with torch.no_grad():
        for start in range(0, num_pts, batch_size):
            end = min(start + batch_size, num_pts)
            chunk_state = dpa._virtual_grid_state(engine, xy=xy[start:end])
            for n8 in NEIGHBOR_TILTS:
                for n10 in NEIGHBOR_TILTS:
                    for own_t in OWN_TILTS:
                        tilt_vec = np.zeros(num_bs)
                        tilt_vec[TARGET_SECTOR] = own_t
                        tilt_vec[NEIGHBOR_SECTORS[0]] = n8
                        tilt_vec[NEIGHBOR_SECTORS[1]] = n10
                        power = engine.kpi_manager.resolve_power_at_tilt(chunk_state, tilt_vec)[0]
                        kpis = engine.kpi_manager.compute_ue_kpis(power[None, :, :])
                        cov_accum[n8][n10][own_t][start:end] = kpis["sinr_db"][0] > engine.coverage_threshold_db
            del chunk_state
            torch.cuda.empty_cache()

    for n8 in NEIGHBOR_TILTS:
        for n10 in NEIGHBOR_TILTS:
            for own_t in OWN_TILTS:
                cov = cov_accum[n8][n10][own_t]
                cov_by_bin = np.full(NUM_BINS, np.nan)
                for b in range(NUM_BINS):
                    mask = idx == b
                    if mask.sum() > 0:
                        cov_by_bin[b] = cov[mask].mean()
                result[n8][n10][own_t].append(cov_by_bin)

print(f"\nFar-region (>={near_dist_m:.0f}m) mean coverage, by (sector8, sector10) neighbor tilt, per own tilt:")
far_mask = bin_centers >= near_dist_m
print(f"{'sector8':>8}{'sector10':>9}" + "".join(f"own={t:+.0f}".rjust(10) for t in OWN_TILTS))
for n8 in NEIGHBOR_TILTS:
    for n10 in NEIGHBOR_TILTS:
        row = []
        for own_t in OWN_TILTS:
            mean_cov = np.nanmean(np.array(result[n8][n10][own_t]), axis=0)
            row.append(np.nanmean(mean_cov[far_mask]))
        print(f"{n8:8.0f}{n10:9.0f}" + "".join(f"{v:10.3f}" for v in row))
print("\nSpread ACROSS the 9 neighbor combos, for each own tilt (shows how much neighbor tilt alone moves far coverage):")
for own_t in OWN_TILTS:
    vals = [np.nanmean(np.nanmean(np.array(result[n8][n10][own_t]), axis=0)[far_mask])
           for n8 in NEIGHBOR_TILTS for n10 in NEIGHBOR_TILTS]
    print(f"  own={own_t:+.0f}: min={min(vals):.3f} max={max(vals):.3f} range={max(vals)-min(vals):.3f}")
print("\nSpread from sector8 ALONE (holding sector10 fixed) vs sector10 ALONE (holding sector8 fixed):")
for own_t in OWN_TILTS:
    for fixed_n10 in NEIGHBOR_TILTS:
        vals = [np.nanmean(np.nanmean(np.array(result[n8][fixed_n10][own_t]), axis=0)[far_mask]) for n8 in NEIGHBOR_TILTS]
        print(f"  own={own_t:+.0f} sector10={fixed_n10:+.0f} fixed: sector8 range={max(vals)-min(vals):.3f}")
    for fixed_n8 in NEIGHBOR_TILTS:
        vals = [np.nanmean(np.nanmean(np.array(result[fixed_n8][n10][own_t]), axis=0)[far_mask]) for n10 in NEIGHBOR_TILTS]
        print(f"  own={own_t:+.0f} sector8={fixed_n8:+.0f} fixed: sector10 range={max(vals)-min(vals):.3f}")

fig, axes = plt.subplots(3, 3, figsize=(15, 13), sharex=True, sharey=True)
for i, n8 in enumerate(NEIGHBOR_TILTS):
    for j, n10 in enumerate(NEIGHBOR_TILTS):
        ax = axes[i, j]
        for own_t in OWN_TILTS:
            mean_cov = np.nanmean(np.array(result[n8][n10][own_t]), axis=0)
            ax.plot(bin_centers, mean_cov, marker="o", markersize=3, label=f"own={own_t:+.0f}",
                   color=colors[own_t])
        ax.axvline(near_dist_m, color="gray", linestyle=":", alpha=0.4)
        ax.set_title(f"sector8={n8:+.0f} sector10={n10:+.0f}", fontsize=10)
        if i == 2:
            ax.set_xlabel("distance (m)")
        if j == 0:
            ax.set_ylabel("coverage fraction")
axes[0, 0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"Sector 0 coverage vs distance, own tilt swept {{-2,0,2}} deg, "
            f"for all 9 (sector8, sector10) neighbor-tilt combos (REAL neighbor_ids[0])\n"
            f"{SCENARIO.upper()}, {CHANNEL_MODE} channel, {len(SEEDS)} seed(s)")
fig.tight_layout()
out_path = f"results/debug/phase2_radio_environment_validation/radial_two_neighbor_factorial_{SCENARIO}_{CHANNEL_MODE}.png"
fig.savefig(out_path, dpi=150)
print("saved", out_path)
