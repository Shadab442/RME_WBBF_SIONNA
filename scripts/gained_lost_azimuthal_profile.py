"""Restrict to cells whose coverage status actually CHANGED (gained or
lost) relative to tilt=0 deg -- not all cells in a near/mid/far band,
which mixes in cells that are "still covered"/"still uncovered" either
way and don't care about tilt at all. For just the gained/lost cells,
split by region (near/mid/far, current distance cuts) AND by
gained-vs-lost, then look at azimuthal position -- this isolates the
locations where tilt actually has the "highest level of change" in
coverage, per region, and shows where in the sector they sit. Requested
2026-10-05.
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
NUM_AZ_BINS = 12
HOTSPOT_NEAR_DIST_M = 900.0
HOTSPOT_FAR_DIST_M = 2300.0
REFERENCE_TILT = 0.0

sionna.phy.config.seed = dpa.DEBUG_SEED
engine = dpa.build_debug_engine()
topo = engine.topology

tilt_values = engine.downtilt_sweep_deg
tilt_lo, tilt_hi = float(tilt_values.min()), float(tilt_values.max())

site_xy = topo.bs_loc[0, TARGET_SECTOR, :2].cpu().numpy()
boresight_rad = float(topo.bs_orientations[0, TARGET_SECTOR, 0])
xy = engine.grid_points[TARGET_SECTOR]
d = xy - site_xy[None, :]
dist = np.linalg.norm(d, axis=1)
abs_angle = np.arctan2(d[:, 1], d[:, 0])
az_offset_deg = np.degrees(np.angle(np.exp(1j * (abs_angle - boresight_rad))))

region_of = np.full(dist.shape, "far", dtype=object)
region_of[dist < HOTSPOT_NEAR_DIST_M] = "near"
region_of[(dist >= HOTSPOT_NEAR_DIST_M) & (dist < HOTSPOT_FAR_DIST_M)] = "mid"
regions = ["near", "mid", "far"]

pert = dpa.compute_single_sector_tilt_perturbation(
    TARGET_SECTOR, np.array([REFERENCE_TILT, tilt_lo, tilt_hi]), other_sectors_tilt_deg=0.0, engine=engine)
num_grid_cells = engine.num_grid_cells
sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
cov_ref = pert["sinr_db"][0][sl] > engine.coverage_threshold_db
cov_lo = pert["sinr_db"][1][sl] > engine.coverage_threshold_db
cov_hi = pert["sinr_db"][2][sl] > engine.coverage_threshold_db

az_bins = np.linspace(-60, 60, NUM_AZ_BINS + 1)
az_bin_centers = 0.5 * (az_bins[:-1] + az_bins[1:])
az_bin_idx = np.digitize(az_offset_deg, az_bins) - 1

for label, cov_cand, tilt_cand in [("uptilt", cov_lo, tilt_lo), ("downtilt", cov_hi, tilt_hi)]:
    gained = (~cov_ref) & cov_cand
    lost = cov_ref & (~cov_cand)
    changed = gained | lost
    print(f"\n=== tilt {REFERENCE_TILT:.1f} -> {tilt_cand:.1f} deg ({label}) ===")
    print(f"total cells={len(cov_ref)}, gained={gained.sum()}, lost={lost.sum()}, "
         f"changed={changed.sum()} ({100*changed.sum()/len(cov_ref):.1f}% of all cells)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, region in zip(axes, regions):
        region_mask = region_of == region
        n_region = region_mask.sum()
        n_gained = (gained & region_mask).sum()
        n_lost = (lost & region_mask).sum()
        print(f"  {region:>5}: n_total={n_region}, gained={n_gained} ({100*n_gained/max(n_region,1):.1f}%), "
             f"lost={n_lost} ({100*n_lost/max(n_region,1):.1f}%)")

        gained_count_by_az = np.zeros(NUM_AZ_BINS, dtype=int)
        lost_count_by_az = np.zeros(NUM_AZ_BINS, dtype=int)
        total_count_by_az = np.zeros(NUM_AZ_BINS, dtype=int)
        for b in range(NUM_AZ_BINS):
            bin_mask = region_mask & (az_bin_idx == b)
            total_count_by_az[b] = bin_mask.sum()
            gained_count_by_az[b] = (bin_mask & gained).sum()
            lost_count_by_az[b] = (bin_mask & lost).sum()
        # fraction of THIS region's cells (at this azimuth) that gained/lost
        gained_frac = np.divide(gained_count_by_az, total_count_by_az,
                                out=np.zeros(NUM_AZ_BINS), where=total_count_by_az > 0)
        lost_frac = np.divide(lost_count_by_az, total_count_by_az,
                              out=np.zeros(NUM_AZ_BINS), where=total_count_by_az > 0)
        ax.bar(az_bin_centers - 1.5, gained_frac, width=3, color="tab:green", label="gained")
        ax.bar(az_bin_centers + 1.5, lost_frac, width=3, color="tab:red", label="lost")
        ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
        ax.set_xlabel("azimuth offset from boresight (deg)")
        ax.set_ylabel("fraction of region's cells at this azimuth")
        ax.set_title(f"{region} (n={n_region})")
        ax.legend(fontsize=8)
    fig.suptitle(f"Sector {TARGET_SECTOR}: gained/lost cells by azimuth, tilt {REFERENCE_TILT:.1f}->{tilt_cand:.1f} deg ({label})")
    fig.tight_layout()
    out_path = f"results/debug/phase2_radio_environment_validation/gained_lost_azimuthal_{label}.png"
    fig.savefig(out_path, dpi=150)
    print(f"  saved {out_path}")
