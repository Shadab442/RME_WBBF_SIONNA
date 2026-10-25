"""For the KNOWN tilt-tradeoff distance bands (near <900m, mid 900-2300m --
_HOTSPOT_NEAR_DIST_M/_HOTSPOT_FAR_DIST_M), how does coverage vary by
AZIMUTHAL angle within the sector wedge, not just by radial distance?
The near/mid distance split is purely radial (tilt is elevation-only, no
azimuthal effect by itself) -- but SINR itself does depend on azimuth too
(closeness to a shared sector boundary changes interference), so the
tradeoff's own STRENGTH could plausibly vary across the wedge even within
one fixed distance band. Requested 2026-10-05.
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
az_offset_rad = np.angle(np.exp(1j * (abs_angle - boresight_rad)))  # wrap to [-pi, pi]
az_offset_deg = np.degrees(az_offset_rad)

near_mask = dist < HOTSPOT_NEAR_DIST_M
mid_mask = (dist >= HOTSPOT_NEAR_DIST_M) & (dist < HOTSPOT_FAR_DIST_M)
print(f"near band: n={near_mask.sum()}, az range=[{az_offset_deg[near_mask].min():.1f},{az_offset_deg[near_mask].max():.1f}]")
print(f"mid band:  n={mid_mask.sum()}, az range=[{az_offset_deg[mid_mask].min():.1f},{az_offset_deg[mid_mask].max():.1f}]")

pert = dpa.compute_single_sector_tilt_perturbation(
    TARGET_SECTOR, np.array([tilt_lo, tilt_hi]), other_sectors_tilt_deg=0.0, engine=engine)
num_grid_cells = engine.num_grid_cells
sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
cov_lo = pert["sinr_db"][0][sl] > engine.coverage_threshold_db
cov_hi = pert["sinr_db"][1][sl] > engine.coverage_threshold_db

az_bins = np.linspace(-60, 60, NUM_AZ_BINS + 1)
az_bin_centers = 0.5 * (az_bins[:-1] + az_bins[1:])
az_bin_idx = np.digitize(az_offset_deg, az_bins) - 1

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, band_mask, band_name in [(axes[0], near_mask, "near (<900m)"), (axes[1], mid_mask, "mid (900-2300m)")]:
    cov_lo_by_az = np.full(NUM_AZ_BINS, np.nan)
    cov_hi_by_az = np.full(NUM_AZ_BINS, np.nan)
    n_by_az = np.zeros(NUM_AZ_BINS, dtype=int)
    for b in range(NUM_AZ_BINS):
        bin_mask = band_mask & (az_bin_idx == b)
        n_by_az[b] = bin_mask.sum()
        if bin_mask.sum() > 0:
            cov_lo_by_az[b] = cov_lo[bin_mask].mean()
            cov_hi_by_az[b] = cov_hi[bin_mask].mean()
    ax.plot(az_bin_centers, cov_lo_by_az, "-o", label=f"tilt={tilt_lo:.1f} deg", color="tab:blue")
    ax.plot(az_bin_centers, cov_hi_by_az, "-s", label=f"tilt={tilt_hi:.1f} deg", color="tab:red")
    ax.axvline(0, color="gray", linestyle=":", alpha=0.6, label="boresight")
    ax.set_xlabel("azimuth offset from boresight (deg)")
    ax.set_ylabel("coverage fraction")
    ax.set_title(f"{band_name}")
    ax.legend(fontsize=8)
    print(f"\n{band_name}:")
    print(f"  {'az_deg':>8}{'n':>6}{'tilt_lo':>10}{'tilt_hi':>10}{'diff':>10}")
    for b in range(NUM_AZ_BINS):
        print(f"  {az_bin_centers[b]:8.1f}{n_by_az[b]:6d}{cov_lo_by_az[b]:10.3f}{cov_hi_by_az[b]:10.3f}"
             f"{cov_hi_by_az[b]-cov_lo_by_az[b]:10.3f}")

fig.suptitle(f"Sector {TARGET_SECTOR}: coverage vs azimuth within the known tradeoff distance bands "
            f"(tilt={tilt_lo:.1f} vs {tilt_hi:.1f} deg)")
fig.tight_layout()
out_path = "results/debug/phase2_radio_environment_validation/azimuthal_tradeoff_profile.png"
fig.savefig(out_path, dpi=150)
print("\nsaved", out_path)
