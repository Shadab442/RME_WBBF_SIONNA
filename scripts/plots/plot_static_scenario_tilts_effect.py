"""Plots for test_static_scenario_tilts_effect.py -- loads
results/tests/static_scenario_tilts_effect/data.npz (scenario.png is generated
directly by the test script, not here).

Saves to results/tests/static_scenario_tilts_effect/:
  1. per_sector_tilt.png       -- Static Local's chosen tilt per sector vs.
                                   Static Global's single common tilt --
                                   the spatial-variability evidence.
  2. coverage_comparison.png   -- pooled coverage, Global vs. Local.
  3. per_sector_coverage.png   -- per-sector coverage, Global vs. Local.
  4. coordinate_ascent_trace.png -- pooled coverage vs. round (monotonic
                                   non-decrease check).
  5. sinr_cdf_global_vs_local.png -- pooled SINR CDF, with the coverage
                                   value marked directly on each curve at
                                   the threshold.

Run: python scripts/plots/plot_static_scenario_tilts_effect.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "static_scenario_tilts_effect")

data = np.load(os.path.join(OUT_DIR, "data.npz"))
DOWNTILT_SWEEP_DEG = data["downtilt_sweep_deg"].tolist()
COVERAGE_THRESHOLD_DB = float(data["coverage_threshold_db"])
NUM_RINGS = int(data["num_rings"])
NUM_BS = int(data["num_bs"])
NUM_UT = int(data["num_ut"])
STATIC_GLOBAL_TILT_DEG = float(data["static_global_tilt_deg"])
GLOBAL_COVERAGE = float(data["global_coverage"])
LOCAL_TILT_DEG_PER_SECTOR = data["local_tilt_deg_per_sector"]
LOCAL_COVERAGE = float(data["local_coverage"])
COVERAGE_TRACE = data["coverage_trace"]
NUM_ROUNDS = int(data["num_rounds"])
SINR_DB_GLOBAL = data["sinr_db_global"]
SINR_DB_LOCAL = data["sinr_db_local"]
PER_SECTOR_COVERAGE_GLOBAL = data["per_sector_coverage_global"]
PER_SECTOR_COVERAGE_LOCAL = data["per_sector_coverage_local"]


def ecdf(x: np.ndarray):
    x_sorted = np.sort(x)
    return x_sorted, np.arange(1, len(x_sorted) + 1) / len(x_sorted)


# ---------------------------- per-sector tilt ----------------------------
fig, ax = plt.subplots(figsize=(9, 5))
sector_idx = np.arange(NUM_BS)
ax.bar(sector_idx, LOCAL_TILT_DEG_PER_SECTOR, color="tab:blue", label="Static Local")
ax.axhline(STATIC_GLOBAL_TILT_DEG, color="black", linestyle="--",
          label=f"Static Global ({STATIC_GLOBAL_TILT_DEG:g}°)")
ax.set_xlabel("Sector index")
ax.set_ylabel("Chosen downtilt (deg)")
ax.set_title(f"Coverage-maximizing tilt per sector ({NUM_BS} sectors, {NUM_RINGS} ring(s))\n"
            f"{len(set(LOCAL_TILT_DEG_PER_SECTOR.tolist()))} distinct tilt(s) in use under Local")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
path = os.path.join(OUT_DIR, "per_sector_tilt.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ------------------------- coverage comparison ----------------------------
fig, ax = plt.subplots(figsize=(5, 5.5))
bars = ax.bar(["Static Global", "Static Local"], [GLOBAL_COVERAGE, LOCAL_COVERAGE],
             color=["tab:gray", "tab:blue"])
for bar, value in zip(bars, [GLOBAL_COVERAGE, LOCAL_COVERAGE]):
    ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
               textcoords="offset points", xytext=(0, 6), ha="center", fontweight="bold")
ax.set_ylabel(f"Coverage (fraction of UEs, SINR > {COVERAGE_THRESHOLD_DB:g} dB)")
ax.set_ylim(0, 1.05)
ax.set_title(f"Pooled coverage: gap = {LOCAL_COVERAGE - GLOBAL_COVERAGE:+.4f}")
ax.grid(True, alpha=0.3, axis="y")
path = os.path.join(OUT_DIR, "coverage_comparison.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ----------------------- per-sector coverage ------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
width = 0.4
ax.bar(sector_idx - width / 2, PER_SECTOR_COVERAGE_GLOBAL, width, label="Static Global", color="tab:gray")
ax.bar(sector_idx + width / 2, PER_SECTOR_COVERAGE_LOCAL, width, label="Static Local", color="tab:blue")
ax.set_xlabel("Sector index")
ax.set_ylabel("Coverage (own attached UEs)")
ax.set_title("Per-sector coverage: Global vs. Local")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
path = os.path.join(OUT_DIR, "per_sector_coverage.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ------------------- coordinate ascent convergence ------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(np.arange(len(COVERAGE_TRACE)), COVERAGE_TRACE, "o-", color="tab:blue")
ax.set_xlabel("Round (0 = Static Global start)")
ax.set_ylabel("Pooled coverage")
ax.set_title(f"Coordinate-ascent convergence ({NUM_ROUNDS} round(s) to converge)")
ax.grid(True, alpha=0.3)
path = os.path.join(OUT_DIR, "coordinate_ascent_trace.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# --------------------------- SINR CDF -------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for sinr_db, label, coverage, color in (
    (SINR_DB_GLOBAL, "Static Global", GLOBAL_COVERAGE, "tab:gray"),
    (SINR_DB_LOCAL, "Static Local", LOCAL_COVERAGE, "tab:blue"),
):
    x, y = ecdf(sinr_db)
    ax.plot(x, y, label=label, color=color)
    # Mark the verified coverage value directly on the curve, at the
    # threshold crossing (1 - coverage is the CDF value there).
    ax.annotate(
        f"coverage={coverage:.3f}",
        xy=(COVERAGE_THRESHOLD_DB, 1.0 - coverage),
        xytext=(COVERAGE_THRESHOLD_DB + 3, 1.0 - coverage - 0.12),
        color=color, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color),
    )
ax.axvline(COVERAGE_THRESHOLD_DB, color="black", linestyle=":", alpha=0.6,
          label=f"threshold ({COVERAGE_THRESHOLD_DB:g} dB)")
ax.set_xlabel("SINR (dB)")
ax.set_ylabel("CDF")
ax.set_title("Pooled per-UE SINR CDF: Global vs. Local")
ax.grid(True, alpha=0.3)
ax.legend()
path = os.path.join(OUT_DIR, "sinr_cdf_global_vs_local.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")
