"""Plots for test_tilt_window_calibration.py -- loads
results/tests/tilt_window_calibration/data.npz (no simulation here).

Saves to results/tests/tilt_window_calibration/, per mobility speed setting:
  1. <setting>_coverage_vs_window.png -- mean coverage (and gap vs. the W=1
     per-slot oracle) as a function of window size W -- the core result:
     where's the knee, i.e. the sweet spot.
  2. <setting>_oracle_tilt_heatmap.png -- the W=1 oracle's per-sector tilt
     over time, for reference (shared across every W's diff plot below).
  3. <setting>_W<w>_tilt_diff_heatmap.png -- per sector x slot, colored by
     (windowed(W) tilt - oracle tilt) -- white where the single windowed
     assignment happens to match what the per-slot oracle would have
     picked; saturated where it doesn't. One per candidate W > 1 (W=1 is
     identical to the oracle by construction, nothing to show).
  4. <setting>_fraction_differing_vs_window.png -- one compact line per W:
     fraction of sectors where windowed(W) disagrees with the oracle, vs.
     time -- the same information as the diff heatmaps, compressed.

Run: python scripts/plots/plot_tilt_window_calibration.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "tilt_window_calibration")

data = np.load(os.path.join(OUT_DIR, "data.npz"))
CANDIDATE_WINDOW_SLOTS = data["candidate_window_slots"].tolist()
MEASUREMENT_INTERVAL_S = float(data["measurement_interval_s"])
NUM_SLOTS = int(data["num_slots"])
NUM_BS = int(data["num_bs"])
DOWNTILT_SWEEP_DEG = data["downtilt_sweep_deg"].tolist()
MOBILITY_SPEED_NAMES = data["mobility_speed_names"].tolist()
MOBILITY_SPEED_MIN = data["mobility_speed_min"].tolist()
MOBILITY_SPEED_MAX = data["mobility_speed_max"].tolist()

time_s = np.arange(NUM_SLOTS) * MEASUREMENT_INTERVAL_S

for name, min_speed, max_speed in zip(MOBILITY_SPEED_NAMES, MOBILITY_SPEED_MIN, MOBILITY_SPEED_MAX):
    oracle_coverage = data[f"oracle_coverage_{name}"]
    oracle_tilt_deg_history = data[f"oracle_tilt_deg_history_{name}"]
    print(f"\n=== {name} ({min_speed:g}-{max_speed:g} m/s) ===")

    # ------------------------- coverage vs. window size ---------------------
    mean_coverage = []
    mean_gap = []
    for w in CANDIDATE_WINDOW_SLOTS:
        windowed_coverage = data[f"windowed_coverage_{name}_{w}"]
        mean_coverage.append(float(windowed_coverage.mean()))
        mean_gap.append(float((oracle_coverage - windowed_coverage).mean()))
        print(f"  W={w:4d} slots ({w * MEASUREMENT_INTERVAL_S:g} s): "
             f"mean_coverage={mean_coverage[-1]:.4f}, mean_gap={mean_gap[-1]:+.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(CANDIDATE_WINDOW_SLOTS, mean_coverage, "o-", color="tab:blue")
    ax1.set_xscale("log")
    ax1.set_xlabel("Window size W (slots)")
    ax1.set_ylabel("Mean coverage")
    ax1.set_title("Windowed non-causal-optimal coverage vs. window size")
    ax1.grid(True, alpha=0.3)

    ax2.plot(CANDIDATE_WINDOW_SLOTS, mean_gap, "o-", color="tab:red")
    ax2.set_xscale("log")
    ax2.set_xlabel("Window size W (slots)")
    ax2.set_ylabel("Mean gap (oracle - windowed)")
    ax2.set_title("Windowing loss vs. window size\n(should be ~0 at W=1 by construction)")
    ax2.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{name} mobility ({min_speed:g}-{max_speed:g} m/s)")
    path = os.path.join(OUT_DIR, f"{name}_coverage_vs_window.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # -------------------------- oracle tilt heatmap --------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        oracle_tilt_deg_history.T, aspect="auto", origin="lower",
        extent=[0, time_s[-1], -0.5, NUM_BS - 0.5], cmap="viridis",
        vmin=min(DOWNTILT_SWEEP_DEG), vmax=max(DOWNTILT_SWEEP_DEG),
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Per-slot oracle tilt (deg)")
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel("Sector index")
    ax.set_title(f"W=1 per-slot oracle tilt over time -- {name} mobility "
                f"({min_speed:g}-{max_speed:g} m/s)")
    path = os.path.join(OUT_DIR, f"{name}_oracle_tilt_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # -------------- per-window-size tilt diff heatmaps + fraction differing --
    fig_frac, ax_frac = plt.subplots(figsize=(9, 4.5))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(CANDIDATE_WINDOW_SLOTS)))
    for w, color in zip(CANDIDATE_WINDOW_SLOTS, colors):
        if w == 1:
            continue  # identical to the oracle by construction -- nothing to show
        windowed_tilt_deg_history = data[f"windowed_tilt_deg_history_{name}_{w}"]
        tilt_diff = windowed_tilt_deg_history - oracle_tilt_deg_history  # [num_slots, num_bs]
        diff_max = max(np.max(np.abs(tilt_diff)), 1e-6)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            tilt_diff.T, aspect="auto", origin="lower",
            extent=[0, time_s[-1], -0.5, NUM_BS - 0.5], cmap="RdBu_r",
            vmin=-diff_max, vmax=diff_max,
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(f"Windowed(W={w}) - oracle (deg)")
        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel("Sector index")
        ax.set_title(f"Windowed(W={w} slots, {w * MEASUREMENT_INTERVAL_S:g} s) tilt vs. per-slot oracle\n"
                    f"{name} mobility ({min_speed:g}-{max_speed:g} m/s) -- "
                    "white = matches oracle; red/blue = over/under-tilted")
        path = os.path.join(OUT_DIR, f"{name}_W{w}_tilt_diff_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

        fraction_differing = np.mean(tilt_diff != 0, axis=1)  # [num_slots]
        ax_frac.plot(time_s, fraction_differing, color=color, label=f"W={w}")

    ax_frac.set_xlabel("Elapsed time (s)")
    ax_frac.set_ylabel("Fraction of sectors differing from oracle")
    ax_frac.set_ylim(0, 1.0)
    ax_frac.set_title(f"Windowed tilt disagreement with the per-slot oracle -- {name} mobility "
                      f"({min_speed:g}-{max_speed:g} m/s)")
    ax_frac.grid(True, alpha=0.3)
    ax_frac.legend()
    path = os.path.join(OUT_DIR, f"{name}_fraction_differing_vs_window.png")
    fig_frac.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig_frac)
    print(f"Saved: {path}")
