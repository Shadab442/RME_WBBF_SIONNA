"""Coverage vs. time across 4 DRL algorithm variants (PPO-WESN, DQN-WESN,
DQN-MLP, PPO-MLP) x 2 location_reporting_fraction conditions (100% anchor --
every UE's location known, the codebase's long-standing default -- vs. 20%
anchor -- only 20% of UEs report location; the rest are excluded from
own-region state entirely, non_anchor_region_handling=exclude, no RME/
Kriging inference; reward and neighbor coarse state always count every UE
regardless of anchor status), plus Adaptive Legacy and Random as fixed
reference lines. All 9 runs share the same config_1 scenario (all-outdoor,
forced-LOS, no shadow fading) -- see project_indoor_outdoor_split_
investigation memory for why that scenario was chosen over the newer
indoor/outdoor one for this batch (2026-09-24 request, explicit "config 1").

Color = algorithm identity (fixed, never cycled); linestyle = anchor
fraction (solid=100%, dashed=20%) -- keeps the two dimensions visually
separable without doubling the color count. Adaptive Legacy/Random have
only one condition each, so solid only.

Run: python scripts/plots/plot_anchor_fraction_comparison.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
OUT_PATH = os.path.join(RESULTS_ROOT, "anchor_fraction_comparison_coverage_vs_time.png")

# (label, results subdir, data.npz column, color, linestyle).
RUNS = [
    ("DQN-WESN 100%", "main_wesn_async_alloutdoor_forcedlos", "coverage_drl", "tab:green", "-"),
    ("DQN-WESN 20%", "main_wesn_async_locrep0.2", "coverage_drl", "tab:green", "--"),
    ("PPO-WESN 100%", "main_ppo", "coverage_drl", "tab:red", "-"),
    ("PPO-WESN 20%", "main_ppo_wesn_async_locrep0.2", "coverage_drl", "tab:red", "--"),
    ("DQN-MLP 100%", "main_mlp_async_locrep1", "coverage_drl", "tab:blue", "-"),
    ("DQN-MLP 20%", "main_mlp_async_locrep0.2", "coverage_drl", "tab:blue", "--"),
    ("PPO-MLP 100%", "main_ppo_mlp_async_locrep1", "coverage_drl", "tab:purple", "-"),
    ("PPO-MLP 20%", "main_ppo_mlp_async_locrep0.2", "coverage_drl", "tab:purple", "--"),
    ("Adaptive Legacy", "main_wesn_async_alloutdoor_forcedlos", "coverage_adaptive_legacy", "tab:orange", "-"),
    ("Random", "main_random_async_opt", "coverage_drl", "tab:gray", "-"),
]

MOVING_AVERAGE_WINDOW = 25


def moving_average(x, window):
    """Centered moving average, edge-padded so the output stays the same
    length as the input rather than biasing smoothed edges toward zero."""
    if window <= 1:
        return x
    pad_before = window // 2
    pad_after = window - 1 - pad_before
    padded = np.pad(x, (pad_before, pad_after), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


data_by_run = [np.load(os.path.join(RESULTS_ROOT, subdir, "data.npz")) for _, subdir, _, _, _ in RUNS]

NUM_TILT_CONTROL_INTERVALS = int(data_by_run[0]["num_tilt_control_intervals"])
COVERAGE_THRESHOLD_DB = float(data_by_run[0]["coverage_threshold_db"])

SERIES = [(label, data[column], color, ls)
         for data, (label, _, column, color, ls) in zip(data_by_run, RUNS)]

interval_idx = np.arange(NUM_TILT_CONTROL_INTERVALS)

fig, ax = plt.subplots(figsize=(13, 7.5))
for label, series, color, ls in SERIES:
    smoothed = moving_average(series, MOVING_AVERAGE_WINDOW)
    ax.plot(interval_idx, smoothed, label=label, color=color, linestyle=ls,
           linewidth=2.2 if ls == "-" else 1.8)

ax.set_xlim(interval_idx[0], interval_idx[-1])
ax.set_xlabel("Tilt control interval")
ax.set_ylabel(f"Coverage (SINR > {COVERAGE_THRESHOLD_DB:g} dB)")
ax.set_title("Network coverage vs time -- anchor-UE fraction comparison\n"
            "(solid = 100% anchor, dashed = 20% anchor)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc="lower right", ncol=2)
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
print("Mean coverage (full run):")
for label, series, _, _ in SERIES:
    print(f"  {label:18s} full={series.mean():.4f} steady={series[-50:].mean():.4f}")
