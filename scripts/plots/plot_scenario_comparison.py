"""Coverage vs. time, 4 algorithms (DQN-WESN 100% anchor, DQN-WESN 20%
anchor, Adaptive Legacy, Random), one figure per scenario (RMa, UMa) --
requested 2026-10-05/06 following the RMa/UMa tilt-tradeoff investigation
(UMa's own near/mid hotspot cut empirically recalibrated to 174.2m,
downtilt sweep -2..+2/0.2 deg vs RMa's -1..+1/0.1 deg -- see main.py's
--scenario flag and _SCENARIO_OVERRIDES). Same moving-average/color/
linestyle convention as plot_anchor_fraction_comparison.py.

Run: python scripts/plots/plot_scenario_comparison.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results")

# scenario -> (label, results subdir, data.npz column, color, linestyle)
SCENARIO_RUNS = {
    "rma": [
        ("DQN-WESN 100% anchor", "main_rma_locrep1", "coverage_drl", "tab:green", "-"),
        ("DQN-WESN 20% anchor", "main_rma_locrep0.2", "coverage_drl", "tab:green", "--"),
        ("Adaptive Legacy", "main_rma_locrep1", "coverage_adaptive_legacy", "tab:orange", "-"),
        ("Random", "main_rma_random", "coverage_drl", "tab:gray", "-"),
    ],
    "uma": [
        ("DQN-WESN 100% anchor", "main_uma_locrep1", "coverage_drl", "tab:green", "-"),
        ("DQN-WESN 20% anchor", "main_uma_locrep0.2", "coverage_drl", "tab:green", "--"),
        ("Adaptive Legacy", "main_uma_locrep1", "coverage_adaptive_legacy", "tab:orange", "-"),
        ("Random", "main_uma_random", "coverage_drl", "tab:gray", "-"),
    ],
}

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


for scenario, runs in SCENARIO_RUNS.items():
    data_by_run = [np.load(os.path.join(RESULTS_ROOT, subdir, "data.npz")) for _, subdir, _, _, _ in runs]
    num_intervals = int(data_by_run[0]["num_tilt_control_intervals"])
    threshold_db = float(data_by_run[0]["coverage_threshold_db"])
    series_list = [(label, data[column], color, ls)
                   for data, (label, _, column, color, ls) in zip(data_by_run, runs)]
    interval_idx = np.arange(num_intervals)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    for label, series, color, ls in series_list:
        smoothed = moving_average(series, MOVING_AVERAGE_WINDOW)
        ax.plot(interval_idx, smoothed, label=label, color=color, linestyle=ls,
               linewidth=2.2 if ls == "-" else 1.8)
    ax.set_xlim(interval_idx[0], interval_idx[-1])
    ax.set_xlabel("Tilt control interval")
    ax.set_ylabel(f"Coverage (SINR > {threshold_db:g} dB)")
    ax.set_title(f"{scenario.upper()}: network coverage vs time -- 4-algorithm comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    out_path = os.path.join(RESULTS_ROOT, f"{scenario}_coverage_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Mean coverage ({scenario}):")
    for label, series, _, _ in series_list:
        print(f"  {label:24s} full={series.mean():.4f} steady={series[-50:].mean():.4f}")
