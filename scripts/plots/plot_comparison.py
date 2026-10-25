"""DQN vs. PPO vs. Adaptive Legacy vs. random-policy coverage comparison --
unlike plot_main.py (one run's own data.npz, picking among ITS columns via
SHOW_METHODS), this loads separate runs' own data.npz files and plots each
one's own named column as a distinct series. Legend used to say just "DRL"
when DQN was the only DRL-family method shown (2026-09-22); now that PPO
(drl/ppo.py, 2026-09-23) is a second DRL-family method, both get their own
label -- see module docstring history for the earlier 3/5-way version of
this script (superseded 2026-09-22 once CP-SAT/Oracle and the global-reward
comparison were dropped from current interest).

DQN and Adaptive Legacy both read from main_wesn_async (the run made after
the 2026-09-22 AdaptiveLegacyTiltController threshold/tie-break fix) rather
than the older main_wesn_async_opt -- confirmed identical coverage_drl
between the two dirs (same deterministic env seed), so this doesn't change
the DQN series, only which dir Adaptive Legacy's real (post-fix) numbers
come from. PPO reads from main_ppo (--model ppo, same scenario/config,
2026-09-23) -- first untuned run, beats Adaptive Legacy/Random, behind DQN
(which has had prior tuning passes this session that PPO hasn't yet).

Plain moving-average-smoothed line plot -- no steady-state shading/markers
(dropped 2026-09-22 per request; this is just the raw comparison curve).

Run: python scripts/plots/plot_comparison.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

RESULTS_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results")
# config_2 (2026-09-24): DQN vs. Adaptive Legacy ONLY, now under the new
# indoor/outdoor scenario (outdoor_ue_ratio=0.2, force_los=null, shadow
# fading off) -- main_wesn_async was overwritten by that run, so this is a
# DIFFERENT propagation scenario than config_1's (all-outdoor/forced-LoS),
# not directly comparable to it. No Random/PPO/MLP-DQN this round.
OUT_PATH = os.path.join(RESULTS_ROOT, "config_2_coverage_vs_time.png")

# (label, results subdir, data.npz column, color).
RUNS = [
    ("DQN", "main_wesn_async", "coverage_drl", "tab:green"),
    ("Adaptive Legacy", "main_wesn_async", "coverage_adaptive_legacy", "tab:orange"),
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


data_by_run = [np.load(os.path.join(RESULTS_ROOT, subdir, "data.npz")) for _, subdir, _, _ in RUNS]

NUM_TILT_CONTROL_INTERVALS = int(data_by_run[0]["num_tilt_control_intervals"])
COVERAGE_THRESHOLD_DB = float(data_by_run[0]["coverage_threshold_db"])

SERIES = [(label, data[column], color) for data, (label, _, column, color) in zip(data_by_run, RUNS)]

interval_idx = np.arange(NUM_TILT_CONTROL_INTERVALS)

fig, ax = plt.subplots(figsize=(12, 6.5))
for label, series, color in SERIES:
    smoothed = moving_average(series, MOVING_AVERAGE_WINDOW)
    ax.plot(interval_idx, smoothed, label=label, color=color)

ax.set_xlim(interval_idx[0], interval_idx[-1])
ax.set_xlabel("Tilt control interval")
ax.set_ylabel(f"Coverage (SINR > {COVERAGE_THRESHOLD_DB:g} dB)")
ax.set_title("Network coverage vs time")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
print("Mean coverage (full run): " + ", ".join(f"{label}={series.mean():.4f}" for label, series, _ in SERIES))
