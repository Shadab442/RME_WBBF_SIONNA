"""Plots comparing all 7 DRL state_type x reward_type combinations against
each other and against the 4 non-DRL methods -- loads every
results/state_reward_comparison/<tag>/data.npz (no simulation/channel
computation here). One data point per TILT CONTROL INTERVAL, same
convention as scripts/plots/plot_main.py.

Oracle/Causal/Adaptive Legacy/No Tilt are read from COMBOS[0] (sector_hard)
only -- see run_state_reward_comparison.py's docstring for why they're
identical across every combo and only computed once.

Saves to results/state_reward_comparison/:
  1. coverage_vs_time.png          -- all 7 DRL variants + the 4 reference
                                       methods, last STEADY_STATE_EPISODES shaded.
  2. steady_state_coverage_bars.png -- one bar per combo + reference methods,
                                       steady-state mean coverage.
  3. drl_reward_vs_episode.png      -- all 7 variants' mean per-sector reward
                                       (NOT directly comparable across
                                       reward_type -- different reward scales
                                       -- shown for trend/convergence shape only).
  4. drl_overshoot_vs_episode.png   -- all 7 variants' mean per-sector overshoot.

Run: python scripts/plots/plot_state_reward_comparison.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_reward_comparison")

MOVING_AVERAGE_WINDOW = 11

# (tag, display label, color) -- sector_hard is COMBOS[0]: the one run with
# optimization_enabled=1, so it's also the source of Oracle/Causal/Adaptive
# Legacy/No Tilt reference data below.
COMBOS = [
    ("sector_hard", "DRL: sector-level / hard", "tab:green"),
    ("sector_hard_network", "DRL: sector-level / hard-network", "yellowgreen"),
    ("network_hard", "DRL: network-level / hard", "tab:purple"),
    ("network_hard_network", "DRL: network-level / hard-network", "mediumpurple"),
    ("network_sector_hard", "DRL: network-sector-level / hard", "tab:pink"),
    ("network_ue_hard", "DRL: network-UE-level / hard", "tab:cyan"),
    ("predicted_radio_map_hard", "DRL: predicted-radio-map / hard", "tab:brown"),
]

DATA = {tag: np.load(os.path.join(OUT_DIR, tag, "data.npz")) for tag, _, _ in COMBOS}
REFERENCE = DATA[COMBOS[0][0]]  # sector_hard -- the only run with optimization_enabled=1

COVERAGE_THRESHOLD_DB = float(REFERENCE["coverage_threshold_db"])
NUM_TILT_CONTROL_INTERVALS = int(REFERENCE["num_tilt_control_intervals"])
TILT_CONTROL_INTERVAL_S = float(REFERENCE["tilt_control_interval_s"])
STEADY_STATE_EPISODES = int(REFERENCE["steady_state_episodes"])
MOBILITY_MODEL = str(REFERENCE["mobility_model"])

REFERENCE_METHODS = {
    "dynamic_local_oracle": ("Dynamic Local Oracle", REFERENCE["coverage_dynamic_local_oracle"], "tab:blue"),
    "dynamic_local_causal": ("Dynamic Local Causal", REFERENCE["coverage_dynamic_local_causal"], "tab:orange"),
    "adaptive_legacy": ("Adaptive Legacy", REFERENCE["coverage_adaptive_legacy"], "tab:red"),
    "no_tilt": ("No Tilt", REFERENCE["coverage_no_tilt"], "black"),
}


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


time_s = np.arange(NUM_TILT_CONTROL_INTERVALS) * TILT_CONTROL_INTERVAL_S
steady = slice(-STEADY_STATE_EPISODES, None)
steady_state_start_s = max(0, NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES) * TILT_CONTROL_INTERVAL_S

# ----------------------------- coverage vs time -----------------------------
fig, ax = plt.subplots(figsize=(12, 7))
for key, (label, series, color) in REFERENCE_METHODS.items():
    ax.plot(time_s, moving_average(series, MOVING_AVERAGE_WINDOW), label=label, color=color,
           linestyle="--", linewidth=1.3, alpha=0.8)
for tag, label, color in COMBOS:
    series = DATA[tag]["coverage_drl"]
    ax.plot(time_s, moving_average(series, MOVING_AVERAGE_WINDOW), label=label, color=color, linewidth=1.6)
ax.axvspan(steady_state_start_s, time_s[-1], color="gray", alpha=0.12,
          label=f"steady state (last {STEADY_STATE_EPISODES})")
ax.set_xlabel("Elapsed time (s)")
ax.set_ylabel(f"Coverage (SINR > {COVERAGE_THRESHOLD_DB:g} dB)")
ax.set_title(f"DRL state/reward taxonomy comparison under {MOBILITY_MODEL} mobility\n"
            f"({MOVING_AVERAGE_WINDOW}-interval moving average, {TILT_CONTROL_INTERVAL_S:g} s/interval)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
path = os.path.join(OUT_DIR, "coverage_vs_time.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ------------------------- steady-state coverage bars ------------------------
bar_labels, bar_values, bar_colors = [], [], []
for key, (label, series, color) in REFERENCE_METHODS.items():
    bar_labels.append(label)
    bar_values.append(series[steady].mean())
    bar_colors.append(color)
for tag, label, color in COMBOS:
    bar_labels.append(label)
    bar_values.append(DATA[tag]["coverage_drl"][steady].mean())
    bar_colors.append(color)

order = np.argsort(bar_values)[::-1]
fig, ax = plt.subplots(figsize=(10, 7))
y_pos = np.arange(len(order))
ax.barh(y_pos, [bar_values[i] for i in order], color=[bar_colors[i] for i in order])
ax.set_yticks(y_pos)
ax.set_yticklabels([bar_labels[i] for i in order], fontsize=9)
ax.invert_yaxis()
for i, idx in enumerate(order):
    ax.text(bar_values[idx] + 0.0005, i, f"{bar_values[idx]:.4f}", va="center", fontsize=8)
ax.set_xlabel(f"Steady-state mean coverage (last {STEADY_STATE_EPISODES} intervals)")
ax.set_title(f"Steady-state coverage: all methods + all 7 DRL state/reward combinations\n"
            f"under {MOBILITY_MODEL} mobility")
ax.grid(True, axis="x", alpha=0.3)
ax.set_xlim(min(bar_values) - 0.005, max(bar_values) + 0.01)
path = os.path.join(OUT_DIR, "steady_state_coverage_bars.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

print("Steady-state mean coverage:")
for label, value in sorted(zip(bar_labels, bar_values), key=lambda item: -item[1]):
    print(f"  {label:38s} {value:.4f}")

# ---------------------------- DRL reward vs episode --------------------------
# NOT directly comparable across reward_type (different scales/populations --
# hard_network's reward is the same scalar for every sector, hard's varies
# per sector) -- shown for convergence SHAPE, not for picking a winner by eye.
episode = np.arange(NUM_TILT_CONTROL_INTERVALS)
fig, ax = plt.subplots(figsize=(11, 6))
for tag, label, color in COMBOS:
    mean_reward = np.nanmean(DATA[tag]["drl_reward_history"], axis=1)
    ax.plot(episode, moving_average(mean_reward, MOVING_AVERAGE_WINDOW), label=label, color=color, linewidth=1.4)
ax.axvspan(NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES, NUM_TILT_CONTROL_INTERVALS - 1,
          color="gray", alpha=0.12, label=f"steady state (last {STEADY_STATE_EPISODES})")
ax.set_xlabel("Episode (= tilt control interval)")
ax.set_ylabel("Mean per-sector reward (moving average)")
ax.set_title("DRL reward vs. episode, all 7 state/reward combinations\n"
            "(not directly comparable across reward_type -- different scales -- trend shape only)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
path = os.path.join(OUT_DIR, "drl_reward_vs_episode.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# --------------------------- DRL overshoot vs episode ------------------------
fig, ax = plt.subplots(figsize=(11, 6))
for tag, label, color in COMBOS:
    mean_overshoot = np.nanmean(DATA[tag]["drl_overshoot_history"], axis=1)
    ax.plot(episode, moving_average(mean_overshoot, MOVING_AVERAGE_WINDOW), label=label, color=color, linewidth=1.4)
ax.axvspan(NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES, NUM_TILT_CONTROL_INTERVALS - 1,
          color="gray", alpha=0.12, label=f"steady state (last {STEADY_STATE_EPISODES})")
ax.set_xlabel("Episode (= tilt control interval)")
ax.set_ylabel("Mean per-sector overshoot (moving average)")
ax.set_title("DRL overshoot vs. episode, all 7 state/reward combinations")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
path = os.path.join(OUT_DIR, "drl_overshoot_vs_episode.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")
