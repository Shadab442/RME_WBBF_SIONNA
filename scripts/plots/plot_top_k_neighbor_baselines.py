"""Five-curve comparison: DRL top_k_neighbor vs. Dynamic Local Oracle +
Dynamic Local Causal (coordinate-ascent search baselines) vs. Adaptive
Legacy + No Tilt -- loads results/state_reward_comparison/
{top_k_neighbor_hard,oracle_causal_hard}/data.npz (no simulation/channel
computation here). One data point per TILT CONTROL INTERVAL, same
convention as scripts/plots/plot_main.py.

Oracle/Causal/Adaptive Legacy/No Tilt come from oracle_causal_hard (the only
run with optimization_enabled=True); its own DRL curve is discarded in favor
of top_k_neighbor_hard's -- that one ran with the per-sector epsilon-decay
fix (drl/independent_dqn.py), oracle_causal_hard did not, and its DRL
state_type (top_k_neighbor) was a don't-care picked only to invoke the
search.

Saves to results/state_reward_comparison/:
  top_k_neighbor_coverage_vs_time.png -- all 5 curves, steady-state means marked.
  top_k_neighbor_reward_vs_episode.png -- DRL mean per-sector reward.
  top_k_neighbor_overshoot_vs_episode.png -- DRL mean per-sector overshoot.

Run: python scripts/plots/plot_top_k_neighbor_baselines.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "state_reward_comparison")

MOVING_AVERAGE_WINDOW = 11

DRL_TAG, DRL_LABEL, DRL_COLOR = "top_k_neighbor_hard", "DRL: top_k_neighbor", "tab:green"

DATA = {DRL_TAG: np.load(os.path.join(OUT_DIR, DRL_TAG, "data.npz"))}
DATA["oracle_causal_hard"] = np.load(os.path.join(OUT_DIR, "oracle_causal_hard", "data.npz"))
REFERENCE = DATA["oracle_causal_hard"]

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
steady_center_s = (steady_state_start_s + time_s[-1]) / 2.0
steady_center_idx = int(np.argmin(np.abs(time_s - steady_center_s)))

# ----------------------------- coverage vs time -----------------------------
fig, ax = plt.subplots(figsize=(12, 7))
for key, (label, series, color) in REFERENCE_METHODS.items():
    ax.plot(time_s, moving_average(series, MOVING_AVERAGE_WINDOW), label=label, color=color,
           linestyle="--", linewidth=1.3, alpha=0.8)
ax.plot(time_s, moving_average(DATA[DRL_TAG]["coverage_drl"], MOVING_AVERAGE_WINDOW),
       label=DRL_LABEL, color=DRL_COLOR, linewidth=1.8)
ax.axvspan(steady_state_start_s, time_s[-1], color="gray", alpha=0.12,
          label=f"steady state (last {STEADY_STATE_EPISODES})")

xlim_right = time_s[-1] + 0.15 * (time_s[-1] - time_s[0])
ax.set_xlim(time_s[0], xlim_right)
all_series = {**REFERENCE_METHODS, "drl": (DRL_LABEL, DATA[DRL_TAG]["coverage_drl"], DRL_COLOR)}
label_specs = sorted(((k, v[1][steady].mean()) for k, v in all_series.items()), key=lambda item: item[1])
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
min_gap = 0.045 * y_range
for i in range(1, len(label_specs)):
    key, y = label_specs[i]
    prev_key, prev_y = label_specs[i - 1]
    if y - prev_y < min_gap:
        label_specs[i] = (key, prev_y + min_gap)

for key, label_y in label_specs:
    label, series, color = all_series[key]
    smoothed = moving_average(series, MOVING_AVERAGE_WINDOW)
    steady_mean = series[steady].mean()
    marker_xy = (time_s[steady_center_idx], smoothed[steady_center_idx])
    ax.plot(*marker_xy, marker="o", color=color, markersize=5, zorder=5)
    ax.annotate(
        f"{steady_mean:.4f}",
        xy=marker_xy, xytext=(xlim_right, label_y),
        color=color, fontsize=9, fontweight="bold", va="center",
        arrowprops=dict(arrowstyle="-", color=color, alpha=0.6, shrinkA=0, shrinkB=4),
    )

ax.set_xlabel("Elapsed time (s)")
ax.set_ylabel(f"Coverage (SINR > {COVERAGE_THRESHOLD_DB:g} dB)")
ax.set_title(f"top_k_neighbor DRL vs. baselines under {MOBILITY_MODEL} mobility\n"
            f"({MOVING_AVERAGE_WINDOW}-interval moving average, {TILT_CONTROL_INTERVAL_S:g} s/interval)\n"
            f"labels: steady-state mean (last {STEADY_STATE_EPISODES} intervals)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
path = os.path.join(OUT_DIR, "top_k_neighbor_coverage_vs_time.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------- DRL reward vs episode --------------------------
episode = np.arange(NUM_TILT_CONTROL_INTERVALS)
fig, ax = plt.subplots(figsize=(10, 5.5))
mean_reward = np.nanmean(DATA[DRL_TAG]["drl_reward_history"], axis=1)
ax.plot(episode, mean_reward, color=DRL_COLOR, linewidth=0.8, alpha=0.35)
ax.plot(episode, moving_average(mean_reward, MOVING_AVERAGE_WINDOW), label=DRL_LABEL, color=DRL_COLOR, linewidth=1.8)
ax.axvspan(NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES, NUM_TILT_CONTROL_INTERVALS - 1,
          color="gray", alpha=0.12, label=f"steady state (last {STEADY_STATE_EPISODES})")
ax.set_xlabel("Episode (= tilt control interval)")
ax.set_ylabel("Mean per-sector reward")
ax.set_title("DRL reward vs. episode (per-sector epsilon-decay fix applied)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
path = os.path.join(OUT_DIR, "top_k_neighbor_reward_vs_episode.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# --------------------------- DRL overshoot vs episode ------------------------
fig, ax = plt.subplots(figsize=(10, 5.5))
mean_overshoot = np.nanmean(DATA[DRL_TAG]["drl_overshoot_history"], axis=1)
ax.plot(episode, mean_overshoot, color=DRL_COLOR, linewidth=0.8, alpha=0.35)
ax.plot(episode, moving_average(mean_overshoot, MOVING_AVERAGE_WINDOW), label=DRL_LABEL, color=DRL_COLOR, linewidth=1.8)
ax.axvspan(NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES, NUM_TILT_CONTROL_INTERVALS - 1,
          color="gray", alpha=0.12, label=f"steady state (last {STEADY_STATE_EPISODES})")
ax.set_xlabel("Episode (= tilt control interval)")
ax.set_ylabel("Mean per-sector overshoot")
ax.set_title("DRL overshoot vs. episode")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)
path = os.path.join(OUT_DIR, "top_k_neighbor_overshoot_vs_episode.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

print("Steady-state mean coverage:")
for label, value in sorted(((v[0], v[1][steady].mean()) for v in all_series.values()), key=lambda item: -item[1]):
    print(f"  {label:38s} {value:.4f}")
