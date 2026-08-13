"""Plots for test_dynamic_scenario_tilts_effect.py -- loads
results/tests/dynamic_scenario_tilts_effect/data.npz (no simulation/channel
computation here). One data point per TILT CONTROL INTERVAL (300 s by
default), not per 1-second measurement draw -- see the test script's own
docstring for the two-timescale structure.

Saves to results/tests/dynamic_scenario_tilts_effect/:
  1. coverage_vs_time.png       -- coverage over time, for whichever methods
                                    are listed in SHOW_METHODS below, last
                                    STEADY_STATE_EPISODES intervals shaded
                                    (see steady-state print block).
  2. median/p5/p1_sinr_vs_time.png -- per-UE SINR percentiles over time, same
                                    SHOW_METHODS, same coverage-search
                                    assignments -- coverage is what the
                                    methods actually optimize, this is just
                                    the same result viewed as raw SINR.
  3. dynamic_local_oracle_tilt_heatmap.png -- sector x interval, colored by
                                    Dynamic Local Oracle's chosen tilt -- the
                                    spatial + temporal variability evidence
                                    together.
  4. drl_reward_loss_vs_episode.png -- DRL's own training curves: mean
                                    per-sector reward (top) and training
                                    loss (bottom) vs. episode (== interval).
  5. drl_overshoot_vs_episode.png -- DRL's mean per-sector overshoot vs.
                                    episode -- no other method tracks this,
                                    so no comparison lines.

Run: python scripts/plots/plot_dynamic_scenario_tilts_effect.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "dynamic_scenario_tilts_effect")

# Coverage is already pooled over many UEs/realizations per interval, so
# it's far less noisy than a single-interval median SINR, but a light
# smoothing option is kept for consistency with the other plot scripts -- a
# display choice only, tune and rerun without redoing the simulation.
MOVING_AVERAGE_WINDOW = 11

# Which methods to draw on coverage_vs_time.png -- edit and rerun, no need
# to redo the simulation (all 5 are always saved to data.npz).
SHOW_METHODS = ["no_tilt", "drl", "adaptive_legacy", "dynamic_local_causal", "dynamic_local_oracle"]

data = np.load(os.path.join(OUT_DIR, "data.npz"))

DYNAMIC_LOCAL_ORACLE_TILT_DEG_HISTORY = data["dynamic_local_oracle_tilt_deg_history"]  # [num_intervals, num_bs]
DYNAMIC_LOCAL_CAUSAL_TILT_DEG_HISTORY = data["dynamic_local_causal_tilt_deg_history"]  # [num_intervals, num_bs]
ADAPTIVE_LEGACY_TILT_DEG_HISTORY = data["adaptive_legacy_tilt_deg_history"]  # [num_intervals, num_bs]
DRL_TILT_DEG_HISTORY = data["drl_tilt_deg_history"]  # [num_intervals, num_bs]
DRL_REWARD_HISTORY = data["drl_reward_history"]  # [num_intervals, num_bs]
DRL_OVERSHOOT_HISTORY = data["drl_overshoot_history"]  # [num_intervals, num_bs]
DRL_LOSS_PER_INTERVAL = data["drl_loss_per_interval"]  # [num_intervals], NaN before learning starts
NO_TILT_DEG = data["no_tilt_deg"]  # [num_bs]
DOWNTILT_SWEEP_DEG = data["downtilt_sweep_deg"].tolist()
COVERAGE_THRESHOLD_DB = float(data["coverage_threshold_db"])
NUM_TILT_CONTROL_INTERVALS = int(data["num_tilt_control_intervals"])
TILT_CONTROL_INTERVAL_S = float(data["tilt_control_interval_s"])
STEADY_STATE_EPISODES = int(data["steady_state_episodes"])
POLICY_NAME = str(data["policy_name"])
MOBILITY_MODEL = str(data["mobility_model"])
NUM_BS = int(data["num_bs"])

METHODS = {
    "dynamic_local_oracle": ("Dynamic Local Oracle", data["coverage_dynamic_local_oracle"], "tab:blue"),
    "dynamic_local_causal": ("Dynamic Local Causal", data["coverage_dynamic_local_causal"], "tab:orange"),
    "adaptive_legacy": ("Adaptive Legacy", data["coverage_adaptive_legacy"], "tab:red"),
    "no_tilt": ("No Tilt", data["coverage_no_tilt"], "black"),
    "drl": (f"DRL ({POLICY_NAME})", data["coverage_drl"], "tab:green"),
}

# sinr_percentiles_<method> is [len(SINR_PERCENTILES), num_intervals];
# SINR_PERCENTILES gives the percentile each row is (e.g. [50, 5, 1] -- see
# test_dynamic_scenario_tilts_effect.py).
SINR_PERCENTILES = data["sinr_percentiles"].tolist()
SINR_LABEL = {50: "Median", 5: "5th-percentile", 1: "1st-percentile"}
SINR_METHODS = {
    "dynamic_local_oracle": ("Dynamic Local Oracle", data["sinr_percentiles_dynamic_local_oracle"], "tab:blue"),
    "dynamic_local_causal": ("Dynamic Local Causal", data["sinr_percentiles_dynamic_local_causal"], "tab:orange"),
    "adaptive_legacy": ("Adaptive Legacy", data["sinr_percentiles_adaptive_legacy"], "tab:red"),
    "no_tilt": ("No Tilt", data["sinr_percentiles_no_tilt"], "black"),
    "drl": (f"DRL ({POLICY_NAME})", data["sinr_percentiles_drl"], "tab:green"),
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
# Data-coordinate (not array-index) start of the steady-state window, clipped
# to 0 -- time_s[-STEADY_STATE_EPISODES] would raise IndexError if
# STEADY_STATE_EPISODES > NUM_TILT_CONTROL_INTERVALS.
steady_state_start_s = max(0, NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES) * TILT_CONTROL_INTERVAL_S

# ----------------------------- coverage vs time ----------------------------
# Steady-state means get MARKED ON THE PLOT (marker on the real curve + an
# arrow to the numeric label), not left to console output alone -- a number
# that only exists in stdout is too easy to lose track of or misreport.
steady_center_s = (steady_state_start_s + time_s[-1]) / 2.0
steady_center_idx = int(np.argmin(np.abs(time_s - steady_center_s)))

fig, ax = plt.subplots(figsize=(10.5, 5.5))
for key in SHOW_METHODS:
    label, series, color = METHODS[key]
    smoothed = moving_average(series, MOVING_AVERAGE_WINDOW)
    ax.plot(time_s, smoothed, label=label, color=color)
ax.axvspan(steady_state_start_s, time_s[-1], color="gray", alpha=0.15,
          label=f"steady state (last {STEADY_STATE_EPISODES})")

# Label y-positions start at each method's own steady-state mean, then get
# nudged apart (preserving rank order) so close-together methods (e.g. DRL
# vs. No Tilt) don't render as overlapping text.
xlim_right = time_s[-1] + 0.22 * (time_s[-1] - time_s[0])
ax.set_xlim(time_s[0], xlim_right)
label_specs = sorted(
    ((key, METHODS[key][1][steady].mean()) for key in SHOW_METHODS),
    key=lambda item: item[1],
)
y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
min_gap = 0.05 * y_range
for i in range(1, len(label_specs)):
    key, y = label_specs[i]
    prev_key, prev_y = label_specs[i - 1]
    if y - prev_y < min_gap:
        label_specs[i] = (key, prev_y + min_gap)

for key, label_y in label_specs:
    label, series, color = METHODS[key]
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
ax.set_title(f"Coverage vs. time under {MOBILITY_MODEL} mobility "
            f"({MOVING_AVERAGE_WINDOW}-interval moving average, {TILT_CONTROL_INTERVAL_S:g} s/interval)\n"
            f"labels: steady-state mean (last {STEADY_STATE_EPISODES} intervals)")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc="lower left")
path = os.path.join(OUT_DIR, "coverage_vs_time.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")
print("Mean coverage (full run): " + ", ".join(f"{key}={METHODS[key][1].mean():.4f}" for key in METHODS))
print(f"Mean coverage (last {STEADY_STATE_EPISODES} intervals -- steady state): " +
     ", ".join(f"{key}={METHODS[key][1][steady].mean():.4f}" for key in METHODS))

# ----------------------------- SINR vs time ---------------------------------
def save_sinr_percentile_plot(percentile, filename, label):
    p_idx = SINR_PERCENTILES.index(percentile)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key in SHOW_METHODS:
        method_label, percentiles, color = SINR_METHODS[key]
        ax.plot(time_s, moving_average(percentiles[p_idx], MOVING_AVERAGE_WINDOW),
               label=method_label, color=color)
    ax.axvspan(steady_state_start_s, time_s[-1], color="gray", alpha=0.15)
    ax.set_xlabel("Elapsed time (s)")
    ax.set_ylabel(f"{label} SINR (dB)")
    ax.set_title(f"{label} SINR vs. time under {MOBILITY_MODEL} mobility "
                f"({MOVING_AVERAGE_WINDOW}-interval moving average)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


save_sinr_percentile_plot(50, "median_sinr_vs_time.png", "Median")
save_sinr_percentile_plot(5, "p5_sinr_vs_time.png", "5th-percentile")
save_sinr_percentile_plot(1, "p1_sinr_vs_time.png", "1st-percentile")

# --------------------- Dynamic Local Oracle tilt heatmap --------------------
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(
    DYNAMIC_LOCAL_ORACLE_TILT_DEG_HISTORY.T,  # [num_bs, num_intervals]
    aspect="auto",
    origin="lower",
    extent=[0, time_s[-1], -0.5, NUM_BS - 0.5],
    cmap="viridis",
    vmin=min(DOWNTILT_SWEEP_DEG),
    vmax=max(DOWNTILT_SWEEP_DEG),
)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Dynamic Local Oracle tilt (deg)")
ax.set_xlabel("Elapsed time (s)")
ax.set_ylabel("Sector index")
ax.set_title("Dynamic Local Oracle's per-sector tilt over time\n(spatial variability: rows differ -- temporal: columns change)")
path = os.path.join(OUT_DIR, "dynamic_local_oracle_tilt_heatmap.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# ------------------------ DRL reward/loss vs episode ------------------------
mean_drl_reward = DRL_REWARD_HISTORY.mean(axis=1)
mean_drl_overshoot = DRL_OVERSHOOT_HISTORY.mean(axis=1)
episode = np.arange(NUM_TILT_CONTROL_INTERVALS)

fig, (ax_reward, ax_loss) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
ax_reward.plot(episode, mean_drl_reward, color="lightgray", linewidth=1)
ax_reward.plot(episode, moving_average(mean_drl_reward, MOVING_AVERAGE_WINDOW), color="tab:green",
              label=f"{MOVING_AVERAGE_WINDOW}-episode moving average")
ax_reward.axvspan(NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES, NUM_TILT_CONTROL_INTERVALS - 1,
                  color="gray", alpha=0.15, label=f"steady state (last {STEADY_STATE_EPISODES})")
ax_reward.set_ylabel("Mean per-sector reward")
ax_reward.set_title(f"DRL ({POLICY_NAME}): reward and loss vs. episode")
ax_reward.grid(True, alpha=0.3)
ax_reward.legend(fontsize=8)

valid = ~np.isnan(DRL_LOSS_PER_INTERVAL)
if valid.any():
    ax_loss.plot(episode[valid], DRL_LOSS_PER_INTERVAL[valid], color="tab:red")
    first_valid = int(np.argmax(valid))
    ax_loss.axvline(first_valid, color="gray", linestyle="--", linewidth=1,
                    label=f"learning starts (episode {first_valid})")
    ax_loss.legend(fontsize=8)
else:
    ax_loss.text(0.5, 0.5, "No learning steps recorded", ha="center", va="center", transform=ax_loss.transAxes)
ax_loss.set_xlabel("Episode (= tilt control interval)")
ax_loss.set_ylabel("Mean per-sector training loss")
ax_loss.grid(True, alpha=0.3)

path = os.path.join(OUT_DIR, "drl_reward_loss_vs_episode.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

# --------------------------- DRL overshoot vs episode ------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(episode, mean_drl_overshoot, color="lightgray", linewidth=1)
ax.plot(episode, moving_average(mean_drl_overshoot, MOVING_AVERAGE_WINDOW), color="tab:purple",
       label=f"{MOVING_AVERAGE_WINDOW}-episode moving average")
ax.axvspan(NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES, NUM_TILT_CONTROL_INTERVALS - 1,
          color="gray", alpha=0.15, label=f"steady state (last {STEADY_STATE_EPISODES})")
ax.set_xlabel("Episode (= tilt control interval)")
ax.set_ylabel("Mean per-sector overshoot")
ax.set_title(f"DRL ({POLICY_NAME}): overshoot vs. episode")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
path = os.path.join(OUT_DIR, "drl_overshoot_vs_episode.png")
fig.savefig(path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path}")

print(
    f"DRL steady state (last {STEADY_STATE_EPISODES} episodes): "
    f"mean_reward={mean_drl_reward[steady].mean():.4f}, "
    f"mean_overshoot={mean_drl_overshoot[steady].mean():.4f}"
)

# ---------------- spatial/temporal variability summary stats ---------------
distinct_tilts_per_interval = np.array([
    len(set(DYNAMIC_LOCAL_ORACLE_TILT_DEG_HISTORY[interval].tolist())) for interval in range(NUM_TILT_CONTROL_INTERVALS)
])
oracle_changes_per_sector = np.count_nonzero(np.diff(DYNAMIC_LOCAL_ORACLE_TILT_DEG_HISTORY, axis=0), axis=0)
causal_changes_per_sector = np.count_nonzero(np.diff(DYNAMIC_LOCAL_CAUSAL_TILT_DEG_HISTORY, axis=0), axis=0)
adaptive_legacy_changes_per_sector = np.count_nonzero(np.diff(ADAPTIVE_LEGACY_TILT_DEG_HISTORY, axis=0), axis=0)
drl_changes_per_sector = np.count_nonzero(np.diff(DRL_TILT_DEG_HISTORY, axis=0), axis=0)
print(
    f"Spatial variability: {distinct_tilts_per_interval.mean():.1f} distinct tilts/interval on average "
    f"(out of {len(DOWNTILT_SWEEP_DEG)} candidates, {NUM_BS} sectors)"
)
print(
    f"Temporal variability (Oracle): tilt changes per sector over {NUM_TILT_CONTROL_INTERVALS - 1} interval transitions -- "
    f"min={oracle_changes_per_sector.min()}, max={oracle_changes_per_sector.max()}, "
    f"mean={oracle_changes_per_sector.mean():.1f}"
)
print(
    f"Temporal variability (Causal): tilt changes per sector over {NUM_TILT_CONTROL_INTERVALS - 1} interval transitions -- "
    f"min={causal_changes_per_sector.min()}, max={causal_changes_per_sector.max()}, "
    f"mean={causal_changes_per_sector.mean():.1f}"
)
print(
    f"Temporal variability (Adaptive Legacy): tilt changes per sector over {NUM_TILT_CONTROL_INTERVALS - 1} interval transitions -- "
    f"min={adaptive_legacy_changes_per_sector.min()}, max={adaptive_legacy_changes_per_sector.max()}, "
    f"mean={adaptive_legacy_changes_per_sector.mean():.1f}"
)
print(
    f"Temporal variability (DRL): tilt changes per sector over {NUM_TILT_CONTROL_INTERVALS - 1} interval transitions -- "
    f"min={drl_changes_per_sector.min()}, max={drl_changes_per_sector.max()}, "
    f"mean={drl_changes_per_sector.mean():.1f}"
)
print(f"No Tilt's fixed assignment: {sorted(set(NO_TILT_DEG.tolist()))} degrees (every sector)")
