"""Plots for main.py -- loads results/main/data.npz (no simulation/channel
computation here). One data point per TILT CONTROL INTERVAL, not per
1-second measurement draw -- see main.py's own docstring for the
two-timescale structure.

Saves to results/main/:
  1. coverage_vs_time.png       -- coverage over time, for whichever methods
                                    are listed in SHOW_METHODS below, last
                                    STEADY_STATE_EPISODES intervals shaded
                                    (see steady-state print block).
  2. median/p5/p1_sinr_vs_time.png -- per-UE SINR percentiles over time, same
                                    SHOW_METHODS, same coverage-search
                                    assignments -- coverage is what the
                                    methods actually optimize, this is just
                                    the same result viewed as raw SINR.
  3. drl_reward_loss_vs_episode.png -- DRL's own training curves: mean
                                    per-sector reward (top) and training
                                    loss (bottom) vs. episode (== interval).
  4. drl_overshoot_vs_episode.png -- DRL's mean per-sector overshoot vs.
                                    episode -- no other method tracks this,
                                    so no comparison lines.

Run: python scripts/plots/plot_main.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "main")

# Coverage is already pooled over many UEs/realizations per interval, so
# it's far less noisy than a single-interval median SINR, but a light
# smoothing option is kept for consistency with the other plot scripts -- a
# display choice only, tune and rerun without redoing the simulation.
MOVING_AVERAGE_WINDOW = 11

# Which methods to draw on coverage_vs_time.png -- edit and rerun, no need
# to redo the simulation (both are always saved to data.npz).
SHOW_METHODS = ["drl", "adaptive_legacy"]

data = np.load(os.path.join(OUT_DIR, "data.npz"))

ADAPTIVE_LEGACY_TILT_DEG_HISTORY = data["adaptive_legacy_tilt_deg_history"]  # [num_intervals, num_bs]
DRL_TILT_DEG_HISTORY = data["drl_tilt_deg_history"]  # [num_intervals, num_bs]
DRL_REWARD_HISTORY = data["drl_reward_history"]  # [num_intervals, num_bs]
DRL_OVERSHOOT_HISTORY = data["drl_overshoot_history"]  # [num_intervals, num_bs]
DRL_LOSS_PER_INTERVAL = data["drl_loss_per_interval"]  # [num_intervals], NaN before learning starts
DOWNTILT_SWEEP_DEG = data["downtilt_sweep_deg"].tolist()
COVERAGE_THRESHOLD_DB = float(data["coverage_threshold_db"])
NUM_TILT_CONTROL_INTERVALS = int(data["num_tilt_control_intervals"])
STEADY_STATE_EPISODES = int(data["steady_state_episodes"])
POLICY_NAME = str(data["policy_name"])
MOBILITY_MODEL = str(data["mobility_model"])
NUM_BS = int(data["num_bs"])

METHODS = {
    "adaptive_legacy": ("Adaptive Legacy", data["coverage_adaptive_legacy"], "tab:red"),
    "drl": (f"DRL ({POLICY_NAME})", data["coverage_drl"], "tab:green"),
}

# sinr_percentiles_<method> is [len(SINR_PERCENTILES), num_intervals];
# SINR_PERCENTILES gives the percentile each row is (e.g. [50, 5, 1] -- see
# main.py).
SINR_PERCENTILES = data["sinr_percentiles"].tolist()
SINR_LABEL = {50: "Median", 5: "5th-percentile", 1: "1st-percentile"}
SINR_METHODS = {
    "adaptive_legacy": ("Adaptive Legacy", data["sinr_percentiles_adaptive_legacy"], "tab:red"),
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


interval_idx = np.arange(NUM_TILT_CONTROL_INTERVALS)
steady = slice(-STEADY_STATE_EPISODES, None)
# Index (not time-coordinate) start of the steady-state window, clipped to 0
# -- interval_idx[-STEADY_STATE_EPISODES] would raise IndexError if
# STEADY_STATE_EPISODES > NUM_TILT_CONTROL_INTERVALS.
steady_state_start_idx = max(0, NUM_TILT_CONTROL_INTERVALS - STEADY_STATE_EPISODES)

# ----------------------------- coverage vs time ----------------------------
# Steady-state means get MARKED ON THE PLOT (marker on the real curve + an
# arrow to the numeric label), not left to console output alone -- a number
# that only exists in stdout is too easy to lose track of or misreport.
steady_center_idx = int(round((steady_state_start_idx + interval_idx[-1]) / 2.0))

fig, ax = plt.subplots(figsize=(10.5, 5.5))
for key in SHOW_METHODS:
    label, series, color = METHODS[key]
    smoothed = moving_average(series, MOVING_AVERAGE_WINDOW)
    ax.plot(interval_idx, smoothed, label=label, color=color)
ax.axvspan(steady_state_start_idx, interval_idx[-1], color="gray", alpha=0.15,
          label=f"steady state (last {STEADY_STATE_EPISODES})")

# Label y-positions start at each method's own steady-state mean, then get
# nudged apart (preserving rank order) so close-together methods (e.g. DRL
# vs. Adaptive Legacy) don't render as overlapping text.
xlim_right = interval_idx[-1] + 0.07 * (interval_idx[-1] - interval_idx[0])
ax.set_xlim(interval_idx[0], xlim_right)
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
    marker_xy = (interval_idx[steady_center_idx], smoothed[steady_center_idx])
    ax.plot(*marker_xy, marker="o", color=color, markersize=5, zorder=5)
    ax.annotate(
        f"{steady_mean:.4f}",
        xy=marker_xy, xytext=(xlim_right, label_y),
        color=color, fontsize=9, fontweight="bold", va="center",
        arrowprops=dict(arrowstyle="-", color=color, alpha=0.6, shrinkA=0, shrinkB=4),
    )

ax.set_xlabel("Tilt control interval")
ax.set_ylabel(f"Coverage (SINR > {COVERAGE_THRESHOLD_DB:g} dB)")
ax.set_title(f"Coverage vs. time under {MOBILITY_MODEL} mobility "
            f"({MOVING_AVERAGE_WINDOW}-interval moving average)\n"
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
        ax.plot(interval_idx, moving_average(percentiles[p_idx], MOVING_AVERAGE_WINDOW),
               label=method_label, color=color)
    ax.axvspan(steady_state_start_idx, interval_idx[-1], color="gray", alpha=0.15)
    ax.set_xlabel("Tilt control interval")
    ax.set_ylabel(f"{label} SINR (dB)")
    ax.set_title(f"{label} SINR vs. time under {MOBILITY_MODEL} mobility "
                f"({MOVING_AVERAGE_WINDOW}-interval moving average)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# Only the percentiles actually present in this run's data (kpi.sinr_percentiles
# in config.yaml) -- e.g. a 1st-percentile plot is skipped if that config only
# tracks [50, 5].
for percentile, filename, label in [
    (50, "median_sinr_vs_time.png", "Median"),
    (5, "p5_sinr_vs_time.png", "5th-percentile"),
    (1, "p1_sinr_vs_time.png", "1st-percentile"),
]:
    if percentile in SINR_PERCENTILES:
        save_sinr_percentile_plot(percentile, filename, label)

# ------------------------ DRL reward/loss vs episode ------------------------
# nanmean, not mean: a sector with no served UEs that interval is NaN (see
# KpiManager.compute_ue_kpis) -- one masked-out sector shouldn't blank out
# the whole interval's plotted point.
mean_drl_reward = np.nanmean(DRL_REWARD_HISTORY, axis=1)
mean_drl_overshoot = np.nanmean(DRL_OVERSHOOT_HISTORY, axis=1)
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
def changes_per_sector(history):
    """Per-sector count of real (nonzero, finite-to-finite) tilt changes.
    A disabled/never-run baseline's history would be all-NaN -- np.diff of
    NaNs is NaN, and count_nonzero treats NaN as "changed", so a
    controller that never ran would otherwise appear to change every
    single interval. Report those sectors as NaN instead (both of this
    codebase's own two algorithms, Adaptive Legacy and DRL, always run as
    of 2026-09-22, so this is defensive/vestigial rather than a live case).
    """
    diffs = np.diff(history, axis=0)
    valid = np.isfinite(diffs)
    counts = np.count_nonzero(valid & (diffs != 0), axis=0).astype(float)
    counts[~valid.any(axis=0)] = np.nan
    return counts


def print_variability(name, changes):
    if np.isnan(changes).all():
        print(f"Temporal variability ({name}): unavailable (baseline did not run)")
        return
    print(
        f"Temporal variability ({name}): tilt changes per sector over {NUM_TILT_CONTROL_INTERVALS - 1} "
        f"interval transitions -- min={np.nanmin(changes)}, max={np.nanmax(changes)}, "
        f"mean={np.nanmean(changes):.1f}"
    )


distinct_tilts_per_interval = np.array([
    len(set(DRL_TILT_DEG_HISTORY[interval].tolist())) for interval in range(NUM_TILT_CONTROL_INTERVALS)
])
print(
    f"Spatial variability (DRL): {distinct_tilts_per_interval.mean():.1f} distinct tilts/interval on average "
    f"(out of {len(DOWNTILT_SWEEP_DEG)} candidates, {NUM_BS} sectors)"
)
print_variability("Adaptive Legacy", changes_per_sector(ADAPTIVE_LEGACY_TILT_DEG_HISTORY))
print_variability("DRL", changes_per_sector(DRL_TILT_DEG_HISTORY))
