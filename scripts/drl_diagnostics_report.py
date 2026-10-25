"""Offline DRL diagnostics report: reads a run's own
<run_dir>/diagnostics/experience_log.{parquet,csv} + state_feature_map.json
(written by helpers/drl_diagnostics.py during training, gated by
algorithms.drl.diagnostics.enabled in config.yaml -- see main.py) and
produces the diagnostics/{state,reward,decisions,q_values,learning,events}/
output tree. Entirely decoupled from the live training loop -- this
script's own cost/complexity is not constrained by a training run's
per-interval budget, unlike the logger itself.

Usage:
    python scripts/drl_diagnostics_report.py <run_out_dir> [--top-k 20]

<run_out_dir> is a run's own OUT_DIR (main.py) -- this script reads
<run_out_dir>/diagnostics/ and writes its OWN figures back into that same
diagnostics/ directory, under state/, reward/, decisions/, q_values/,
learning/, events/ subfolders.

Everything here reads column names/groupings from state_feature_map.json
and the experience log's own columns -- no hardcoded region names, action
count, or feature count (works for any configured num_own_regions/
num_bs/downtilt_sweep_deg).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_run(run_dir: Path):
    diag_dir = run_dir / "diagnostics"
    meta = json.loads((diag_dir / "state_feature_map.json").read_text())
    parquet_path = diag_dir / "experience_log.parquet"
    csv_path = diag_dir / "experience_log.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No experience_log.{{parquet,csv}} found under {diag_dir}")
    return df, meta, diag_dir


def _decode_json_matrix(series, num_features: int) -> np.ndarray:
    """[len(series), num_features] float array -- NaN row wherever the
    cell is null (e.g. next_state on a turn with has_data=False)."""
    out = np.full((len(series), num_features), np.nan)
    for i, raw in enumerate(series):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        out[i] = json.loads(raw) if isinstance(raw, str) else raw
    return out


def _semantic_groups(feature_names) -> dict:
    """feature name -> group, for grouped heatmaps -- derived from each
    name's own prefix (see drl.interface.state_feature_names), never a
    fixed index list, so this works for any num_own_regions/region_names.
    """
    groups = {}
    for name in feature_names:
        if name.startswith("own_occupancy"):
            key = "occupancy"
        elif name.startswith("own_coverage"):
            key = "coverage"
        elif name.startswith("own_sinr"):
            key = "sinr"
        elif name.startswith("neighbor"):
            key = "neighbor"
        elif "tilt" in name:
            key = "tilt"
        else:
            key = "other"
        groups.setdefault(key, []).append(name)
    return groups


def _agent_time_pivot(df, values) -> pd.DataFrame:
    return df.pivot_table(index="sector", columns="interval", values=values, aggfunc="mean")


def _heatmap(ax, matrix, title, cmap="viridis"):
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("interval")
    ax.set_ylabel("sector")
    plt.colorbar(im, ax=ax)


def state_diagnostics(df: pd.DataFrame, meta: dict, out_dir: Path) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = meta["feature_names"]
    state_mat = _decode_json_matrix(df["state"], len(feature_names))
    groups = _semantic_groups(feature_names)

    # --- Feature-info summary ---
    per_agent_means = {}
    rows = []
    for i, name in enumerate(feature_names):
        col = state_mat[:, i]
        valid = col[~np.isnan(col)]
        by_agent = df.assign(_v=col).groupby("sector")["_v"].mean()
        per_agent_means[name] = by_agent
        rows.append(dict(
            feature=name,
            mean=float(np.mean(valid)) if len(valid) else np.nan,
            std_over_time=float(np.std(valid)) if len(valid) else np.nan,
            std_across_agents=float(np.nanstd(by_agent.values)) if len(valid) else np.nan,
            min=float(np.min(valid)) if len(valid) else np.nan,
            max=float(np.max(valid)) if len(valid) else np.nan,
            near_constant_fraction=(float(np.mean(np.isclose(valid, valid[0])))
                                    if len(valid) else np.nan),
            nan_count=int(np.isnan(col).sum()),
        ))
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "state_feature_summary.csv", index=False)

    # --- Grouped heatmaps, per semantic category, one PNG per group,
    # one row of the figure per feature in that group (physical values --
    # everything here except own_tilt_norm already IS the physical
    # quantity fed to the network; own_tilt_norm is the one already-
    # normalized feature, plotted alongside the others as-is since that's
    # what the network actually sees). ---
    for group_name, names in groups.items():
        fig, axes = plt.subplots(len(names), 1, figsize=(10, max(2.0, 2.0 * len(names))),
                                 squeeze=False, sharex=True)
        for ax_row, name in zip(axes[:, 0], names):
            idx = feature_names.index(name)
            mat = _agent_time_pivot(df.assign(_v=state_mat[:, idx]), "_v")
            _heatmap(ax_row, mat.values, name)
        fig.suptitle(f"State diagnostics -- {group_name} (agent x interval)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{group_name}_heatmaps.png", dpi=110)
        plt.close(fig)

    # --- Feature-variability plot ---
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.scatter(summary_df["std_over_time"], summary_df["std_across_agents"])
    for _, r in summary_df.iterrows():
        ax.annotate(r["feature"], (r["std_over_time"], r["std_across_agents"]), fontsize=6)
    ax.set_xlabel("std over time (within an agent, across intervals)")
    ax.set_ylabel("std across agents (of each agent's own mean)")
    ax.set_title("State feature variability")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_variability.png", dpi=110)
    plt.close(fig)

    # --- Automated consistency checks ---
    checks = []
    for key in ("occupancy", "coverage"):
        for name in groups.get(key, []):
            v = state_mat[:, feature_names.index(name)]
            v = v[~np.isnan(v)]
            if len(v) and ((v < -1e-6).any() or (v > 1 + 1e-6).any()):
                checks.append(f"{name}: values outside [0,1] (min={v.min():.4f}, max={v.max():.4f})")
    if "own_tilt_norm" in feature_names:
        v = state_mat[:, feature_names.index("own_tilt_norm")]
        v = v[~np.isnan(v)]
        if len(v) and ((v < -1e-6).any() or (v > 1 + 1e-6).any()):
            checks.append(f"own_tilt_norm: values outside [0,1] (min={v.min():.4f}, max={v.max():.4f})")
    sinr_regions = sorted({n.rsplit("_", 1)[-1] for n in groups.get("sinr", [])})
    for region in sinr_regions:
        names3 = [f"own_sinr_p25_{region}", f"own_sinr_p50_{region}", f"own_sinr_p75_{region}"]
        if all(n in feature_names for n in names3):
            p25, p50, p75 = (state_mat[:, feature_names.index(n)] for n in names3)
            valid = ~(np.isnan(p25) | np.isnan(p50) | np.isnan(p75))
            bad = valid & ((p25 > p50 + 1e-6) | (p50 > p75 + 1e-6))
            if bad.any():
                checks.append(f"SINR percentile ordering violated for region={region}: "
                             f"{int(bad.sum())}/{int(valid.sum())} rows")
    for i, name in enumerate(feature_names):
        col = state_mat[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) > 5 and np.allclose(valid, valid[0]):
            checks.append(f"{name}: constant for the whole run -- possible silent freezing")
    (out_dir / "consistency_checks.txt").write_text(
        "\n".join(checks) if checks else "No consistency issues detected.")

    return state_mat


def reward_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global reward curve -- this codebase's own reward is a single
    # network-wide coverage ratio, the SAME conceptual quantity credited
    # to whichever sector's turn it is (see drl.interface.compute_reward's
    # own docstring) -- there is no separately-trackable per-sector LOCAL
    # reward contribution to decompose it into (C_i/G_i from the spec)
    # without additional per-sector connected/total-UE instrumentation,
    # out of scope here per "instrumentation only, no simulation behavior
    # change" -- own_coverage (per region, in state/) is the closest
    # already-logged decomposable signal, plotted separately below.
    fig, ax = plt.subplots(figsize=(9, 4))
    by_interval = df.groupby("interval")["reward"].mean()
    ax.plot(by_interval.index, by_interval.values, marker=".", markersize=3)
    ax.set_xlabel("interval")
    ax.set_ylabel("reward (network coverage ratio)")
    ax.set_title("Global reward vs interval (mean across this interval's own credited turns)")
    fig.tight_layout()
    fig.savefig(out_dir / "global_reward.png", dpi=110)
    plt.close(fig)

    if "dqn_coverage" in df.columns and df["dqn_coverage"].notna().any():
        mat = _agent_time_pivot(df, "dqn_coverage")
        fig, ax = plt.subplots(figsize=(10, 5))
        _heatmap(ax, mat.values, "Counterfactual network coverage under each agent's own DQN action "
                                  "(oracle_coverage_per_candidate[action_idx])")
        fig.tight_layout()
        fig.savefig(out_dir / "dqn_action_coverage_heatmap.png", dpi=110)
        plt.close(fig)

    (out_dir / "README.txt").write_text(
        "This codebase's own DRL reward is a single global network-coverage-ratio scalar (see "
        "drl/interface.py::compute_reward's own docstring) -- there is no separately-logged per-agent "
        "LOCAL coverage C_i or global contribution G_i (spec section 3) without new per-sector "
        "connected/total-UE-count instrumentation, which was out of scope for this diagnostics pass "
        "(instrumentation only, no simulation-behavior change). See state/coverage_heatmaps.png for "
        "each agent's own per-region coverage instead -- the closest already-logged decomposable signal."
    )


def decisions_diagnostics(df: pd.DataFrame, out_dir: Path, downtilt_sweep_deg) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    num_tilts = len(downtilt_sweep_deg)

    for col, title in [("action_idx", "DQN action index"), ("oracle_action_idx", "Oracle action index")]:
        if col in df.columns and df[col].notna().any():
            mat = _agent_time_pivot(df, col)
            fig, ax = plt.subplots(figsize=(10, 5))
            _heatmap(ax, mat.values, title, cmap="tab20" if num_tilts <= 20 else "viridis")
            fig.tight_layout()
            fname = "dqn_actions.png" if col == "action_idx" else "oracle_actions.png"
            fig.savefig(out_dir / fname, dpi=110)
            plt.close(fig)

    for col, fname, title, cmap in [
        ("mismatch", "mismatch_heatmap.png", "DQN != oracle action", "Greys"),
        ("action_distance", "action_distance.png", "|DQN action - oracle action| (tilt-index units)", "magma"),
        ("oracle_regret", "oracle_regret.png", "Oracle regret (oracle_coverage - dqn_coverage)", "magma"),
    ]:
        if col in df.columns and df[col].notna().any():
            mat = _agent_time_pivot(df, col)
            fig, ax = plt.subplots(figsize=(10, 5))
            _heatmap(ax, mat.astype(float).values, title, cmap=cmap)
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=110)
            plt.close(fig)

    # Action frequency: per-agent x action count, over the WHOLE run, and
    # split into exploration-dominated (epsilon > 0.5) vs
    # exploitation-dominated (epsilon <= 0.5) periods when epsilon is
    # available (Dqn only -- RandomPolicy/masked rows leave it null).
    if "action_idx" in df.columns:
        def _freq_heatmap(sub_df, fname, title):
            if sub_df.empty:
                return
            counts = sub_df.pivot_table(index="sector", columns="action_idx", values="interval",
                                        aggfunc="count", fill_value=0)
            counts = counts.reindex(columns=range(num_tilts), fill_value=0)
            fig, ax = plt.subplots(figsize=(8, 6))
            _heatmap(ax, counts.values, title, cmap="viridis")
            ax.set_xlabel("action index")
            fig.tight_layout()
            fig.savefig(out_dir / fname, dpi=110)
            plt.close(fig)

        _freq_heatmap(df, "action_frequency.png", "Action frequency (whole run)")
        if "epsilon" in df.columns and df["epsilon"].notna().any():
            _freq_heatmap(df[df["epsilon"] > 0.5], "action_frequency_exploration.png",
                         "Action frequency (epsilon > 0.5, exploration-dominated)")
            _freq_heatmap(df[df["epsilon"] <= 0.5], "action_frequency_exploitation.png",
                         "Action frequency (epsilon <= 0.5, exploitation-dominated)")

        # Flags: per-sector action collapse / still-uniform-after-exploration.
        flags = []
        exploit_df = df[df["epsilon"] <= 0.5] if "epsilon" in df.columns and df["epsilon"].notna().any() else df
        for sector, sub in exploit_df.groupby("sector"):
            counts = sub["action_idx"].value_counts(normalize=True)
            if len(counts) and counts.iloc[0] > 0.9:
                flags.append(f"sector {sector}: collapsed to action {counts.index[0]} "
                             f"({counts.iloc[0]*100:.1f}% of exploitation-period decisions)")
            elif len(counts) == num_tilts and counts.std() < 0.02:
                flags.append(f"sector {sector}: still ~uniform over all {num_tilts} actions "
                             "after exploration -- policy may not have learned a preference")
        (out_dir / "action_frequency_flags.txt").write_text(
            "\n".join(flags) if flags else "No action-collapse or still-uniform sectors flagged.")


def q_value_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "q_max" not in df.columns or not df["q_max"].notna().any():
        (out_dir / "README.txt").write_text("No Q-values logged this run (policy is not Dqn, or "
                                            "algorithms.drl.diagnostics.log_q_values was false).")
        return

    for col, fname, title in [("q_max", "q_value_statistics.png", "max Q-value"),
                              ("q_gap", "q_gap_heatmap.png", "Q-gap (top-1 - top-2 Q-value)")]:
        mat = _agent_time_pivot(df, col)
        fig, ax = plt.subplots(figsize=(10, 5))
        _heatmap(ax, mat.values, title)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=110)
        plt.close(fig)

    # Confidence x correctness classification, where oracle comparison is available.
    if "mismatch" in df.columns and df["mismatch"].notna().any():
        sub = df[df["mismatch"].notna() & df["q_gap"].notna()].copy()
        median_gap = sub["q_gap"].median()
        sub["confident"] = sub["q_gap"] >= median_gap

        def classify(row):
            if row["mismatch"]:
                return "wrong_high_confidence" if row["confident"] else "wrong_low_confidence"
            return "correct_high_confidence" if row["confident"] else "correct_low_confidence"

        sub["class"] = sub.apply(classify, axis=1)
        counts = sub["class"].value_counts()
        counts.to_csv(out_dir / "confidence_mismatch_classification.csv", header=["count"])


def learning_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [("td_loss_mean", "TD loss"), ("epsilon", "epsilon"), ("reward", "reward"),
             ("q_gap", "Q-gap"), ("dqn_coverage", "own coverage (DQN action)")]
    available = [(c, t) for c, t in fields if c in df.columns and df[c].notna().any()]
    if not available:
        return
    fig, axes = plt.subplots(len(available), 1, figsize=(9, 2.4 * len(available)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (col, title) in zip(axes, available):
        by_interval = df.groupby("interval")[col].mean()
        ax.plot(by_interval.index, by_interval.values)
        ax.set_ylabel(title, fontsize=8)
    axes[-1].set_xlabel("interval")
    fig.suptitle("Learning diagnostics (mean across agents per interval)")
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=110)
    plt.close(fig)

    if "action_idx" in df.columns:
        # Replay-buffer action composition proxy: cumulative action share
        # over the run (the real replay buffer's contents aren't reachable
        # from the logged CSV -- this is every logged decision's own
        # action, the same population the buffer is drawn from).
        comp = df.groupby(["sector", "action_idx"]).size().unstack(fill_value=0)
        comp = comp.div(comp.sum(axis=1), axis=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        _heatmap(ax, comp.values, "Logged-decision action share per sector (replay-buffer composition proxy)")
        ax.set_xlabel("action index")
        fig.tight_layout()
        fig.savefig(out_dir / "replay_action_distribution.png", dpi=110)
        plt.close(fig)


def event_drilldown(df: pd.DataFrame, meta: dict, out_dir: Path, top_k: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = meta["feature_names"]

    cols = ["interval", "sector", "async_order", "action_idx", "oracle_action_idx", "mismatch",
           "action_distance", "oracle_regret", "q_max", "q_gap", "epsilon", "exploratory",
           "reward", "td_loss_mean", "state", "next_state"]
    cols = [c for c in cols if c in df.columns]

    if "oracle_regret" in df.columns and df["oracle_regret"].notna().any():
        top_regret = df.nlargest(top_k, "oracle_regret")[cols]
        top_regret.to_csv(out_dir / "high_regret_events.csv", index=False)

    if "mismatch" in df.columns and "q_gap" in df.columns:
        confident_mismatches = df[df["mismatch"] == True].nlargest(top_k, "q_gap")[cols]  # noqa: E712
        confident_mismatches.to_csv(out_dir / "high_confidence_mismatches.csv", index=False)

    if "td_loss_mean" in df.columns and df["td_loss_mean"].notna().any():
        df.nlargest(top_k, "td_loss_mean")[cols].to_csv(out_dir / "high_td_loss_events.csv", index=False)

    # One compact plot for the single highest-regret event, if any --
    # "optional, one compact plot per high-regret event" from the spec;
    # kept to the single worst case here to avoid hundreds of tiny figures.
    if "oracle_regret" in df.columns and df["oracle_regret"].notna().any():
        worst = df.loc[df["oracle_regret"].idxmax()]
        state_vec = json.loads(worst["state"]) if isinstance(worst["state"], str) else None
        if state_vec is not None:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(range(len(feature_names)), state_vec)
            ax.set_xticks(range(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=90, fontsize=6)
            ax.set_title(f"Highest-regret event: interval={int(worst['interval'])} sector={int(worst['sector'])} "
                        f"regret={worst['oracle_regret']:.4f}")
            fig.tight_layout()
            fig.savefig(out_dir / "highest_regret_event_state.png", dpi=110)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A run's own OUT_DIR (contains diagnostics/).")
    parser.add_argument("--top-k", type=int, default=20, help="Suspicious-event ranking size (default 20).")
    args = parser.parse_args()

    df, meta, diag_dir = load_run(args.run_dir)
    print(f"Loaded {len(df)} logged decisions from {diag_dir}")

    state_diagnostics(df, meta, diag_dir / "state")
    reward_diagnostics(df, diag_dir / "reward")
    decisions_diagnostics(df, diag_dir / "decisions", meta["downtilt_sweep_deg"])
    q_value_diagnostics(df, diag_dir / "q_values")
    learning_diagnostics(df, diag_dir / "learning")
    event_drilldown(df, meta, diag_dir / "events", args.top_k)

    print(f"Diagnostics report written under {diag_dir}")


if __name__ == "__main__":
    main()
