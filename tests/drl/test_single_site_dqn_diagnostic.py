"""Small-scale DQN convergence diagnostic: a standalone single-site (3
sector) deployment, DRL only (no Oracle/Causal/Adaptive Legacy/No-Tilt),
run once with each DQN model ("mlp" and "wesn") for 1000 tilt control
intervals. Follows up on the full 21-sector run showing DRL's greedy
action selection behaving close to uniform-random even after epsilon
decays to its floor (see project memory) -- this isolates the learning
problem to the smallest topology that still has real geometry (one site's
3 mutually-adjacent sectors, no cross-site neighbors), to see whether the
non-convergence is topology-scale/interference-complexity related or
something more fundamental in the state/reward/network setup.

Logs every (s_t, a_t, r_t+1, s_t+1) transition -- for all 1000 intervals,
every sector -- to a CSV, by wrapping RLTiltController.update() (not
touching any production file for this): _prev_observations/_prev_actions/
_prev_valid are read BEFORE calling through to the real update(), exactly
mirroring the internal `transition_valid = schedule & self._prev_valid &
has_data` check that decides which sectors get a real logged transition
this call.

Production changes made to support this (not just this script):
  - CellularTopology gained an optional `num_sites` param -- Sionna's
    HexGrid requires num_rings>=1 (7 sites minimum), so a true single-site
    topology isn't reachable via num_rings alone; num_sites builds the
    normal grid then truncates to the first N sites, recomputing
    sector_adjacency/corner_neighbor_ids on the smaller set with the SAME
    general-purpose geometry code (naturally reports zero cross-site
    neighbors, since none exist in the truncated set).
  - SimulationEngine.__init__ gained an optional `topology` param to accept
    a pre-built (e.g. truncated) topology instead of always building one
    from cfg["topology"].
Both are additive and backward compatible -- default behavior (no
num_sites, no topology override) is unchanged.

Run: python tests/drl/test_single_site_dqn_diagnostic.py [--num-intervals 1000]

Saves to results/tests/single_site_dqn_diagnostic/<model>/:
  transitions.csv  -- interval, sector, action_idx, action_deg, reward,
                       state_t_0..N, state_t1_0..N (one row per real transition).
  reward_loss.png  -- mean reward and loss per interval, raw + moving average.
  run.log
"""

import argparse
import copy
import csv
import os
import sys

import numpy as np
import sionna
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from helpers.cellular_topology import CellularTopology
from helpers.simulation_engine import SimulationEngine
from helpers.utils import RepoLogging, get_logger, load_config

OUT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "single_site_dqn_diagnostic")

NUM_UT = 45          # 45/9 = 5 UEs/group, matching the main experiment's group size
NUM_GROUPS = 9        # 9 groups over 1 site's 3 sectors -> 3 groups/sector, clean


def build_single_site_topology(cfg):
    scenario = cfg["topology"]["scenario"]
    min_bs_ut_dist, isd, bs_height, min_ut_height, *_ = set_3gpp_scenario_parameters(scenario)
    scenario_params = {"isd": isd, "bs_height": bs_height, "min_bs_ut_dist": min_bs_ut_dist,
                       "min_ut_height": min_ut_height}
    return CellularTopology(scenario_params, num_rings=cfg["topology"]["num_rings"], num_sites=1)


def wrap_controller_for_logging(controller, rows):
    """Records every (s_t, a_t, r_t+1, s_t+1) the real controller.update()
    would feed to policy.observe(), without touching tilt_controller.py."""
    original_update = controller.update
    state = {"interval": -1}

    def wrapped_update(observations, rewards, training, has_data=None, schedule=None, terminal=False):
        state["interval"] += 1
        num_sectors = len(observations)
        _has_data = has_data if has_data is not None else np.ones(num_sectors, dtype=bool)
        _schedule = schedule if schedule is not None else np.ones(num_sectors, dtype=bool)
        if training and controller._prev_observations is not None and _schedule.any():
            transition_valid = _schedule & controller._prev_valid & _has_data
            for s in np.where(transition_valid)[0]:
                rows.append({
                    "interval": state["interval"], "sector": int(s),
                    "action_idx": int(controller._prev_actions[s]),
                    "reward": float(rewards[s]),
                    "state_t": controller._prev_observations[s].copy(),
                    "state_t1": observations[s].copy(),
                })
        return original_update(observations, rewards, training, has_data=has_data, schedule=schedule, terminal=terminal)

    controller.update = wrapped_update


def save_transitions_csv(rows, downtilt_sweep_deg, path):
    if not rows:
        return
    num_features = len(rows[0]["state_t"])
    header = (["interval", "sector", "action_idx", "action_deg", "reward"]
             + [f"state_t_{i}" for i in range(num_features)]
             + [f"state_t1_{i}" for i in range(num_features)])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            action_deg = float(downtilt_sweep_deg[row["action_idx"]])
            writer.writerow([row["interval"], row["sector"], row["action_idx"], action_deg, row["reward"]]
                           + row["state_t"].tolist() + row["state_t1"].tolist())


def save_reward_loss_plot(reward_history, loss_history, path, title):
    mean_reward = np.nanmean(reward_history, axis=1)  # [num_intervals]
    window = 20

    def moving_average(values):
        out = np.full_like(values, np.nan, dtype=float)
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            seg = values[lo:i + 1]
            seg = seg[~np.isnan(seg)]
            if seg.size:
                out[i] = seg.mean()
        return out

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax1.plot(mean_reward, color="tab:blue", alpha=0.3, label="raw")
    ax1.plot(moving_average(mean_reward), color="tab:blue", label=f"moving avg ({window})")
    ax1.set_ylabel("mean reward (over sectors)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss_history, color="tab:red", alpha=0.3, label="raw")
    ax2.plot(moving_average(loss_history), color="tab:red", label=f"moving avg ({window})")
    ax2.set_xlabel("tilt control interval")
    ax2.set_ylabel("mean loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_one_model(base_cfg, model, num_intervals, device):
    out_dir = os.path.join(OUT_ROOT, model)
    os.makedirs(out_dir, exist_ok=True)
    logger = get_logger(__name__)

    cfg = copy.deepcopy(base_cfg)
    cfg["topology"]["num_ut"] = NUM_UT
    cfg["mobility"]["num_groups"] = NUM_GROUPS
    cfg["simulation"]["num_tilt_control_intervals"] = num_intervals
    cfg["algorithms"]["optimization"]["enabled"] = False  # skip Oracle/Causal
    cfg["algorithms"]["drl"]["dqn"]["model"] = model

    sionna.phy.config.seed = cfg["simulation"]["environment_seed"]
    topology = build_single_site_topology(cfg)
    engine = SimulationEngine(cfg, topology=topology)
    logger.info("run_one_model: model=%s num_bs=%d num_ut=%d num_intervals=%d",
               model, engine.num_bs, NUM_UT, num_intervals)

    rows = []
    wrap_controller_for_logging(engine.drl_controller, rows)

    results = engine.run_simulation()

    save_transitions_csv(rows, engine.downtilt_sweep_deg, os.path.join(out_dir, "transitions.csv"))
    save_reward_loss_plot(results["drl_reward_history"], results["drl_loss_per_interval"],
                          os.path.join(out_dir, "reward_loss.png"),
                          title=f"Single-site DQN diagnostic ({model})")

    mean_reward_first = np.nanmean(results["drl_reward_history"][:100])
    mean_reward_last = np.nanmean(results["drl_reward_history"][-100:])
    changes_per_sector = np.count_nonzero(np.diff(results["drl_tilt_deg_history"], axis=0), axis=0)
    print(f"[{model}] mean reward: first 100 intervals={mean_reward_first:.4f}, "
         f"last 100 intervals={mean_reward_last:.4f}")
    print(f"[{model}] tilt changes/sector over {num_intervals - 1} transitions: "
         f"min={changes_per_sector.min()}, max={changes_per_sector.max()}, mean={changes_per_sector.mean():.1f}")
    print(f"[{model}] saved: {out_dir}/transitions.csv ({len(rows)} rows), {out_dir}/reward_loss.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-intervals", type=int, default=1000)
    parser.add_argument("--model", choices=["mlp", "wesn"], default=None,
                       help="Run only this model (default: both, sequentially).")
    RepoLogging.add_argument(parser)
    args = parser.parse_args()
    RepoLogging.configure(args.log_level, overrides=RepoLogging.parse_overrides(args.log_level_override))

    sionna.phy.config.precision = "single"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sionna.phy.config.device = device

    os.makedirs(OUT_ROOT, exist_ok=True)
    base_cfg = load_config()

    models = [args.model] if args.model is not None else ["mlp", "wesn"]
    for model in models:
        run_one_model(base_cfg, model, args.num_intervals, device)


if __name__ == "__main__":
    main()
