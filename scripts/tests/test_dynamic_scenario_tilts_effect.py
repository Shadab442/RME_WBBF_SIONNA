"""Temporal + spatial variability sanity check: does the coverage-maximizing
tilt assignment keep changing as UEs move, and does letting it vary
per-sector actually help?

Two-timescale structure: measurement_interval_s (1 s) draws a fresh channel
and advances mobility every second, but a tilt decision only happens once
every tilt_control_interval_s (300 s) -- every measurement draw in between
is pooled into that interval's data before anyone decides or is scored.
300 s was confirmed sufficient, and robust across a 1-80 m/s mobility range,
by test_tilt_window_calibration.py's non-causal windowed-oracle study; this
script applies that confirmed interval to the real method comparison instead
of an isolated calibration run. num_tilt_control_intervals=500 such
intervals total (500 x 300 s = 150,000 measurement draws).

This script is a thin driver: it loads config, calls
helpers.tilt_environment.configure_environment()/run_history()/
save_mobility_animation() (the environment/experiment logic lives there,
not here -- see that module's own docstring for why), and prints/saves the
results. The RL-specific state/action/reward definitions DRL uses live in
drl/interface.py, a separate concern again from the environment itself.

Five methods, maximizing pooled coverage (fraction of UEs with
best-serving-sector SINR > coverage_threshold_db), each scored every
CONTROL INTERVAL against that interval's own pooled data:

* Dynamic Local Oracle: DynamicTiltController(LocalTiltSelector()) -- a
  coordinate-ascent search, rerun independently every interval, warm-started
  from the PREVIOUS interval's converged assignment -- decided AND scored
  against the SAME interval's pooled power table.
  Despite the "Dynamic" name this is non-causal: it implicitly assumes zero
  latency between measuring an interval's data and having a tilt decision
  already in effect for that same interval, which no real controller can
  achieve -- hence "Oracle." It's the upper bound the causal version below
  is chasing.
* Dynamic Local Causal: a second DynamicTiltController(LocalTiltSelector())
  instance, decided and scored on a ONE-INTERVAL LAG: at interval N, first
  SCORE the assignment that was decided using data through interval N-1
  (the most recent data a real controller could actually have had) against
  interval N's actual (now-different, since UEs moved) pooled power table
  -- THEN decide the next assignment from interval N's own pooled data, for
  use/scoring at interval N+1. This is what "Dynamic Local" should honestly
  cost once you can no longer assume your decision is already in effect the
  instant you measure. Each interval's own 300-second pool already gives
  the decision plenty of samples to average over, so no additional rolling
  window across multiple intervals is needed on top.
* Adaptive Legacy: helpers.tilt_controller.AdaptiveLegacyTiltController, a
  reactive per-sector controller inspired by the classical rule-based RET
  literature (see that class's docstring) -- NOT a power-table search.
  Every interval it's SCORED at the tilt already in effect (against the
  same pooled live power this interval's other methods use), THEN updated
  from a SINGLE realization's (mobility_list[0], matching the animation/
  position history's "representative" realization) measurements POOLED
  over the interval's own 300 one-second draws -- a real deployed
  controller aggregates its indicators over its measurement window too, not
  just one instant. Same one-interval lag as Dynamic Local Causal.
* No Tilt: a fixed 0-degree (boresight) tilt on every sector, forever --
  never decided, never adapted. The "do nothing" floor every other method
  should beat.
* DRL: helpers.tilt_controller.RLTiltController wrapping a pluggable
  drl.factory policy (default: drl.independent_dqn.IndependentDqn -- 21
  fully independent per-sector Double-DQN learners, no parameter sharing).
  "Episode" == one tilt control interval: this whole run IS the training
  run, a single continuous online pass, not repeated resets. Per interval:
  scored at the tilt already in effect (pooled live power, like every other
  method), THEN its per-sector state/reward is computed (drl.interface,
  off the SAME realization-0-only pooled data Adaptive Legacy uses) and fed
  to the policy to decide the next interval's tilt -- same one-interval lag
  as Dynamic Local Causal and Adaptive Legacy.

Dynamic Local Causal has no prior interval to have decided from at interval
0, so its interval-0 SCORE uses Dynamic Local Oracle's interval-0 assignment
as a stand-in "default before any measurement" -- a free reuse, not a
separate calibration: Oracle's own interval-0 decision (LocalTiltSelector's
warm_start=None behavior, on the same interval-0 pooled power table) is
already exactly that common-tilt bootstrap search, computed anyway.

Saves to results/tests/dynamic_scenario_tilts_effect/:

1. mobility_animation.gif: the selected UE mobility model over the grid, one
   frame per CONTROL INTERVAL (the representative realization's position at
   that interval's first measurement draw).
2. drl_policy.pt: DRL's learned parameters (agents.base.TiltPolicy.save()).
3. data.npz: per-interval coverage AND per-interval SINR percentiles
   (median/p5/p1, see sinr_percentiles) for all five methods, plus Dynamic
   Local Oracle/Causal, Adaptive Legacy, and DRL's per-sector tilt
   histories, and DRL's per-sector reward/overshoot history and per-interval
   training loss -- run plot_dynamic_scenario_tilts_effect.py separately to
   produce the coverage-vs-time, SINR-vs-time, tilt-heatmap, DRL
   reward/loss, and steady-state comparison plots from it.

Run: python scripts/tests/test_dynamic_scenario_tilts_effect.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import sionna
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from helpers.config import load_config
from helpers.tilt_environment import configure_environment, run_history, save_mobility_animation

sionna.phy.config.precision = "single"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cuda:0"
sionna.phy.config.device = DEVICE

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "tests", "dynamic_scenario_tilts_effect")

CFG = load_config("test_dynamic_scenario_tilts_effect")

# ENVIRONMENT-only seed (topology/mobility/UE-drop randomness, plus Sionna's
# own pathloss/shadow-fading sampling below) -- every algorithm sees the
# identical environment trajectory regardless of its own seed.
# algorithm_seed (DRL's network init/exploration/replay) is fully separate
# -- see helpers.tilt_environment.configure_environment and
# drl/independent_dqn.py.
sionna.phy.config.seed = CFG["environment_seed"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(
        f"Device={DEVICE}; scenario={CFG['scenario']}; {CFG['num_rings']} ring(s); "
        f"{CFG['num_ut']} UEs; tilts={CFG['downtilt_sweep_deg']}; threshold={CFG['coverage_threshold_db']:g} dB"
    )
    measurement_slots_per_interval = round(CFG["tilt_control_interval_s"] / CFG["measurement_interval_s"])
    print(
        f"History={CFG['num_tilt_control_intervals']} tilt control intervals x {CFG['tilt_control_interval_s']:g} s "
        f"({measurement_slots_per_interval} x {CFG['measurement_interval_s']:g} s measurement draws/interval) x "
        f"{CFG['num_realizations_per_slot']} realizations/draw; "
        f"mobility={CFG['mobility_model']}; speed={CFG['min_ut_speed']:g}-{CFG['max_ut_speed']:g} m/s"
    )
    if CFG["mobility_model"] == "rpgm":
        print(f"RPGM={CFG['num_groups']} groups x {CFG['num_ut'] // CFG['num_groups']} UEs")

    simulation = configure_environment(CFG)
    (
        coverage_dynamic_local_oracle,
        coverage_dynamic_local_causal,
        coverage_adaptive_legacy,
        coverage_no_tilt,
        coverage_drl,
        sinr_percentiles_dynamic_local_oracle,
        sinr_percentiles_dynamic_local_causal,
        sinr_percentiles_adaptive_legacy,
        sinr_percentiles_no_tilt,
        sinr_percentiles_drl,
        dynamic_local_oracle_tilt_deg_history,
        dynamic_local_causal_tilt_deg_history,
        adaptive_legacy_tilt_deg_history,
        drl_tilt_deg_history,
        drl_reward_history,
        drl_overshoot_history,
        drl_loss_per_interval,
        no_tilt_deg,
        position_history,
        ref_xy_history,
        drl_policy,
    ) = run_history(CFG, *simulation)

    topology = simulation[0]
    # deviation_radius is recomputed the same way configure_environment()
    # derives it (a pure function of topology and deviation_radius_frac_area,
    # not worth threading through the return value just for this plot).
    deviation_radius = CFG["deviation_radius_frac_area"] * topology.default_drop_radius
    # simulation[1] is mobility_list; realization 0 is the representative
    # trajectory plotted in the animation.
    member_group_idx = getattr(simulation[1][0], "member_group_idx", None)
    save_mobility_animation(CFG, OUT_DIR, topology, position_history, ref_xy_history, deviation_radius,
                            member_group_idx)

    print(
        f"Mean coverage (full run): dynamic_local_oracle={coverage_dynamic_local_oracle.mean():.4f}, "
        f"dynamic_local_causal={coverage_dynamic_local_causal.mean():.4f}, "
        f"adaptive_legacy={coverage_adaptive_legacy.mean():.4f}, "
        f"no_tilt={coverage_no_tilt.mean():.4f}, "
        f"drl={coverage_drl.mean():.4f}"
    )
    oracle_changes_per_sector = np.count_nonzero(np.diff(dynamic_local_oracle_tilt_deg_history, axis=0), axis=0)
    causal_changes_per_sector = np.count_nonzero(np.diff(dynamic_local_causal_tilt_deg_history, axis=0), axis=0)
    adaptive_legacy_changes_per_sector = np.count_nonzero(np.diff(adaptive_legacy_tilt_deg_history, axis=0), axis=0)
    drl_changes_per_sector = np.count_nonzero(np.diff(drl_tilt_deg_history, axis=0), axis=0)
    num_transitions = CFG["num_tilt_control_intervals"] - 1
    print(
        f"Dynamic Local Oracle tilt changes per sector: min={oracle_changes_per_sector.min()}, "
        f"max={oracle_changes_per_sector.max()}, mean={oracle_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )
    print(
        f"Dynamic Local Causal tilt changes per sector: min={causal_changes_per_sector.min()}, "
        f"max={causal_changes_per_sector.max()}, mean={causal_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )
    print(
        f"Adaptive Legacy tilt changes per sector: min={adaptive_legacy_changes_per_sector.min()}, "
        f"max={adaptive_legacy_changes_per_sector.max()}, mean={adaptive_legacy_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )
    print(
        f"DRL tilt changes per sector: min={drl_changes_per_sector.min()}, "
        f"max={drl_changes_per_sector.max()}, mean={drl_changes_per_sector.mean():.1f} "
        f"(over {num_transitions} interval transitions)"
    )

    # Steady-state comparison: last steady_state_episodes intervals' mean
    # coverage, computed the SAME way for every method -- the classical
    # methods are roughly stationary from interval 0, so this mostly
    # matters for DRL, but a uniform window keeps the comparison fair.
    steady_state_episodes = CFG["steady_state_episodes"]
    steady = slice(-steady_state_episodes, None)
    print(f"Mean coverage (last {steady_state_episodes} intervals -- steady state): "
         f"dynamic_local_oracle={coverage_dynamic_local_oracle[steady].mean():.4f}, "
         f"dynamic_local_causal={coverage_dynamic_local_causal[steady].mean():.4f}, "
         f"adaptive_legacy={coverage_adaptive_legacy[steady].mean():.4f}, "
         f"no_tilt={coverage_no_tilt[steady].mean():.4f}, "
         f"drl={coverage_drl[steady].mean():.4f}")

    drl_policy.save(Path(OUT_DIR) / "drl_policy.pt")

    data_path = os.path.join(OUT_DIR, "data.npz")
    np.savez(
        data_path,
        coverage_dynamic_local_oracle=coverage_dynamic_local_oracle,
        coverage_dynamic_local_causal=coverage_dynamic_local_causal,
        coverage_adaptive_legacy=coverage_adaptive_legacy,
        coverage_no_tilt=coverage_no_tilt,
        coverage_drl=coverage_drl,
        sinr_percentiles_dynamic_local_oracle=sinr_percentiles_dynamic_local_oracle,
        sinr_percentiles_dynamic_local_causal=sinr_percentiles_dynamic_local_causal,
        sinr_percentiles_adaptive_legacy=sinr_percentiles_adaptive_legacy,
        sinr_percentiles_no_tilt=sinr_percentiles_no_tilt,
        sinr_percentiles_drl=sinr_percentiles_drl,
        sinr_percentiles=np.asarray(CFG["sinr_percentiles"]),
        dynamic_local_oracle_tilt_deg_history=dynamic_local_oracle_tilt_deg_history,
        dynamic_local_causal_tilt_deg_history=dynamic_local_causal_tilt_deg_history,
        adaptive_legacy_tilt_deg_history=adaptive_legacy_tilt_deg_history,
        drl_tilt_deg_history=drl_tilt_deg_history,
        drl_reward_history=drl_reward_history,
        drl_overshoot_history=drl_overshoot_history,
        drl_loss_per_interval=drl_loss_per_interval,
        no_tilt_deg=no_tilt_deg,
        downtilt_sweep_deg=np.asarray(CFG["downtilt_sweep_deg"]),
        coverage_threshold_db=CFG["coverage_threshold_db"],
        num_tilt_control_intervals=CFG["num_tilt_control_intervals"],
        tilt_control_interval_s=CFG["tilt_control_interval_s"],
        measurement_interval_s=CFG["measurement_interval_s"],
        steady_state_episodes=steady_state_episodes,
        policy_name=CFG["policy_name"],
        mobility_model=CFG["mobility_model"],
        num_bs=topology.num_bs,
    )
    print(f"Saved: {data_path}")


if __name__ == "__main__":
    main()
