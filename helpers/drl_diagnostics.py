"""DrlDiagnosticsLogger: per-agent DRL decision/experience logging for the
REAL training loop (helpers/simulation_engine.py::drl_process_turn) --
config-gated (algorithms.drl.diagnostics.enabled), a true no-op when off
(nothing in this module is even imported/constructed then). See
scripts/drl_diagnostics_report.py for the offline analysis/plotting that
consumes what this writes.

One row per sector's one decision turn (async_schedule="async": num_bs
rows/interval, one per sector's own Gauss-Seidel turn, in TRUE decision
order via async_order; "sync": num_bs rows/interval too, all sharing the
same async_order since one turn covers every sector at once). Each row is
a genuine, same-turn (state, action_idx, reward, next_state) tuple:
SimulationEngine.drl_process_turn decides action_idx from a real PRE-
decision measurement (`state`), applies it, then takes a second real
POST-decision measurement (`next_state`) whose reward credits that exact
decision immediately -- no lag, no reconstruction across rows needed. Each
row also carries that decision's own Q-values/epsilon/exploratory flag and
oracle/counterfactual comparison, both evaluated at the SAME pre-decision
tilt vector the decision actually saw.
"""

import json
from pathlib import Path

import numpy as np

from helpers.utils import get_logger

logger = get_logger(__name__)


class DrlDiagnosticsLogger:
    """:ivar rows: list of plain dicts, one per log_decision call, buffered
    in memory for the whole run (flush() writes them out once at the end --
    see run_simulation).
    """

    def __init__(self, num_bs: int, downtilt_sweep_deg, policy_name: str, feature_names):
        self.num_bs = num_bs
        self.downtilt_sweep_deg = np.asarray(downtilt_sweep_deg, dtype=float)
        self.policy_name = policy_name
        self.feature_names = list(feature_names)
        self.rows = []
        self._interval = -1
        self._decision_counter = 0

    def start_interval(self, interval: int) -> None:
        """Call once at the top of each tilt-control interval -- resets
        the within-interval decision-order counter (async_order)."""
        self._interval = interval
        self._decision_counter = 0

    def log_decision(self, sector: int, training: bool, state: np.ndarray,
                     tilt_before_deg: float, action_before_idx: int,
                     action_idx: int, tilt_after_deg: float,
                     reward, has_data: bool,
                     next_state=None,
                     decision_info: dict = None,
                     oracle_action_idx: int = None, oracle_coverage_per_candidate=None,
                     td_losses=None) -> None:
        """One sector's one decision turn -- a genuine, same-turn (s, a, r,
        s') tuple (2026-09-21: SimulationEngine's reward timing is now
        always no-lag/per-turn -- decide from a real pre-decision
        measurement, apply, re-measure, credit immediately -- so `state`/
        `action_idx`/`reward`/`next_state` here really are one transition,
        not a reconstruction across two rows).

        :param state: [num_features] -- this turn's real PRE-decision
            observation, what decide() actually used to pick action_idx.
        :param tilt_before_deg, tilt_after_deg: this sector's own live tilt
            immediately before/after this turn's decision was applied.
        :param action_before_idx, action_idx: same, as tilt-sweep indices.
        :param reward, has_data: the real, POST-decision measurement's own
            reward, crediting THIS SAME turn's action_idx (no lag);
            has_data=False means reward/next_state weren't well-defined
            this turn (RLTiltController's own mask) -- the transition
            wasn't actually learned from.
        :param next_state: [num_features] -- this turn's real POST-decision
            observation (s' of the tuple) -- None if has_data is False.
        :param decision_info: Dqn._last_decision_info[sector], optional --
            {"epsilon", "exploratory", "q_values"}; None for a
            non-Dqn policy (e.g. RandomPolicy) or a masked sector.
        :param oracle_action_idx: this sector's own oracle/counterfactual
            best tilt index at the SAME live tilt vector this decision
            saw (SimulationEngine._oracle_single_sector_step), or None if
            oracle comparison is disabled this run.
        :param oracle_coverage_per_candidate: [num_tilts] network coverage
            for every candidate tilt of THIS sector, other sectors held at
            their current assignment -- oracle_coverage_per_candidate[
            action_idx] is therefore directly comparable to
            oracle_coverage_per_candidate[oracle_action_idx] (the DQN
            action's own counterfactual coverage vs. the oracle's).
        :param td_losses: this sector's own TD-loss values from THIS
            turn's observe()/learn() call, optional -- e.g.
            Dqn.step_losses[losses_before:] filtered to this sector via
            the parallel Dqn.step_loss_sectors (train_steps_per_interval
            can produce more than one gradient step per turn). None/empty
            if this sector didn't train this turn.
        """
        row = {
            "interval": self._interval,
            "sector": int(sector),
            "async_order": self._decision_counter,
            "training": bool(training),
            "has_data": bool(has_data),
            "state": json.dumps(np.asarray(state, dtype=float).round(6).tolist()),
            "next_state": (json.dumps(np.asarray(next_state, dtype=float).round(6).tolist())
                          if next_state is not None else None),
            "reward": (float(reward) if reward is not None and np.isfinite(reward) else None),
            "tilt_before_deg": float(tilt_before_deg),
            "action_before_idx": int(action_before_idx),
            "action_idx": int(action_idx),
            "tilt_after_deg": float(tilt_after_deg),
        }

        decision_info = decision_info or {}
        epsilon = decision_info.get("epsilon")
        exploratory = decision_info.get("exploratory")
        q_values = decision_info.get("q_values")
        row["epsilon"] = float(epsilon) if epsilon is not None else None
        row["exploratory"] = bool(exploratory) if exploratory is not None else None
        if q_values is not None:
            q_arr = np.asarray(q_values, dtype=float)
            q_sorted = np.sort(q_arr)[::-1]
            row["q_values"] = json.dumps(q_arr.round(6).tolist())
            row["q_max"] = float(q_sorted[0])
            row["q_gap"] = float(q_sorted[0] - q_sorted[1]) if len(q_sorted) > 1 else None
            row["greedy_action_idx"] = int(np.argmax(q_arr))
        else:
            row["q_values"] = None
            row["q_max"] = None
            row["q_gap"] = None
            row["greedy_action_idx"] = None

        if oracle_action_idx is not None:
            oc = np.asarray(oracle_coverage_per_candidate, dtype=float)
            row["oracle_action_idx"] = int(oracle_action_idx)
            row["mismatch"] = bool(oracle_action_idx != action_idx)
            row["action_distance"] = int(abs(oracle_action_idx - action_idx))
            row["oracle_coverage"] = float(oc[oracle_action_idx])
            row["dqn_coverage"] = float(oc[action_idx])
            row["oracle_regret"] = row["oracle_coverage"] - row["dqn_coverage"]
        else:
            row["oracle_action_idx"] = None
            row["mismatch"] = None
            row["action_distance"] = None
            row["oracle_coverage"] = None
            row["dqn_coverage"] = None
            row["oracle_regret"] = None

        if td_losses:
            row["td_loss_mean"] = float(np.mean(td_losses))
            row["td_loss_count"] = len(td_losses)
        else:
            row["td_loss_mean"] = None
            row["td_loss_count"] = 0

        self.rows.append(row)
        self._decision_counter += 1

    def flush(self, out_dir) -> None:
        """Writes state_feature_map.json + experience_log.parquet (CSV
        fallback if pyarrow isn't installed) to out_dir. Safe to call with
        zero rows (e.g. a policy that never got a real decision) -- writes
        an empty table rather than raising.
        """
        import pandas as pd

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "state_feature_map.json").write_text(json.dumps({
            "feature_names": self.feature_names,
            "policy_name": self.policy_name,
            "downtilt_sweep_deg": self.downtilt_sweep_deg.tolist(),
            "num_bs": self.num_bs,
        }, indent=2))

        df = pd.DataFrame(self.rows)
        try:
            import pyarrow  # noqa: F401 -- presence check only
            df.to_parquet(out_dir / "experience_log.parquet", index=False)
            logger.info("DrlDiagnosticsLogger.flush: wrote %d rows to %s/experience_log.parquet",
                       len(df), out_dir)
        except ImportError:
            df.to_csv(out_dir / "experience_log.csv", index=False)
            logger.info("DrlDiagnosticsLogger.flush: pyarrow not installed, wrote %d rows to "
                       "%s/experience_log.csv", len(df), out_dir)
