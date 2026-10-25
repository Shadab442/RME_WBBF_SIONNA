"""Second follow-up, 2026-09-28: the targeted threshold/tx-power scan found
switching forced-LOS/no-shadow -> stochastic (realistic per-link LOS/NLOS +
shadow fading) roughly doubled the two-population coverage-swap score (2-2.5%
-> 4-5%), while threshold/tx-power alone barely moved it. This pushes that
one lever harder: wider antenna range (up to 64 elements), finer thresholds,
fixed at stochastic LOS/shadow and tx_power=40 (confirmed power-invariant),
AVERAGED over multiple random seeds -- a single stochastic draw is one
shadow-fading realization, not necessarily representative, so a real effect
should survive averaging while a lucky/unlucky one-seed fluke would not.
"""
import sys, os, time, itertools, json
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import debug_pipeline_audit as dpa

SCENARIO = "rma"
TX_POWER_DBM = 40.0
ANTENNA_M = [16, 24, 32, 48, 64]
COVERAGE_THRESHOLDS_DB = [3.0, 6.0, 9.0, 12.0]
SEEDS = [1, 2, 3, 4, 5]
TILT_LO, TILT_HI = -10.0, 10.0
TARGET_SECTOR = 0

_orig_load_config = dpa.load_config


def make_loader(m, threshold):
    def _patched_load_config():
        cfg = _orig_load_config()
        cfg["topology"]["scenario"] = SCENARIO
        cfg["channel"]["bs_tx_power_dbm"] = TX_POWER_DBM
        cfg["channel"]["force_los"] = None
        cfg["channel"]["disable_shadow_fading"] = False
        cfg["antenna"]["m"] = m
        cfg["antenna"]["antenna_window"] = "rectangular"
        cfg["kpi"]["coverage_threshold_db"] = threshold
        cfg["downtilt"]["downtilt_sweep_start_deg"] = TILT_LO
        cfg["downtilt"]["downtilt_sweep_end_deg"] = TILT_HI
        cfg["downtilt"]["tilt_step_deg"] = 20
        return cfg
    return _patched_load_config


def main():
    out_path = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", "stochastic_push_scan.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["m"], r["threshold"], r["seed"]))

    combos = list(itertools.product(ANTENNA_M, COVERAGE_THRESHOLDS_DB, SEEDS))
    total = len(combos)
    print(f"Total combinations: {total}", flush=True)
    t_start = time.time()

    with open(out_path, "a") as fout:
        for i, (m, threshold, seed) in enumerate(combos):
            if (m, threshold, seed) in done:
                continue
            t0 = time.time()
            try:
                dpa.load_config = make_loader(m, threshold)
                sionna.phy.config.seed = seed
                engine = dpa.build_debug_engine()
                tilt_values = np.array([TILT_LO, 0.0, TILT_HI])
                pert = dpa.compute_single_sector_tilt_perturbation(
                    TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
                num_grid_cells = engine.num_grid_cells
                sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
                sinr_lo = pert["sinr_db"][0][sl]
                sinr_hi = pert["sinr_db"][2][sl]
                cov_lo = sinr_lo > engine.coverage_threshold_db
                cov_hi = sinr_hi > engine.coverage_threshold_db
                n = cov_lo.size
                losers = float((cov_lo & ~cov_hi).sum()) / n
                gainers = float((~cov_lo & cov_hi).sum()) / n
                swap_score = min(losers, gainers)
                result = dict(m=m, threshold=threshold, seed=seed, losers=losers, gainers=gainers,
                             swap_score=swap_score, cov_lo_frac=float(cov_lo.mean()),
                             cov_hi_frac=float(cov_hi.mean()), n=n)
            except Exception as e:
                result = dict(m=m, threshold=threshold, seed=seed, error=str(e))
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            dt = time.time() - t0
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"[{i+1}/{total}] m={m} thr={threshold} seed={seed} ({dt:.1f}s, elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min)", flush=True)

    print("Stochastic push scan done.")


if __name__ == "__main__":
    main()
