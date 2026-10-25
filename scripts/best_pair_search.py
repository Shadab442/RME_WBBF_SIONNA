"""Third follow-up, 2026-09-28: prior scans fixed the swap comparison at the
tilt sweep's own endpoints (-10/+10 deg), capping the observed two-population
coverage-swap score around ~4.5%. But the true best-case swap for a given
antenna config might occur between two DIFFERENT tilt values -- e.g.
exploiting a sidelobe (not just the main lobe) landing on a second distance
band -- rather than the arbitrary endpoints. This does a FINE, WIDE tilt
sweep (-20..30 deg, 1 deg step) for the top candidate configs from
stochastic_push_scan.py, computes per-cell coverage at every tilt, then
searches ALL PAIRS of tilts for the one maximizing swap_score -- the true
achievable ceiling for that config, not one biased by an arbitrary endpoint
choice.
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import debug_pipeline_audit as dpa

SCENARIO = "rma"
TX_POWER_DBM = 40.0
# top candidates from stochastic_push_scan.py's seed-averaged ranking
CANDIDATES = [(32, 6.0), (24, 3.0), (48, 3.0), (32, 9.0), (24, 6.0)]
SEEDS = [1, 2, 3]
TILT_LO, TILT_HI, TILT_STEP = -20.0, 30.0, 1.0
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
        cfg["downtilt"]["tilt_step_deg"] = TILT_STEP
        return cfg
    return _patched_load_config


def main():
    out_path = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", "best_pair_search.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["m"], r["threshold"], r["seed"]))

    combos = [(m, thr, seed) for (m, thr) in CANDIDATES for seed in SEEDS]
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
                tilt_values = np.arange(TILT_LO, TILT_HI + TILT_STEP, TILT_STEP)
                pert = dpa.compute_single_sector_tilt_perturbation(
                    TARGET_SECTOR, tilt_values, other_sectors_tilt_deg=0.0, engine=engine)
                num_grid_cells = engine.num_grid_cells
                sl = slice(TARGET_SECTOR * num_grid_cells, (TARGET_SECTOR + 1) * num_grid_cells)
                sinr_all = pert["sinr_db"][:, sl]                     # [T, N]
                cov_all = sinr_all > engine.coverage_threshold_db     # [T, N]
                n = cov_all.shape[1]

                # search all tilt PAIRS for the max swap score
                best = (-1.0, None, None, None, None)
                T = len(tilt_values)
                for a in range(T):
                    for b in range(a + 1, T):
                        cov_a, cov_b = cov_all[a], cov_all[b]
                        losers = float((cov_a & ~cov_b).sum()) / n
                        gainers = float((~cov_a & cov_b).sum()) / n
                        swap = min(losers, gainers)
                        if swap > best[0]:
                            best = (swap, float(tilt_values[a]), float(tilt_values[b]), losers, gainers)

                result = dict(m=m, threshold=threshold, seed=seed, best_swap=best[0],
                             tilt_a=best[1], tilt_b=best[2], losers=best[3], gainers=best[4], n=n)
            except Exception as e:
                result = dict(m=m, threshold=threshold, seed=seed, error=str(e))
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            dt = time.time() - t0
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"[{i+1}/{total}] m={m} thr={threshold} seed={seed} best_swap={result.get('best_swap')} "
                  f"({dt:.1f}s, elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min)", flush=True)

    print("Best-pair search done.")


if __name__ == "__main__":
    main()
