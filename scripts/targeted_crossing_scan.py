"""Targeted follow-up to factorial_crossing_scan.py, 2026-09-28: the full
405-combo factorial found only a weak (~4.5%) two-population coverage swap
at best, capped because near-BS cells have too generous a SINR margin to
ever cross a lenient coverage threshold (0/-3/-6 dB) regardless of antenna
pattern, AND because TX power (40/46/50 dBm) made zero difference -- the
network is interference-limited (every sector transmits equally, so raising
power scales signal and interference together). This script targets the two
unexplored levers that could break both ceilings:
  - coverage_threshold_db pushed HIGHER (more demanding: 0..15 dB) so even
    near cells' generous margin can be crossed by a real misalignment.
  - bs_tx_power_dbm pushed LOWER (0..40 dBm) to find the transition into a
    noise-limited regime, where power actually matters.

Fixed at rma (only scenario with enough grid resolution, 3364 cells/sector)
and antenna.m in {16, 32} (the two best performers from the full factorial),
sweeping all 3 LOS/shadow modes. Same swap_score metric as
factorial_crossing_scan.py (min of loser/gainer fraction, sector 0, tilt
extremes -10/+10 deg).
"""
import sys, os, time, itertools, json
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import debug_pipeline_audit as dpa

SCENARIO = "rma"
TX_POWERS_DBM = [0.0, 10.0, 20.0, 30.0, 40.0]
LOS_SHADOW = [
    ("forcedlos_noshadow", True, True),
    ("forcedlos_shadow", True, False),
    ("stochastic", None, False),
]
ANTENNA_M = [16, 32]
COVERAGE_THRESHOLDS_DB = [0.0, 3.0, 6.0, 10.0, 15.0]
TILT_LO, TILT_HI = -10.0, 10.0
TARGET_SECTOR = 0

_orig_load_config = dpa.load_config


def make_loader(tx_power, force_los, disable_shadow, m, threshold):
    def _patched_load_config():
        cfg = _orig_load_config()
        cfg["topology"]["scenario"] = SCENARIO
        cfg["channel"]["bs_tx_power_dbm"] = tx_power
        cfg["channel"]["force_los"] = force_los
        cfg["channel"]["disable_shadow_fading"] = disable_shadow
        cfg["antenna"]["m"] = m
        cfg["antenna"]["antenna_window"] = "rectangular"
        cfg["kpi"]["coverage_threshold_db"] = threshold
        cfg["downtilt"]["downtilt_sweep_start_deg"] = TILT_LO
        cfg["downtilt"]["downtilt_sweep_end_deg"] = TILT_HI
        cfg["downtilt"]["tilt_step_deg"] = 20
        return cfg
    return _patched_load_config


def main():
    out_path = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", "targeted_crossing_scan.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["tag"])

    combos = list(itertools.product(TX_POWERS_DBM, LOS_SHADOW, ANTENNA_M, COVERAGE_THRESHOLDS_DB))
    total = len(combos)
    print(f"Total combinations: {total}", flush=True)
    t_start = time.time()

    with open(out_path, "a") as fout:
        for i, (tx_power, (los_label, force_los, disable_shadow), m, threshold) in enumerate(combos):
            tag = f"{SCENARIO}_tx{int(tx_power)}_{los_label}_m{m}_thr{int(threshold)}"
            if tag in done:
                continue
            t0 = time.time()
            try:
                dpa.load_config = make_loader(tx_power, force_los, disable_shadow, m, threshold)
                sionna.phy.config.seed = dpa.DEBUG_SEED
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
                result = dict(tag=tag, scenario=SCENARIO, tx_power=tx_power, los_label=los_label, m=m,
                             threshold=threshold, losers=losers, gainers=gainers, swap_score=swap_score,
                             cov_lo_frac=float(cov_lo.mean()), cov_hi_frac=float(cov_hi.mean()), n=n)
            except Exception as e:
                result = dict(tag=tag, error=str(e))
            fout.write(json.dumps(result) + "\n")
            fout.flush()
            dt = time.time() - t0
            elapsed = time.time() - t_start
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"[{i+1}/{total}] {tag} ({dt:.1f}s, elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min)", flush=True)

    print("Targeted crossing scan done.")


if __name__ == "__main__":
    main()
