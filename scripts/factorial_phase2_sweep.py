"""One-off factorial sweep over phase 2 (radio_environment_validation) across
scenario x tx_power x los/shadow x antenna.m x coverage_threshold, all at a
fixed -10..10/2deg downtilt sweep and antenna_window=rectangular. Requested
2026-09-27 to map out where a genuine near/mid/far/neighbor tilt tradeoff is
achievable, after single-point exploration proved slow to converge by hand.

Each combination gets its own build_debug_engine() (topology geometry
depends on scenario) and its own phase2_radio_environment_validation() call,
writing to a uniquely-named subdirectory under
results/debug/phase2_radio_environment_validation/factorial/. Intended to
run for hours unattended (see this repo's own timing probe: ~74s/combo at
RMa's grid size, less for umi/uma) -- run inside tmux, not a plain background
shell, so it survives the CLI session ending.
"""
import sys, os, time, itertools, traceback
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import sionna
import debug_pipeline_audit as dpa

SCENARIOS = ["umi", "uma", "rma"]
TX_POWERS_DBM = [40.0, 46.0, 50.0]
# (label, force_los, disable_shadow_fading)
LOS_SHADOW = [
    ("forcedlos_noshadow", True, True),
    ("forcedlos_shadow", True, False),
    ("stochastic", None, False),
]
ANTENNA_M = [4, 8, 10, 16, 32]
COVERAGE_THRESHOLDS_DB = [0.0, -3.0, -6.0]
TILT_START, TILT_END, TILT_STEP = -10, 10, 2

OUT_ROOT = os.path.join(dpa.OUT_ROOT, "phase2_radio_environment_validation", "factorial")
PROGRESS_LOG = os.path.join(OUT_ROOT, "_progress.log")

_orig_load_config = dpa.load_config


def make_loader(scenario, tx_power, force_los, disable_shadow, m, threshold):
    def _patched_load_config():
        cfg = _orig_load_config()
        cfg["topology"]["scenario"] = scenario
        cfg["channel"]["bs_tx_power_dbm"] = tx_power
        cfg["channel"]["force_los"] = force_los
        cfg["channel"]["disable_shadow_fading"] = disable_shadow
        cfg["antenna"]["m"] = m
        cfg["antenna"]["antenna_window"] = "rectangular"
        cfg["kpi"]["coverage_threshold_db"] = threshold
        cfg["downtilt"]["downtilt_sweep_start_deg"] = TILT_START
        cfg["downtilt"]["downtilt_sweep_end_deg"] = TILT_END
        cfg["downtilt"]["tilt_step_deg"] = TILT_STEP
        return cfg
    return _patched_load_config


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    combos = list(itertools.product(SCENARIOS, TX_POWERS_DBM, LOS_SHADOW, ANTENNA_M, COVERAGE_THRESHOLDS_DB))
    total = len(combos)
    print(f"Total combinations: {total}")

    done_already = set()
    if os.path.exists(PROGRESS_LOG):
        with open(PROGRESS_LOG) as f:
            done_already = set(line.strip() for line in f if line.strip())

    t_start = time.time()
    for i, (scenario, tx_power, (los_label, force_los, disable_shadow), m, threshold) in enumerate(combos):
        tag = f"{scenario}_tx{int(tx_power)}_{los_label}_m{m}_thr{int(threshold)}"
        if tag in done_already:
            continue
        out_dir = os.path.join(OUT_ROOT, tag)
        t0 = time.time()
        try:
            dpa.load_config = make_loader(scenario, tx_power, force_los, disable_shadow, m, threshold)
            sionna.phy.config.seed = dpa.DEBUG_SEED
            engine = dpa.build_debug_engine()
            dpa.phase2_radio_environment_validation(engine, None, out_dir)
            status = "OK"
        except Exception as e:
            status = f"FAILED: {e}"
            traceback.print_exc()
        dt = time.time() - t0
        elapsed = time.time() - t_start
        eta = elapsed / (i + 1) * (total - i - 1)
        print(f"[{i+1}/{total}] {tag} -- {status} ({dt:.1f}s, elapsed={elapsed/60:.1f}min, eta={eta/60:.1f}min)", flush=True)
        with open(PROGRESS_LOG, "a") as f:
            f.write(tag + "\n")

    print("Factorial sweep done.")


if __name__ == "__main__":
    main()
