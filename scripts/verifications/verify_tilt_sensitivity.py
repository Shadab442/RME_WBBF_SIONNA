"""Verification: how many of a sector's spatial-grid locations are
actually SENSITIVE to that sector's own tilt choice -- i.e. the coverage
outcome (SINR > threshold) at that location flips somewhere across the
candidate tilts, holding every other sector fixed -- versus locations
whose outcome would be the same no matter which tilt is picked.

Evaluates the FIXED grid-cell centers directly (CellularTopology.build_sector_rhombus_grid),
not UEs sampled via mobility -- deliberately, for two reasons: (1) mobility
only samples wherever UEs happen to wander, an incomplete and biased view
of the 36 canonical locations; (2) shadow fading is a fresh random draw
every time the channel model is queried, even at a fixed location, so a
single mobility-driven visit is one noisy sample, not a reliable read on
whether that LOCATION is actually sensitive. This script instead draws
N_REALIZATIONS independent shadow-fading realizations per grid point and
reports the FLIP RATE (fraction of realizations where the outcome
changed), not just whether it ever happened once -- "ever flipped" turns
out to be a near-meaningless bar (any location near a plausible SINR range
will occasionally cross a threshold by chance given enough draws); the
flip RATE is what actually answers "does tilt matter here."

This directly informs top_k_neighbor's top_k_locations config value (see
config.yaml) -- see project_drl_state_taxonomy_v2 memory for how the
number 10 was chosen from an earlier run of this same check.

Run: python scripts/verifications/verify_tilt_sensitivity.py

Saves to results/verifications/tilt_sensitivity/:
  flip_rate_summary.png -- per-sector count of grid cells clearing each
                           flip-rate bar (any / >=20% / >=50% / >=95%).
"""

import os
import sys

import numpy as np
import sionna
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from helpers.utils import load_config
from helpers.simulation_engine import SimulationEngine

# ----------------------------- adjustable constants -------------------------
N_REALIZATIONS = 20          # independent shadow-fading draws per grid point
BASELINE_OTHER_TILT_DEG = 0  # every OTHER sector held at this tilt throughout
FLIP_RATE_THRESHOLDS = [0.0, 0.2, 0.5, 0.95]  # "any" / "sometimes" / "often" / "reliably"

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "verifications", "tilt_sensitivity")
os.makedirs(OUT_DIR, exist_ok=True)

sionna.phy.config.precision = "single"
sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"


def evaluate_flip_rates(engine: SimulationEngine, ut_height: float) -> np.ndarray:
    """[num_bs, num_grid_cells] flip rate -- fraction of N_REALIZATIONS
    independent shadow-fading draws in which the coverage outcome at that
    grid cell differs somewhere across engine.downtilt_sweep_deg, sweeping
    ONLY that cell's own sector while every other sector stays at
    BASELINE_OTHER_TILT_DEG.
    """
    dtype, device = engine.topology.bs_loc.dtype, engine.topology.bs_loc.device
    baseline_tilt = np.full(engine.num_bs, BASELINE_OTHER_TILT_DEG, dtype=float)
    for etilt, t in zip(engine.kpi_manager.sector_etilts, baseline_tilt):
        etilt.set_tilt(t)

    def draw_outcomes(points_xy: np.ndarray, sector_i: int) -> np.ndarray:
        """[num_tilts, n_points] covered/not, one fresh channel draw."""
        n = points_xy.shape[0]
        z = torch.full((n, 1), ut_height, dtype=dtype, device=device)
        xy = torch.as_tensor(points_xy, dtype=dtype, device=device)
        ut_loc = torch.cat([xy, z], dim=-1).unsqueeze(0)
        ut_orient = torch.zeros(1, n, 3, dtype=dtype, device=device)
        ut_vel = torch.zeros_like(ut_orient)
        in_state = torch.zeros(1, n, dtype=torch.bool, device=device)
        bs_virtual = engine.topology.mirror_bs_loc(ut_loc)
        engine.channel_model.set_topology(ut_loc, engine.topology.bs_loc[:1], ut_orient,
                                          engine.topology.bs_orientations[:1], ut_vel,
                                          in_state, None, bs_virtual)
        state = engine.large_scale_channel.generate_state()
        outcomes = []
        for t in engine.downtilt_sweep_deg:
            engine.kpi_manager.sector_etilts[sector_i].set_tilt(t)
            power = engine.kpi_manager.compute_tilt_sector_ue_rx_power(state)
            kpis = engine.kpi_manager.compute_ue_kpis(power, threshold_db=engine.coverage_threshold_db)
            outcomes.append(kpis["sinr_db"][0] > engine.coverage_threshold_db)
        engine.kpi_manager.sector_etilts[sector_i].set_tilt(baseline_tilt[sector_i])
        return np.stack(outcomes, axis=0)

    flip_rates = np.zeros((engine.num_bs, engine.num_grid_cells))
    for i in range(engine.num_bs):
        flip_counts = np.zeros(engine.num_grid_cells)
        for _ in range(N_REALIZATIONS):
            outcomes = draw_outcomes(engine.grid_points[i], i)
            flip_counts += ~np.all(outcomes == outcomes[0:1], axis=0)
        flip_rates[i] = flip_counts / N_REALIZATIONS
    return flip_rates


def main():
    cfg = load_config()
    cfg["algorithms"]["drl"]["state_type"] = "top_k_neighbor"  # only to build the grid; no training happens
    torch.manual_seed(0)
    engine = SimulationEngine(cfg)

    print(f"scenario={cfg['topology']['scenario']}, {engine.num_bs} sectors, "
         f"{engine.grid_n}x{engine.grid_n}={engine.num_grid_cells} grid cells/sector, "
         f"candidate tilts={engine.downtilt_sweep_deg.tolist()} deg, "
         f"other sectors held at {BASELINE_OTHER_TILT_DEG} deg, N={N_REALIZATIONS} draws/cell")

    flip_rates = evaluate_flip_rates(engine, cfg["topology"]["ut_height"])  # [num_bs, num_grid_cells]

    counts_per_threshold = {tau: (flip_rates >= tau if tau > 0 else flip_rates > 0).sum(axis=1)
                            for tau in FLIP_RATE_THRESHOLDS}
    for tau, counts in counts_per_threshold.items():
        label = "any flip" if tau == 0.0 else f">={tau:.0%} flip rate"
        print(f"cells/sector clearing '{label}': mean={counts.mean():.2f}, "
             f"median={np.median(counts):.1f}, max={counts.max()}")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sector_idx = np.arange(engine.num_bs)
    width = 0.8 / len(FLIP_RATE_THRESHOLDS)
    for k, (tau, counts) in enumerate(counts_per_threshold.items()):
        label = "any flip (>0%)" if tau == 0.0 else f">={tau:.0%} flip rate"
        ax.bar(sector_idx + k * width, counts, width=width, label=label)
    ax.set_xlabel("Sector index")
    ax.set_ylabel(f"Grid cells clearing threshold (out of {engine.num_grid_cells})")
    ax.set_title(f"Tilt-sensitive grid cells per sector\n"
                f"(N={N_REALIZATIONS} shadow-fading draws/cell, others held at {BASELINE_OTHER_TILT_DEG} deg)")
    ax.set_xticks(sector_idx)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    path = os.path.join(OUT_DIR, "flip_rate_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
