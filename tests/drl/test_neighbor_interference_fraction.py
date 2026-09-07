"""Check: for real, simulated UEs, what fraction of TOTAL interference
power comes from a restricted neighbor set vs the full sum over every
other sector -- informs whether a per-sector-RSRP Kriging reconstruction
can safely approximate interference using neighbors only, for THIS
topology specifically (num_rings=1, 7 sites, 21 sectors -- small enough
that "non-neighbor" may not mean "far" the way it would in a larger
network, so this is checked empirically rather than assumed).

Compares two neighbor-set definitions:
  edge     -- CellularTopology.sector_adjacency (up to 4/sector: 2 radial
              same-site siblings + 2 outer cross-site neighbors sharing
              a rhombus EDGE).
  expanded -- edge neighbors plus every OTHER sector whose rhombus has a
              CORNER coincident with one of this sector's 2 outer
              corners (where 3 hexagons meet in a full tiling). Only the
              3 center-site sectors reach the full 10; boundary sectors
              get fewer (4/6/8) since num_rings=1 leaves some corners
              with no 3rd hexagon to contribute extra sectors.

NOT an RL script: no policy, no training.

Run: python tests/drl/test_neighbor_interference_fraction.py [--num-slots 10]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import sionna
import torch

from helpers.simulation_engine import SimulationEngine
from helpers.utils import RepoLogging, get_logger, load_config

RepoLogging.configure("INFO")
logger = get_logger(__name__)


def compute_expanded_neighbor_ids(topology) -> list:
    """[num_bs] list of int arrays: edge neighbors (sector_adjacency) unioned
    with sectors sharing a rhombus CORNER at either of this sector's 2 outer
    corners (the ones opposite the site center, at boresight -+ 60 deg)."""
    bs_xy = topology.bs_loc[0, :, :2].detach().cpu().numpy()
    boresight = topology.bs_orientations[0, :, 0].detach().cpu().numpy()
    R = float(topology.grid.cell_radius.item())
    num_bs = topology.num_bs

    e1 = R * np.stack([np.cos(boresight - np.pi / 3), np.sin(boresight - np.pi / 3)], axis=-1)
    e2 = R * np.stack([np.cos(boresight + np.pi / 3), np.sin(boresight + np.pi / 3)], axis=-1)
    corner_sets = [bs_xy + e1, bs_xy + e2, bs_xy, bs_xy + e1 + e2]  # e1, e2, center, far

    tol = topology.isd.item() * 0.02 if hasattr(topology.isd, "item") else topology.isd * 0.02
    expanded = [set() for _ in range(num_bs)]
    for corner_pts in corner_sets:
        for i in range(num_bs):
            for other_pts in corner_sets:
                d = np.linalg.norm(other_pts - corner_pts[i], axis=-1)
                for j in np.where(d < tol)[0]:
                    if j != i:
                        expanded[i].add(j)

    edge_neighbor_ids = topology.neighbor_ids  # [num_bs, max_neighbors], -1-padded
    for i in range(num_bs):
        for nid in edge_neighbor_ids[i]:
            if nid >= 0:
                expanded[i].add(int(nid))

    return [np.array(sorted(s), dtype=int) for s in expanded]


def fraction_captured(power_w, power_db, neighbor_ids_per_sector):
    serving_idx = power_db.argmax(axis=0)  # [num_ut], RSRP-based attachment
    num_ut = power_w.shape[1]
    total_w = power_w.sum(axis=0)
    serving_w = power_w[serving_idx, np.arange(num_ut)]
    total_interference_w = total_w - serving_w

    captured_w = np.zeros(num_ut)
    for ue in range(num_ut):
        ids = neighbor_ids_per_sector[serving_idx[ue]]
        if len(ids) > 0:
            captured_w[ue] = power_w[ids, ue].sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total_interference_w > 0, captured_w / total_interference_w, np.nan)


def summarize(name, fractions):
    fractions = fractions[~np.isnan(fractions)]
    print(f"\n--- {name} ---")
    print(f"samples = {len(fractions)}")
    print(f"mean   = {fractions.mean():.4f}")
    print(f"median = {np.median(fractions):.4f}")
    print(f"min    = {fractions.min():.4f}")
    print(f"p5     = {np.percentile(fractions, 5):.4f}")
    print(f"p25    = {np.percentile(fractions, 25):.4f}")
    print(f"p75    = {np.percentile(fractions, 75):.4f}")
    print(f"max    = {fractions.max():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-slots", type=int, default=10,
                        help="Real measurement slots sampled (each has num_ut UEs) -- "
                             "num_slots * num_ut total UE samples.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    sionna.phy.config.precision = "single"
    sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg = load_config()
    cfg["algorithms"]["drl"]["policy_name"] = "random"  # skip constructing an unused DQN
    cfg["algorithms"]["optimization"]["enabled"] = False  # skip unused Oracle/Causal search
    sionna.phy.config.seed = args.seed
    torch.manual_seed(args.seed)
    engine = SimulationEngine(cfg)

    baseline_tilt = np.full(engine.num_bs, engine.downtilt_sweep_deg[0])
    edge_neighbor_ids = [ids[ids >= 0] for ids in engine.topology.neighbor_ids]
    expanded_neighbor_ids = compute_expanded_neighbor_ids(engine.topology)

    counts_edge = [len(x) for x in edge_neighbor_ids]
    counts_expanded = [len(x) for x in expanded_neighbor_ids]
    print("edge-only neighbor counts/sector:    ", counts_edge)
    print("expanded (edge+corner) counts/sector:", counts_expanded)

    edge_fractions, expanded_fractions = [], []
    for slot in range(args.num_slots):
        states = engine.step_environment(is_last_step=False)
        power_w = engine.kpi_manager.resolve_power_at_tilt(states[0], baseline_tilt)[0]  # [num_bs, num_ut]
        power_db = 10.0 * np.log10(power_w)

        edge_fractions.append(fraction_captured(power_w, power_db, edge_neighbor_ids))
        expanded_fractions.append(fraction_captured(power_w, power_db, expanded_neighbor_ids))

        if slot % 5 == 0:
            logger.info("slot %2d/%d done", slot, args.num_slots - 1)

    print("\n" + "=" * 70)
    print(f"NEIGHBOR-ONLY vs TOTAL INTERFERENCE FRACTION ({args.num_slots} slots)")
    print("=" * 70)
    summarize("edge-only (<=4/sector)", np.concatenate(edge_fractions))
    summarize("edge+corner expanded (4-10/sector)", np.concatenate(expanded_fractions))
    print("=" * 70)


if __name__ == "__main__":
    main()
