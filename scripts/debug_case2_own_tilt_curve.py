"""One-off (2026-10-12): why does case2_HLL (own=H, pure 0:0:10 far-only
population) pick own_tilt=9deg instead of 7deg (the lowest available in the
7-13 sweep)? Prints the full own-tilt curve at the winning N1/N2 combo,
plus own group's OWN region breakdown at own_tilt=7 vs 9, to see whether
own's far coverage is actually still improving at 7 (and something else is
dragging the global metric down) or whether own's far coverage itself
already peaks by 9.

Run: python scripts/debug_case2_own_tilt_curve.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import torch
import debug_pipeline_audit as dpa
from helpers.utils import load_config


def _cond_load_config():
    cfg = load_config()
    cfg["antenna"]["antenna_window"] = "hanning"
    cfg["mobility"]["hotspot"]["demand_states"][1] = [10, 0, 0]
    cfg["mobility"]["hotspot"]["demand_states"][2] = [0, 0, 10]
    cfg["downtilt"]["downtilt_sweep_start_deg"] = 7
    cfg["downtilt"]["downtilt_sweep_end_deg"] = 13
    cfg["downtilt"]["tilt_step_deg"] = 2
    return cfg
dpa.load_config = _cond_load_config

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
sionna.phy.config.seed = dpa.DEBUG_SEED

engine = dpa.build_debug_engine()
scenario = dpa.load_or_run_scenario(engine, use_cache=False)

target_sector = 0
nbr = engine.topology.neighbor_ids[target_sector]
nbr = nbr[nbr >= 0]
n1, n2 = int(nbr[0]), int(nbr[1])

base_state_idx = engine.mobility._state_idx.copy()
own_group = np.flatnonzero(base_state_idx == base_state_idx[target_sector])
n1_group = np.flatnonzero(base_state_idx == base_state_idx[n1])
n2_group = np.flatnonzero(base_state_idx == base_state_idx[n2])

res = dpa.compute_neighbor_joint_tilt_sweep_grouped(
    engine, target_sector, n1, n2, "H", "L", "L", own_group, n1_group, n2_group)
tilt_values = res["tilt_values"]
gc = res["global_coverage"]
oi, ni, nj = np.unravel_index(np.argmax(gc), gc.shape)
print(f"tilt_values={tilt_values}")
print(f"argmax at own_idx={oi} n1_idx={ni} n2_idx={nj} -> "
     f"own={tilt_values[oi]:.0f} n1={tilt_values[ni]:.0f} n2={tilt_values[nj]:.0f} cov={gc[oi,ni,nj]:.4f}")
print(f"\nown-tilt curve at N1={tilt_values[ni]:.0f}, N2={tilt_values[nj]:.0f} (the winning combo):")
for o in range(len(tilt_values)):
    print(f"  own_tilt={tilt_values[o]:.0f}: global_coverage={gc[o, ni, nj]:.4f}")

# Now decompose own group's OWN far-region coverage at own_tilt=7 vs 9,
# N1/N2 fixed at the winning combo.
state_map = {}
for group, idx in ((own_group, dpa._FAR_HEAVY_STATE_IDX), (n1_group, dpa._NEAR_HEAVY_STATE_IDX),
                   (n2_group, dpa._NEAR_HEAVY_STATE_IDX)):
    for s in group:
        state_map[int(s)] = idx
dpa._force_specific_demand_states(engine, state_map, default_state_idx=0, non_anchor_fraction=0.0)
state = engine.step_environment(True)
ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
sector_idx, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])
cell_to_region = engine.regional_info_provider.cell_to_region
region_idx = cell_to_region[np.clip(cell_idx, 0, len(cell_to_region) - 1)]
own_mask = np.isin(sector_idx, own_group)
n1_mask = np.isin(sector_idx, n1_group)
n2_mask = np.isin(sector_idx, n2_group)

print(f"\nown group (n={own_mask.sum()}) + neighbor breakdown at own_tilt=7 vs 9 (N1={tilt_values[ni]:.0f}, N2={tilt_values[nj]:.0f} fixed):")
num_bs = engine.num_bs
for own_tilt in (7.0, 9.0, 11.0, 13.0):
    tilt_vec = np.zeros(num_bs)
    tilt_vec[own_group] = own_tilt
    tilt_vec[n1_group] = tilt_values[ni]
    tilt_vec[n2_group] = tilt_values[nj]
    power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]
    kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
    sinr_db = kpis["sinr_db"][0]
    covered = sinr_db > engine.coverage_threshold_db
    print(f"  own_tilt={own_tilt:.0f}: global={covered.mean():.4f}  "
         f"own(n={own_mask.sum()})={covered[own_mask].mean():.4f}(sinr={sinr_db[own_mask].mean():.2f}dB)  "
         f"N1(n={n1_mask.sum()})={covered[n1_mask].mean():.4f}(sinr={sinr_db[n1_mask].mean():.2f}dB)  "
         f"N2(n={n2_mask.sum()})={covered[n2_mask].mean():.4f}(sinr={sinr_db[n2_mask].mean():.2f}dB)")
