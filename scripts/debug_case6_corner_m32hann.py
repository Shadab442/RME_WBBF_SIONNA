"""One-off diagnostic (2026-10-12): case6_LHH's grouped+rectangular sweep
shows global coverage CRASHING at the own=11-13deg corner when N1/N2 are
also at 13deg, even though own group is near-heavy (L) and shouldn't have
much own far/mid population to lose. Decomposes global coverage by GROUP
(own/n1/n2) and, within own group, by REGION (near/mid/far) at own_tilt in
{9, 11, 13} with N1=N2=13deg fixed, to find where the drop actually comes
from.

Run: python scripts/debug_case6_corner.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sionna
import torch
import debug_pipeline_audit as dpa
from helpers.utils import load_config

sionna.phy.config.device = "cuda:0" if torch.cuda.is_available() else "cpu"


def _cond_load_config():
    cfg = load_config()
    cfg["antenna"]["m"] = 32
    cfg["antenna"]["antenna_window"] = "hanning"
    return cfg
dpa.load_config = _cond_load_config

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
print(f"groups: own={own_group.tolist()} n1={n1_group.tolist()} n2={n2_group.tolist()}")

own_s, n1_s, n2_s = dpa.NEIGHBOR_JOINT_CASES["case6_LHH"]
own_idx = dpa._FAR_HEAVY_STATE_IDX if own_s == "H" else dpa._NEAR_HEAVY_STATE_IDX
n1_idx = dpa._FAR_HEAVY_STATE_IDX if n1_s == "H" else dpa._NEAR_HEAVY_STATE_IDX
n2_idx = dpa._FAR_HEAVY_STATE_IDX if n2_s == "H" else dpa._NEAR_HEAVY_STATE_IDX
state_map = {}
for group, idx in ((own_group, own_idx), (n1_group, n1_idx), (n2_group, n2_idx)):
    for s in group:
        state_map[int(s)] = idx
dpa._force_specific_demand_states(engine, state_map, default_state_idx=0, non_anchor_fraction=0.0)

state = engine.step_environment(True)
ut_loc = engine.mobility.ut_loc.detach().cpu().numpy().copy()
sector_idx, cell_idx = engine.topology.assign_to_sector_rhombus_grid(ut_loc[:, :2])
cell_to_region = engine.regional_info_provider.cell_to_region
region_idx = cell_to_region[np.clip(cell_idx, 0, len(cell_to_region) - 1)]  # 0=near,1=mid,2=far
region_names = np.array(["near", "mid", "far"])

group_of_sector = np.full(engine.num_bs, -1)
group_of_sector[own_group] = 0
group_of_sector[n1_group] = 1
group_of_sector[n2_group] = 2
group_names = np.array(["own", "n1", "n2"])
ue_group = group_of_sector[sector_idx]

num_bs = engine.num_bs
for own_tilt in (9.0, 11.0, 13.0):
    tilt_vec = np.zeros(num_bs)
    tilt_vec[own_group] = own_tilt
    tilt_vec[n1_group] = 13.0
    tilt_vec[n2_group] = 13.0
    power_w = engine.kpi_manager.resolve_power_at_tilt(state, tilt_vec)[0]
    kpis = engine.kpi_manager.compute_ue_kpis(power_w[None, :, :])
    sinr_db = kpis["sinr_db"][0]
    covered = sinr_db > engine.coverage_threshold_db

    print(f"\n=== own_tilt={own_tilt:.0f} (N1=N2=13) === global coverage={covered.mean():.4f} "
          f"(n_ue={len(covered)})")
    for g in range(3):
        mask_g = ue_group == g
        if mask_g.sum() == 0:
            continue
        print(f"  group={group_names[g]:>3s} (n={mask_g.sum():4d}): "
              f"coverage={covered[mask_g].mean():.4f}  mean_sinr_db={sinr_db[mask_g].mean():7.2f}")
        for r in range(3):
            mask_gr = mask_g & (region_idx == r)
            if mask_gr.sum() == 0:
                continue
            print(f"      region={region_names[r]:>4s} (n={mask_gr.sum():4d}): "
                  f"coverage={covered[mask_gr].mean():.4f}  mean_sinr_db={sinr_db[mask_gr].mean():7.2f}")
