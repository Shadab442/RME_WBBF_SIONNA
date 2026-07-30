"""Visual verification of the mobility models in helpers/mobility.py.

Purely geometric -- no channel/SINR computation here. This validates that the
mobility models themselves behave correctly (clusters form, groups wander within
the coverage area, the random-walk baseline stays in bounds) before wiring them
into the SINR pipeline as a follow-up step.

Saves to results/mobility/:
  1. initial_positions.png    -- RPGM's t=0 cluster positions, colored by group,
                                  over the hex grid -- confirms UEs actually start
                                  clustered (not scattered like a uniform drop).
  2. rpgm_animation.gif       -- RPGM groups moving: each group's reference point
                                  wanders (random-waypoint) within the coverage area.
  3. random_walk_animation.gif -- RandomWalkMobility baseline: independent UEs
                                  wandering/reflecting at the boundary, for comparison.

Run: python scripts/verify_mobility.py
"""

import os
import sys

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sionna
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from helpers.cellular_topology import CellularTopology
from helpers.ue_drop import sample_uniform_ut_loc, sample_clustered_ut_loc
from helpers.scenario_view import save_scenario
from helpers.mobility import ReferencePointGroupMobility, RandomWalkMobility

sionna.phy.config.seed = 42
sionna.phy.config.precision = "single"
sionna.phy.config.device = "cpu"  # pure geometry, no channel computation -- CPU is plenty

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "mobility")
os.makedirs(OUT_DIR, exist_ok=True)

# parameters
SCENARIO = "umi"
NUM_RINGS = 2
NUM_UT = 100
UT_HEIGHT = 1.5  # m

# Mobility parameters
NUM_GROUPS = 5
MEMBERS_PER_GROUP = NUM_UT // NUM_GROUPS  # 20
DEVIATION_RADIUS_FRAC_AREA = 0.15  # cluster spread, as a fraction of the WHOLE coverage
                                   # area's radius (topo.default_drop_radius)
MEMBER_JITTER_SPEED = 1.5  # m/s (how fast individual members shuffle within their own cluster)
MIN_SPEED, MAX_SPEED = 1.0, 2.0  # m/s -- pedestrian-ish

# Simulation time parameters
SLOT_DURATION = 1.0  # s
NUM_SLOTS = 120

# Cellular site/sector topology 
min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height, indoor_probability, \
    min_ut_velocity, max_ut_velocity = set_3gpp_scenario_parameters(SCENARIO)
scenario_params = {"isd": isd, "bs_height": bs_height, "min_bs_ut_dist": min_bs_ut_dist,
                   "min_ut_height": min_ut_height}
topo = CellularTopology(scenario_params, num_rings=NUM_RINGS)
isd_m = float(isd.item())
deviation_radius = DEVIATION_RADIUS_FRAC_AREA * topo.default_drop_radius

print(f"Scenario: {NUM_RINGS} ring(s), {topo.num_cells} sites, {topo.num_bs} sectors, "
     f"ISD={isd_m:.0f} m, deviation_radius={deviation_radius:.1f} m")

# Pick real sites as each group's starting reference-point position.
site_dist = torch.linalg.norm(topo.site_loc, dim=-1)
order = torch.argsort(site_dist)
start_sites = order[:NUM_GROUPS]
start_xy_list = [tuple(topo.site_loc[s].tolist()) for s in start_sites]

# Initial (t=0) clustered UE drop 
ut_loc, member_group_idx = sample_clustered_ut_loc(
    topo, start_xy_list, MEMBERS_PER_GROUP, deviation_radius, UT_HEIGHT,
    dtype=topo.bs_loc.dtype, device=topo.bs_loc.device,
)

# RPGM
rpgm = ReferencePointGroupMobility(
    ut_loc, member_group_idx, start_xy_list, deviation_radius=deviation_radius,
    topo=topo, min_speed=MIN_SPEED, max_speed=MAX_SPEED,
    member_jitter_speed=MEMBER_JITTER_SPEED,
)

# Initial (t=0) uniform UE drop
random_walk_init_loc = sample_uniform_ut_loc(topo, NUM_UT, UT_HEIGHT, dtype=topo.bs_loc.dtype,
                                             device=topo.bs_loc.device)

# Random Walk mobility model
random_walk = RandomWalkMobility(random_walk_init_loc, topo=topo,
                                 min_speed=MIN_SPEED, max_speed=MAX_SPEED)

# 1. Initial RPGM cluster positions, colored by group
group_idx_np = rpgm.member_group_idx.numpy()
out_path = os.path.join(OUT_DIR, "initial_positions.png")
save_scenario(out_path, topo.grid, ut_loc=rpgm.ut_loc, colors=group_idx_np)
print(f"Saved: {out_path}")


def make_animation(mobility, num_slots, dt, out_path, colors, title):
    fig = topo.grid.show(show_sectors=True)
    ax = fig.gca()
    xy0 = mobility.ut_loc[:, :2].detach().cpu().numpy()
    scatter = ax.scatter(xy0[:, 0], xy0[:, 1], c=colors, cmap="tab10" if colors is not None else None,
                         marker="x")
    ax.set_title(title)

    def update(_frame):
        mobility.step(dt)
        scatter.set_offsets(mobility.ut_loc[:, :2].detach().cpu().numpy())
        return (scatter,)

    anim = FuncAnimation(fig, update, frames=num_slots, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Saved: {out_path}")


# 2. RPGM animation
make_animation(
    rpgm, NUM_SLOTS, SLOT_DURATION, os.path.join(OUT_DIR, "rpgm_animation.gif"),
    colors=group_idx_np,
    title=f"RPGM: {NUM_GROUPS} random-waypoint groups",
)

# 3. RandomWalkMobility baseline animation
make_animation(
    random_walk, NUM_SLOTS, SLOT_DURATION, os.path.join(OUT_DIR, "random_walk_animation.gif"),
    colors=None, title="RandomWalkMobility baseline (ungrouped)",
)
