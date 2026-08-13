"""Visual verification of the mobility models in helpers/mobility.py.

Purely geometric -- no channel/SINR computation here. This validates that the
mobility models themselves behave correctly (clusters form, groups wander within
the coverage area, the random-walk baseline stays in bounds) before wiring them
into the SINR pipeline as a follow-up step.

Saves to results/verifications/mobility/:
  1. initial_positions.png    -- RPGM's t=0 cluster positions, colored by group,
                                  over the hex grid -- confirms UEs actually start
                                  clustered (not scattered like a uniform drop).
  2. rpgm_animation.gif       -- RPGM groups moving: each group's reference point
                                  wanders (random-waypoint) within the coverage area.
  3. random_walk_animation.gif -- RandomWalkMobility baseline: independent UEs
                                  wandering/reflecting at the boundary, for comparison.

Run: python scripts/verifications/verify_mobility.py
"""

import os
import sys

import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import sionna
from sionna.phy.channel.utils import set_3gpp_scenario_parameters
from helpers.cellular_topology import CellularTopology
from helpers.config import load_config
from helpers.ue_drop import sample_uniform_ut_loc, sample_clustered_ut_loc, sample_valid_offset
from helpers.scenario_view import save_scenario, plot_scenario, add_cluster_ellipses, compute_cell_colors
from helpers.mobility import ReferencePointGroupMobility, RandomWalkMobility

sionna.phy.config.seed = 42
sionna.phy.config.precision = "single"
sionna.phy.config.device = "cpu"  # pure geometry, no channel computation -- CPU is plenty

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "verifications", "mobility")
os.makedirs(OUT_DIR, exist_ok=True)

CFG = load_config("verify_mobility")

# parameters
SCENARIO = CFG["scenario"]
NUM_RINGS = CFG["num_rings"]
NUM_UT = CFG["num_ut"]
UT_HEIGHT = CFG["ut_height"]  # m

# Mobility parameters. NUM_GROUPS (60) exceeds the number of sites (7, at 1
# ring), so clusters can't be pinned one-per-site -- see the per-site uniform
# scatter below instead, which gives every cell a controlled, roughly-even
# share of clusters rather than one.
NUM_GROUPS = CFG["num_groups"]
MEMBERS_PER_GROUP = NUM_UT // NUM_GROUPS
DEVIATION_RADIUS_FRAC_AREA = CFG["deviation_radius_frac_area"]
MEMBER_JITTER_SPEED = CFG["member_jitter_speed"]  # m/s
MIN_SPEED, MAX_SPEED = CFG["min_ut_speed"], CFG["max_ut_speed"]  # m/s

# Simulation time parameters
SLOT_DURATION = CFG["measurement_interval_s"]  # s
NUM_SLOTS = CFG["num_slots"]  # quick visual check, not a full study

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

# Split NUM_GROUPS as evenly as possible across every site, and scatter each
# site's share of cluster centers uniformly within that site's own cell (a
# disk of its hex circumradius around the site, rejected against the real
# coverage footprint) -- a controlled, roughly-even number of clusters per
# cell, not one cluster per site.
num_cells = topo.num_cells
base, extra = divmod(NUM_GROUPS, num_cells)
clusters_per_site = [base + 1 if site_idx < extra else base for site_idx in range(num_cells)]
cell_radius = float(topo.grid.cell_radius.item())
start_xy_list = []
for site_idx, n_clusters in enumerate(clusters_per_site):
    if n_clusters == 0:
        continue
    site_center = topo.site_loc[site_idx:site_idx + 1].expand(n_clusters, -1)
    cluster_xy = sample_valid_offset(
        site_center, cell_radius, topo, topo.bs_loc.dtype, topo.bs_loc.device,
    )
    start_xy_list.extend(tuple(xy) for xy in cluster_xy.tolist())

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

# Color each cluster by its (site, sector-within-site) combination -- a
# proper map coloring, so any two geometrically adjacent cells (same site,
# different sector, OR different but neighboring sites) always get
# different colors. Computed once from the INITIAL cluster reference
# points and held fixed for the whole animation -- recomputing it as
# clusters wander would make colors flicker/change as they cross sector
# boundaries, which reads as noise, not signal. Ellipses stay plain black
# outlines (just showing cluster extent); the color goes on the UEs.
cluster_colors, _ = compute_cell_colors(rpgm.ref_xy.numpy(), topo.site_loc, topo.num_sectors_per_site)
ue_colors = [cluster_colors[g] for g in member_group_idx.tolist()]

# 1. Initial RPGM cluster positions -- one circle per cluster (its exact
# deviation_radius, not a statistical fit to just 3 member points) instead
# of per-point group coloring, since NUM_GROUPS=60 is far more clusters
# than a colormap (tab10 has 10) can distinguish.
out_path = os.path.join(OUT_DIR, "initial_positions.png")
save_scenario(
    out_path, topo.grid, ut_loc=rpgm.ut_loc, colors=ue_colors,
    cluster_centers=rpgm.ref_xy.numpy(), cluster_radius=deviation_radius,
)
print(f"Saved: {out_path}")


def make_animation(mobility, num_slots, dt, out_path, title, cluster_radius=None, colors=None):
    """cluster_radius, if given, draws and updates one circle per cluster
    each frame from mobility.ref_xy (RPGM only -- RandomWalkMobility has no
    groups/reference points to draw one from)."""
    fig = topo.grid.show(show_sectors=True)
    ax = fig.gca()
    # grid.show()'s own layout leaves no room for a title added afterward;
    # FuncAnimation.save() bakes in a FIXED canvas (animations can't use
    # bbox_inches="tight" the way a static savefig can, since every frame
    # must share one frame size), so an unreserved title just gets clipped
    # by the top of that fixed canvas -- reserve the room explicitly.
    fig.subplots_adjust(top=0.90)
    xy0 = mobility.ut_loc[:, :2].detach().cpu().numpy()
    scatter = ax.scatter(xy0[:, 0], xy0[:, 1], marker="x", color=colors if colors is not None else "tab:red")
    ax.set_title(title)

    ellipses = None
    if cluster_radius is not None:
        ellipses = add_cluster_ellipses(ax, mobility.ref_xy.numpy(), cluster_radius)

    def update(_frame):
        mobility.step(dt)
        scatter.set_offsets(mobility.ut_loc[:, :2].detach().cpu().numpy())
        artists = [scatter]
        if ellipses is not None:
            for patch, (cx, cy) in zip(ellipses, mobility.ref_xy.numpy()):
                patch.set_center((float(cx), float(cy)))
            artists.extend(ellipses)
        return artists

    anim = FuncAnimation(fig, update, frames=num_slots, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=10))
    plt.close(fig)
    print(f"Saved: {out_path}")


# 2. RPGM animation
make_animation(
    rpgm, NUM_SLOTS, SLOT_DURATION, os.path.join(OUT_DIR, "rpgm_animation.gif"),
    title=f"RPGM: {NUM_GROUPS} random-waypoint groups",
    cluster_radius=deviation_radius, colors=ue_colors,
)

# 3. RandomWalkMobility baseline animation
make_animation(
    random_walk, NUM_SLOTS, SLOT_DURATION, os.path.join(OUT_DIR, "random_walk_animation.gif"),
    title="RandomWalkMobility baseline (ungrouped)",
)
