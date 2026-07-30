"""Uses Sionna's own gen_hexgrid_topology and HexGrid directly (not the
repo-local functions/topology.py / functions/simulation.py versions that
used to exist here -- those added UE-placement constraints and a
UE-plotting wrapper we don't need, and have since been deleted; see
helpers/hex_grid_view.py for the one genuinely missing piece, UE
plotting, added on top of Sionna's own HexGrid.show()).

Run: python scripts/verify_topology.py
"""

import os
import sys

from sionna.sys.topology import gen_hexgrid_topology

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from helpers.hex_grid_view import HexGridView

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "topology")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_UT_PER_SECTOR = 5
SLOT_DURATION = 0.5e-3  # representative value for this standalone check only

ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state, los, \
    bs_virtual_loc, grid = gen_hexgrid_topology(
        batch_size=1,
        num_rings=1,
        num_ut_per_sector=NUM_UT_PER_SECTOR,
        scenario="umi",
        min_ut_velocity=1.0,
        max_ut_velocity=3.0,
        return_grid=True,
    )

out_path = os.path.join(OUT_DIR, "hex_topology.png")
HexGridView(grid, ut_loc).save(out_path)
print(f"Saved: {out_path}")
print(f"BS positions: {tuple(bs_loc.shape)}, UE positions: {tuple(ut_loc.shape)}")

# Mobility check: one slot-duration position update (same update rule
# SystemLevelSimulator._run_simulation uses), confirming UEs actually move.
ut_loc_after = ut_loc + ut_velocities * SLOT_DURATION
displacement = (ut_loc_after - ut_loc).abs().max().item()
print(f"Max UE displacement after one {SLOT_DURATION * 1e3:.1f} ms step: "
     f"{displacement:.6f} m (expect > 0)")
