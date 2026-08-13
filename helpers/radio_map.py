"""Radio map: a precomputed spatial lookup of predicted per-sector received
power -- helpers.kpi_calculator.KpiCalculator.compute_power_table's SAME
[num_tilts, num_sectors, samples] structure, just keyed by a fixed grid of
locations (helpers.ue_drop.sample_grid_points) instead of live/random UE
samples. Unlike a per-slot power table, this depends only on topology +
channel model geometry -- not on any particular UE population or point in
time -- so it's built ONCE and reused indefinitely, rather than redrawn
every slot.

SINR/coverage for any per-sector tilt ASSIGNMENT is still computed on
demand from the resulting power table via
KpiCalculator.coverage_from_assignment, exactly as for a live power table
-- a radio map is a POWER map, not a SINR map (a sector's own received
power depends only on its own tilt; SINR depends on every sector's
simultaneous tilt, since interference sums over all of them).
"""

import numpy as np
import torch


def build_radio_map(kpi_calc, channel_model, topology, grid_loc, tilt_values,
                    num_realizations: int, batch_size: int, grid_chunk_size: int,
                    ut_orientations, ut_velocities, in_state,
                    keep_realizations: bool = False) -> np.ndarray:
    """[num_tilts, num_sectors, num_grid_points] received power [W].

    By default (keep_realizations=False), this is num_realizations
    independent shadow-fading draws AVERAGED at each fixed grid position -- a
    smooth, low-noise map, good for visualization (verify_radio_map.py's
    heatmaps). But averaging LINEAR POWER before it's later used to compute
    a THRESHOLDED metric (coverage_from_assignment's sinr_db > threshold_db)
    introduces a real bias, not just noise: E[1(SINR(H)>T)] (what live
    per-sample evaluation computes) != 1(SINR(E[H])>T) (what an averaged
    map's single "representative" sample computes). Diagnosed by comparing
    live vs. map-substituted coverage/SINR at matched UE positions: the
    averaged map under-estimated both coverage and median SINR by a
    consistent margin that tracked this Jensen's-gap explanation, not grid
    resolution (serving-sector disagreement was flat across
    nearest-grid-point distance, ruling that out).

    keep_realizations=True instead returns EVERY realization as its own
    separate sample, shape [num_tilts, num_sectors,
    num_realizations*num_grid_points] (realization-major, i.e. matching
    np.tile(grid_xy, (num_realizations, 1)) for a caller's nearest-neighbor
    lookup) -- no averaging, so coverage_from_assignment thresholds each
    noisy sample individually, exactly like it does for a live power table.
    This is what tilt-search code should use; the averaged form remains the
    right one for a smooth visualization.

    A radio map's grid is typically much denser than a normal UE drop
    (hundreds of points vs. ~180 UEs elsewhere in this repo), and topology
    memory scales with batch_size*num_points -- so grid points are ALSO
    chunked (grid_chunk_size at a time), not just realizations, to keep
    each set_topology() call the same size as everywhere else in the repo.

    :param grid_loc: [num_grid_points, 3] fixed positions from
        helpers.ue_drop.sample_grid_points -- treated as "UEs" for this
        computation only, never moved.
    :param batch_size: realizations drawn per channel computation; must
        evenly divide num_realizations.
    :param grid_chunk_size: grid points processed per channel computation;
        the last chunk may be smaller.
    :param ut_orientations, ut_velocities, in_state: [batch_size,
        grid_chunk_size, ...] -- sized for ONE grid chunk, reused as-is for
        every chunk (all zeros in every caller so far, so shape is all
        that matters) except however many points the LAST chunk actually
        has, which gets a matching slice.
    """
    assert num_realizations % batch_size == 0, \
        "num_realizations must be a multiple of batch_size"
    num_grid_points = grid_loc.shape[0]
    num_tilts = len(tilt_values)
    num_sectors = len(kpi_calc.sector_etilts)
    if keep_realizations:
        total_power = np.zeros((num_tilts, num_sectors, num_realizations, num_grid_points))
    else:
        total_power = np.zeros((num_tilts, num_sectors, num_grid_points))

    for chunk_start in range(0, num_grid_points, grid_chunk_size):
        chunk_end = min(chunk_start + grid_chunk_size, num_grid_points)
        chunk_points = chunk_end - chunk_start
        grid_loc_chunk = grid_loc[chunk_start:chunk_end]
        # .repeat(), not .expand() -- expand's batch dim is a 0-stride VIEW
        # (every "copy" aliases the same memory), which Sionna's own
        # set_topology() can't write through internally (it needs distinct
        # per-batch-element memory, even when the values start identical).
        grid_loc_batch = grid_loc_chunk.unsqueeze(0).repeat(batch_size, 1, 1)
        bs_virtual_loc = topology.mirror_bs_loc(grid_loc_batch)

        num_chunks = num_realizations // batch_size
        chunk_total = None
        for c in range(num_chunks):
            # The last grid chunk can be smaller than grid_chunk_size --
            # Sionna's own scenario object rejects a shape change without
            # reset_topology() first; reset unconditionally rather than
            # track whether THIS call's shape actually differs from the
            # last one (same defensive pattern as
            # test_dynamic_scenario_tilts_effect.py's controller pass).
            channel_model.reset_topology()
            channel_model.set_topology(
                grid_loc_batch, topology.bs_loc, ut_orientations[:, :chunk_points],
                topology.bs_orientations, ut_velocities[:, :chunk_points],
                in_state[:, :chunk_points], None, bs_virtual_loc,
            )
            state = kpi_calc.generate_large_scale_state()
            power_table = kpi_calc.compute_power_table(tilt_values, state)  # [num_tilts, num_sectors, batch_size*chunk_points]
            # compute_power_table flattens [batch, chunk_points] batch-major
            # (see its reshape(power_w.shape[0], -1)) -- un-flatten the same way.
            power_table = power_table.reshape(num_tilts, num_sectors, batch_size, chunk_points)
            if keep_realizations:
                total_power[:, :, c * batch_size:(c + 1) * batch_size, chunk_start:chunk_end] = power_table
            else:
                realization_sum = power_table.sum(axis=2)  # sum over realizations -> [num_tilts, num_sectors, chunk_points]
                chunk_total = realization_sum if chunk_total is None else chunk_total + realization_sum
        if not keep_realizations:
            total_power[:, :, chunk_start:chunk_end] = chunk_total / num_realizations

    if keep_realizations:
        total_power = total_power.reshape(num_tilts, num_sectors, num_realizations * num_grid_points)
    return total_power


def nearest_grid_power(grid_xy: np.ndarray, radio_map: np.ndarray, query_xy: np.ndarray,
                       tilt_idx: int, sector_idx: int) -> np.ndarray:
    """[num_queries] radio_map's power at each query point's NEAREST grid
    point, for one (tilt, sector) -- a simple nearest-neighbor lookup, no
    interpolation, since sample_grid_points' spacing already controls
    resolution directly.

    :param grid_xy: [num_grid_points, 2].
    :param query_xy: [num_queries, 2].
    """
    dist = np.linalg.norm(query_xy[:, None, :] - grid_xy[None, :, :], axis=-1)  # [num_queries, num_grid_points]
    nearest_idx = dist.argmin(axis=-1)
    return radio_map[tilt_idx, sector_idx, nearest_idx]


def grid_density_weights(grid_xy: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    """[num_grid_points] count of query_xy points whose nearest grid point
    is each grid point -- a simple density weighting, for
    KpiCalculator.coverage_from_assignment's weights parameter (and so
    helpers.tilt_controller's selectors', via their own weights
    passthrough). Turns "where are UEs actually concentrated" into a
    per-grid-point weight without needing a live channel draw at their
    exact positions -- only their positions, not their measured SINR/power,
    are used; the power itself still comes entirely from the static map.

    :param grid_xy: [num_grid_points, 2].
    :param query_xy: [num_queries, 2] -- e.g. live UE positions, pooled
        across realizations.
    """
    dist = np.linalg.norm(query_xy[:, None, :] - grid_xy[None, :, :], axis=-1)  # [num_queries, num_grid_points]
    nearest_idx = dist.argmin(axis=-1)
    return np.bincount(nearest_idx, minlength=grid_xy.shape[0]).astype(float)
