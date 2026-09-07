"""Per-UE downlink KPIs (SINR, RSRP, coverage, overshoot) across candidate
sectors (KpiManager), plus pooling raw measurement draws into either the
shared tilt-control-interval boundary or DRL's own per-sector staggered
window (IntervalMeasurementPool).

Large-scale only: pathloss + shadow fading, no fast/frequency-selective
fading.
"""

from typing import List

import numpy as np
import torch

from .electrical_downtilt import ElectricalDowntilt
from .large_scale_channel import LargeScaleState
from helpers.utils import get_logger

logger = get_logger(__name__)


class KpiManager:
    """Pure function of (channel state, tilt) -> received power/KPIs --
    holds no pooling state of its own (see IntervalMeasurementPool below).

    :param sector_etilts: One :class:`~helpers.electrical_downtilt.ElectricalDowntilt`
        per sector, in the same order as the channel model's tx (BS) dimension.
    :param tx_power_w: Total sector transmit power [W].
    :param noise_power_w: Total-bandwidth noise power [W].
    :param bs_xy: [num_sectors, 2] per-sector (x, y) position -- needed for
        overshoot/edge-restricted coverage's UE-to-site distance.
    :param neighbor_ids: [num_sectors, max_neighbors] int, -1-padded --
        from CellularTopology (adjacency is pure site/sector geometry,
        computed there).
    :param max_neighbors: widest neighbor count across all sectors.
    :param sector_adjacency: [num_sectors, num_sectors] bool, symmetric.
    """

    def __init__(self, sector_etilts: List[ElectricalDowntilt], tx_power_w: float,
                noise_power_w: float, bs_xy, neighbor_ids: np.ndarray, max_neighbors: int,
                sector_adjacency: np.ndarray):
        self.sector_etilts = sector_etilts
        self.tx_power_w = tx_power_w
        self.noise_power_w = noise_power_w
        self.bs_xy = bs_xy.detach().cpu().numpy() if hasattr(bs_xy, "detach") else np.asarray(bs_xy)
        self.neighbor_ids = neighbor_ids
        self.max_neighbors = max_neighbors
        self.sector_adjacency = sector_adjacency

    def _rx_power_now(self, state: LargeScaleState) -> np.ndarray:
        """One tilt snapshot at every sector's CURRENT tilt, [batch, sector, ue], numpy."""
        logger.function("_rx_power_now start")
        num_sectors = len(self.sector_etilts)
        batch, num_ue = state.total_pathloss_db.shape[0], state.total_pathloss_db.shape[2]
        power = torch.zeros(batch, num_sectors, num_ue,
                            dtype=state.total_pathloss_db.dtype, device=state.total_pathloss_db.device)
        for s, etilt in enumerate(self.sector_etilts):
            # Tilt angles
            theta_flat = state.theta_lcs[:, s, :].reshape(-1)
            phi_flat = state.phi_lcs[:, s, :].reshape(-1)

            # Antenna array gain
            gain = etilt.gain_pattern(theta_flat, phi_flat).reshape(batch, num_ue)

            # Path loss linear
            pathloss_lin = 10.0 ** (state.total_pathloss_db[:, s, :] / 10.0)

            # Rx power
            power[:, s, :] = self.tx_power_w * gain / pathloss_lin
        result = power.detach().cpu().numpy()
        logger.debug("_rx_power_now: result shape=%s", result.shape)
        logger.function("_rx_power_now end")
        return result

    def resolve_power_at_tilt(self, state: LargeScaleState, tilt_deg_per_sector) -> np.ndarray:
        """Sets every sector to tilt_deg_per_sector and resolves power --
        [batch, sector, ue], numpy. The one place sector_etilts gets
        mutated for a single-tilt resolve, so callers (IntervalMeasurementPool)
        never touch sector_etilts directly.
        """
        logger.function("resolve_power_at_tilt start")
        tilt_list = tilt_deg_per_sector.tolist() if hasattr(tilt_deg_per_sector, "tolist") else list(tilt_deg_per_sector)
        for etilt, tilt in zip(self.sector_etilts, tilt_list):
            etilt.set_tilt(tilt)
        power = self._rx_power_now(state)
        logger.function("resolve_power_at_tilt end")
        return power

    def compute_tilt_sector_ue_rx_power(self, state: LargeScaleState, tilt_values=None) -> np.ndarray:
        """Received power [W], numpy. [batch, sector, ue] at current tilts,
        or [tilt, batch, sector, ue] if tilt_values given.

        :param state: A large-scale state (see helpers.large_scale_channel)
        :param tilt_values: candidate downtilts [deg], optional
        """
        logger.function("compute_tilt_sector_ue_rx_power start: tilt_values=%s", tilt_values)
        if tilt_values is None:
            power = self._rx_power_now(state)
            logger.function("compute_tilt_sector_ue_rx_power end")
            return power

        # Sweep: every sector set to the SAME candidate tilt per row.
        rows = []
        for downtilt_deg in tilt_values:
            for etilt in self.sector_etilts:
                etilt.set_tilt(downtilt_deg)
            rows.append(self._rx_power_now(state))
        result = np.stack(rows, axis=0)
        logger.debug("compute_tilt_sector_ue_rx_power: swept table shape=%s", result.shape)
        logger.function("compute_tilt_sector_ue_rx_power end")
        return result

    def _compute_serving_sinr(self, power_w: np.ndarray, sector_axis: int = 1) -> dict:
        """Signal/interference/SINR (every candidate server's, not just the
        best) and per-sector power in dB -- the shared core compute_ue_kpis
        builds on.

        :param power_w: received power [W], sector along sector_axis
        :output: dict with sinr_db, power_db (both power_w's shape),
            serving_idx (sector_axis removed) -- the argmax-RSRP sector
            (max received power, i.e. 3GPP-style attachment -- NOT
            argmax-SINR; interference can make those disagree)
        """
        logger.function("_compute_serving_sinr start")
        total_w = power_w.sum(axis=sector_axis, keepdims=True)
        interference_w = total_w - power_w
        sinr_db = 10.0 * np.log10(power_w / (interference_w + self.noise_power_w))
        power_db = 10.0 * np.log10(power_w)
        serving_idx = power_db.argmax(axis=sector_axis)
        if not np.isfinite(sinr_db).all():
            logger.warning("_compute_serving_sinr: sinr_db contains %d non-finite value(s)",
                          int((~np.isfinite(sinr_db)).sum()))
        logger.function("_compute_serving_sinr end")
        return {"sinr_db": sinr_db, "power_db": power_db, "serving_idx": serving_idx}

    def compute_ue_kpis(self, power_w: np.ndarray, tilt_idx_per_sector=None,
                        threshold_db: float = None,
                        ut_loc: np.ndarray = None, edge_percentile: float = None,
                        distance_threshold_m: float = None, overlap_margin_db: float = None,
                        return_counts: bool = False, return_per_neighbor: bool = False) -> dict:
        """Per-UE SINR/RSRP/serving-sector, plus coverage/overshoot, from a
        resolved power array.

        :param power_w: [batch, sector, ue], or [tilt, batch, sector, ue]
            with tilt_idx_per_sector given
        :param tilt_idx_per_sector: [num_sectors] int array/list, optional
        :param threshold_db: also returns coverage if given
        :param ut_loc: [batch, ue, 3] UE positions -- required (with
            threshold_db) for per_sector_coverage, and (with
            distance_threshold_m + overlap_margin_db) for overshoot
        :param edge_percentile: if given (needs ut_loc + threshold_db),
            per_sector_coverage is restricted to each sector's own cell-edge
            population (percentile of that sector's served UEs by distance)
            -- Adaptive Legacy's r_bc ingredient. Omit for DRL's
            whole-population per-sector coverage.
        :param distance_threshold_m, overlap_margin_db: if BOTH given (needs
            ut_loc), also returns "overshoot" -- per-sector overshoot
            fraction (n_os), shared by Adaptive Legacy and DRL.
        :param return_counts: if True, also returns the raw numerator/
            denominator COUNTS behind per_sector_coverage/overshoot
            (per_sector_served_count, per_sector_covered_count,
            neighbor_served_count, overshoot_hit_count)
        :output: dict -- sinr_db, rsrp_dbm, serving_idx [batch,ue] always;
            coverage [scalar] if threshold_db given; per_sector_coverage
            [sector] if ut_loc+threshold_db given (edge-restricted if
            edge_percentile also given); overshoot [sector] if ut_loc+
            distance_threshold_m+overlap_margin_db given; raw counts too if
            return_counts.
        """
        logger.function("compute_ue_kpis start: power_w shape=%s", power_w.shape)

        # Resolve a swept table down to one tilt per sector first.
        if tilt_idx_per_sector is not None:
            idx = np.asarray(tilt_idx_per_sector, dtype=int).reshape(1, 1, -1, 1)
            idx = np.broadcast_to(idx, (1,) + power_w.shape[1:])
            power_w = np.take_along_axis(power_w, idx, axis=0)[0]

        num_sectors = power_w.shape[1]

        # Shared signal/interference/SINR core
        core = self._compute_serving_sinr(power_w, sector_axis=1)
        serving_idx = core["serving_idx"]

        # Reported SINR
        sinr_db = np.take_along_axis(
            core["sinr_db"], serving_idx[:, None, :], axis=1
        ).squeeze(1)

        # RSRP
        best_power_db = np.take_along_axis(
            core["power_db"], serving_idx[:, None, :], axis=1
        ).squeeze(1)
        rsrp_dbm = best_power_db + 30.0

        # Aggregated KPIs
        result = {
            "sinr_db": sinr_db,
            "rsrp_dbm": rsrp_dbm,
            "serving_idx": serving_idx,
        }

        # --------------------------------------------------------------
        # Overall network coverage
        # --------------------------------------------------------------
        if threshold_db is not None:
            is_covered = sinr_db > threshold_db
            result["coverage"] = float(np.mean(is_covered))
            logger.debug("compute_ue_kpis: overall coverage=%.4f", result["coverage"])

        # Stop early if UE positions are not available
        if ut_loc is None:
            logger.function("compute_ue_kpis end: no ut_loc, early return")
            return result

        # UE-to-site distance
        dist_to_site = np.linalg.norm(
            ut_loc[:, None, :, :2] - self.bs_xy[None, :, None, :],
            axis=-1,
        )

        # [batch, sector, ue] -- every sector's own power, not just serving
        power_db_full = core["power_db"]

        # --------------------------------------------------------------
        # Initialization
        # --------------------------------------------------------------
        if threshold_db is not None:
            per_sector_coverage = np.full(num_sectors, np.nan)
            served_count = np.zeros(num_sectors)
            covered_count = np.zeros(num_sectors)

        compute_overshoot = (
            distance_threshold_m is not None
            and overlap_margin_db is not None
        )

        if compute_overshoot:
            overshoot = np.full(num_sectors, np.nan)
            neighbor_served_count = np.zeros(num_sectors)
            overshoot_hit_count = np.zeros(num_sectors)

            if return_per_neighbor:
                per_neighbor_served = np.zeros(
                    (num_sectors, self.max_neighbors)
                )
                per_neighbor_hit = np.zeros(
                    (num_sectors, self.max_neighbors)
                )

        logger.debug("compute_ue_kpis: num_sectors=%d compute_overshoot=%s return_per_neighbor=%s",
                    num_sectors, compute_overshoot, return_per_neighbor)

        # --------------------------------------------------------------
        # One main loop over sectors
        # --------------------------------------------------------------
        for i in range(num_sectors):

            # ==========================================================
            # Per-sector coverage
            # ==========================================================
            if threshold_db is not None:

                # List of served UEs by the sector
                served_by_i = serving_idx == i

                if served_by_i.any():

                    # whole population SINR of the sector
                    sinr_i = sinr_db[served_by_i]

                    # edge restricted SINR of the sector
                    if edge_percentile is not None:
                        dist_i = dist_to_site[:, i, :][served_by_i]
                        edge_threshold = np.percentile(
                            dist_i, edge_percentile
                        )
                        edge_mask = dist_i >= edge_threshold
                        sinr_i = sinr_i[edge_mask]

                    # coverage of the sector
                    if sinr_i.size > 0:
                        per_sector_coverage[i] = np.mean(
                            sinr_i > threshold_db
                        )
                        served_count[i] = sinr_i.size
                        covered_count[i] = np.sum(
                            sinr_i > threshold_db
                        )

            # ==========================================================
            # Overshoot
            # ==========================================================
            if compute_overshoot:

                # Compare power gap between serving sector and sector i
                power_gap_db = (
                    best_power_db - power_db_full[:, i, :]
                )

                # Check if sector i power is sufficiently close to serving sector
                comparable = (
                    power_gap_db <= overlap_margin_db
                )

                # Check if the UE is far from sector i
                is_far = (
                    dist_to_site[:, i, :] > distance_threshold_m
                )

                # Combined overshoot condition for sector i
                overshoot_condition = comparable & is_far

                # Find neighboring sectors of sector i
                neighbor_ids = np.where(
                    self.sector_adjacency[i]
                )[0]

                # List of UEs served by the neighboring sectors of sector i
                served_by_neighbor = np.isin(
                    serving_idx, neighbor_ids
                )

                # ======================================================
                # Overall sector overshoot
                # ======================================================
                if served_by_neighbor.any():

                    # If both conditions are met, overshoot happens
                    hits = overshoot_condition[served_by_neighbor]

                    # Overall sector overshoot
                    overshoot[i] = np.mean(hits)
                    neighbor_served_count[i] = hits.size
                    overshoot_hit_count[i] = np.sum(hits)

                # ======================================================
                # Per-neighbor overshoot
                # ======================================================
                if return_per_neighbor:

                    # for each neighbor
                    for k in range(self.max_neighbors):

                        # Find neighboring sector k of sector i
                        j = self.neighbor_ids[i, k]

                        if j < 0:
                            continue

                        # List of UEs served by neighboring sector k of sector i
                        served_by_j = serving_idx == j

                        if not served_by_j.any():
                            continue

                        # If both conditions are met, overshoot happens
                        hits = overshoot_condition[served_by_j]

                        # Per neighbor overshoot counts
                        per_neighbor_served[i, k] = hits.size
                        per_neighbor_hit[i, k] = np.sum(hits)

        logger.debug("compute_ue_kpis: main loop over %d sectors done", num_sectors)

        # --------------------------------------------------------------
        # Save per-sector coverage results
        # --------------------------------------------------------------
        if threshold_db is not None:
            result["per_sector_coverage"] = per_sector_coverage

            if return_counts:
                result["per_sector_served_count"] = served_count
                result["per_sector_covered_count"] = covered_count
            logger.debug("compute_ue_kpis: per_sector_coverage mean=%.4f",
                        float(np.nanmean(per_sector_coverage)))

        # --------------------------------------------------------------
        # Save overshoot results
        # --------------------------------------------------------------
        if compute_overshoot:
            result["overshoot"] = overshoot

            if return_counts:
                result["neighbor_served_count"] = neighbor_served_count
                result["overshoot_hit_count"] = overshoot_hit_count

            # Save per neighbor overshoot results
            if return_per_neighbor:
                result["per_neighbor_served_count"] = per_neighbor_served
                result["per_neighbor_overshoot_hit_count"] = per_neighbor_hit
            logger.debug("compute_ue_kpis: overshoot mean=%.4f", float(np.nanmean(overshoot)))

        logger.function("compute_ue_kpis end")
        return result

class IntervalMeasurementPool:
    """Pools KpiManager's per-slot output against the shared TILT-CONTROL
    INTERVAL boundary -- used to score Oracle/Causal/Adaptive Legacy/No
    Tilt/DRL's own per-interval logging, all against the same pooled
    snapshot. (DRL's own reward/state accumulation, against its own
    per-sector staggered window rather than this shared boundary, is
    SpatialGridEstimator's job -- see helpers/spatial_grid_estimator.py --
    not this class's.)

    :param kpi_manager: resolves power at a given tilt; this class never
        mutates sector_etilts directly.
    """

    def __init__(self, kpi_manager: KpiManager, num_sectors: int):
        self.kpi_manager = kpi_manager
        self.num_sectors = num_sectors
        self.start_interval()

    # ------------------------------------------------------- interval pool

    def start_interval(self):
        """Reset accumulators for a new tilt-control interval -- no
        retention beyond the interval currently being pooled."""
        logger.function("IntervalMeasurementPool.start_interval")
        self._table_chunks = []
        self._adaptive_r0_chunks = []
        self._drl_r0_chunks = []
        self._no_tilt_chunks = []
        self._ut_loc_r0_chunks = []

    def pool_measurement(self, state: LargeScaleState, adaptive_tilt_deg, drl_tilt_deg,
                         downtilt_sweep_deg, ut_loc_r0=None, needs_sweep: bool = True):
        """One measurement draw's contribution: the swept table (Oracle/
        Causal) plus Adaptive Legacy/DRL/No Tilt's own current-tilt power,
        appended to this interval's running pool.

        :param state: this draw's large-scale state (one realization-chunk
            -- call once per chunk if num_realizations_per_slot exceeds a
            single channel batch)
        :param adaptive_tilt_deg, drl_tilt_deg: [num_sectors] current tilt [deg]
        :param downtilt_sweep_deg: candidate tilts [deg] for the swept table
        :param ut_loc_r0: [num_ue, 3] realization-0's UE positions this draw
            -- only given on the chunk containing realization 0
        :param needs_sweep: if False, skips building the swept
            [tilt, sector, ue] table -- only Dynamic Local Oracle/Causal's
            coordinate-ascent search needs it
        :output: this draw's resolved DRL power, [sector, ue] (realization
            0's slice, batch dim dropped) -- lets a caller reuse it (e.g.
            for a per-sector rolling-window accumulator) without resolving
            the same power a second time.
        """
        logger.function("pool_measurement start: needs_sweep=%s", needs_sweep)
        if needs_sweep:
            self._table_chunks.append(self.kpi_manager.compute_tilt_sector_ue_rx_power(state, downtilt_sweep_deg))

        adaptive_power_w = self.kpi_manager.resolve_power_at_tilt(state, adaptive_tilt_deg)
        drl_power_w = self.kpi_manager.resolve_power_at_tilt(state, drl_tilt_deg)

        if ut_loc_r0 is not None:
            self._adaptive_r0_chunks.append(adaptive_power_w[0])  # -> [sector, ue]
            self._drl_r0_chunks.append(drl_power_w[0])
            self._ut_loc_r0_chunks.append(ut_loc_r0)

        no_tilt_power_w = self.kpi_manager.resolve_power_at_tilt(state, np.zeros(self.num_sectors))
        self._no_tilt_chunks.append(no_tilt_power_w)

        logger.function("pool_measurement end")
        return drl_power_w[0]

    def pooled_interval(self) -> dict:
        """Concatenates this interval's accumulated measurement draws into
        the pooled result. Call start_interval() again before the next one.
        """
        logger.function("pooled_interval start")
        pooled = {
            "power_table": np.concatenate(self._table_chunks, axis=1) if self._table_chunks else None,
            "adaptive_power_w_r0": np.concatenate(self._adaptive_r0_chunks, axis=1),
            "drl_power_w_r0": np.concatenate(self._drl_r0_chunks, axis=1),
            "no_tilt_power_w": np.concatenate(self._no_tilt_chunks, axis=0),
            "ut_loc_r0": np.concatenate(self._ut_loc_r0_chunks, axis=0),
        }
        logger.debug("pooled_interval: drl_power_w_r0 shape=%s", pooled["drl_power_w_r0"].shape)
        logger.function("pooled_interval end")
        return pooled
