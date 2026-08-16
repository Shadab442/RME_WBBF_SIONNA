"""Per-UE downlink KPIs (SINR, RSRP, coverage, overshoot) across candidate
sectors, plus within-interval pooling across measurement draws.

Large-scale only: pathloss + shadow fading, no fast/frequency-selective
fading.
"""

from typing import List

import numpy as np
import torch

from .electrical_downtilt import ElectricalDowntilt
from .large_scale_channel import LargeScaleState


class KpiManager:
    """
    :param sector_etilts: One :class:`~helpers.electrical_downtilt.ElectricalDowntilt`
        per sector, in the same order as the channel model's tx (BS) dimension.
    :param tx_power_w: Total sector transmit power [W].
    :param noise_power_w: Total-bandwidth noise power [W].
    :param bs_xy: [num_sectors, 2] per-sector (x, y) position -- needed for
        overshoot/edge-restricted coverage's UE-to-site distance.
    """

    def __init__(self, sector_etilts: List[ElectricalDowntilt], tx_power_w: float,
                noise_power_w: float, bs_xy):
        self.sector_etilts = sector_etilts
        self.tx_power_w = tx_power_w
        self.noise_power_w = noise_power_w
        self.bs_xy = bs_xy.detach().cpu().numpy() if hasattr(bs_xy, "detach") else np.asarray(bs_xy)
        self.start_interval()

    def _rx_power_now(self, state: LargeScaleState) -> np.ndarray:
        """One tilt snapshot at every sector's CURRENT tilt, [batch, sector, ue], numpy."""
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
        return power.detach().cpu().numpy()

    def compute_tilt_sector_ue_rx_power(self, state: LargeScaleState, tilt_values=None) -> np.ndarray:
        """Received power [W], numpy. [batch, sector, ue] at current tilts,
        or [tilt, batch, sector, ue] if tilt_values given.

        :param state: A large-scale state (see helpers.large_scale_channel)
        :param tilt_values: candidate downtilts [deg], optional
        """
        if tilt_values is None:
            return self._rx_power_now(state)

        # Sweep: every sector set to the SAME candidate tilt per row.
        rows = []
        for downtilt_deg in tilt_values:
            for etilt in self.sector_etilts:
                etilt.set_tilt(downtilt_deg)
            rows.append(self._rx_power_now(state))
        return np.stack(rows, axis=0)

    def _compute_serving_sinr(self, power_w: np.ndarray, sector_axis: int = 1) -> dict:
        """Signal/interference/SINR (every candidate server's, not just the
        best) and per-sector power in dB -- the shared core compute_ue_kpis
        builds on.

        :param power_w: received power [W], sector along sector_axis
        :output: dict with sinr_db, power_db (both power_w's shape),
            serving_idx (sector_axis removed) -- the argmax-SINR sector
        """
        total_w = power_w.sum(axis=sector_axis, keepdims=True)
        interference_w = total_w - power_w
        sinr_db = 10.0 * np.log10(power_w / (interference_w + self.noise_power_w))
        power_db = 10.0 * np.log10(power_w)
        serving_idx = sinr_db.argmax(axis=sector_axis)
        return {"sinr_db": sinr_db, "power_db": power_db, "serving_idx": serving_idx}

    def compute_ue_kpis(self, power_w: np.ndarray, tilt_idx_per_sector=None,
                        threshold_db: float = None, weights: np.ndarray = None,
                        ut_loc: np.ndarray = None, edge_percentile: float = None,
                        distance_threshold_m: float = None, overlap_margin_db: float = None) -> dict:
        """Per-UE SINR/RSRP/serving-sector, plus coverage/overshoot, from a
        resolved power array. 

        :param power_w: [batch, sector, ue], or [tilt, batch, sector, ue]
            with tilt_idx_per_sector given
        :param tilt_idx_per_sector: [num_sectors] int array/list, optional
        :param threshold_db: also returns coverage if given
        :param weights: [num_samples] optional per-sample weight, for coverage
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
        :output: dict -- sinr_db, rsrp_dbm, serving_idx [batch,ue] always;
            coverage [scalar] if threshold_db given; per_sector_coverage
            [sector] if ut_loc+threshold_db given (edge-restricted if
            edge_percentile also given); overshoot [sector] if ut_loc+
            distance_threshold_m+overlap_margin_db given.
        """
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
        sinr_db = np.take_along_axis(core["sinr_db"], serving_idx[:, None, :], axis=1).squeeze(1)

        # RSRP
        best_power_db = np.take_along_axis(core["power_db"], serving_idx[:, None, :], axis=1).squeeze(1)
        rsrp_dbm = best_power_db + 30.0

        # Aggregated KPIs
        result = {"sinr_db": sinr_db, "rsrp_dbm": rsrp_dbm, "serving_idx": serving_idx}

        # Coverage
        if threshold_db is not None:
            is_covered = sinr_db > threshold_db
            result["coverage"] = (float(np.average(is_covered, weights=weights))
                                  if weights is not None else float(np.mean(is_covered)))

        if ut_loc is None:
            return result

        # UE-to-site distance, [batch, sector, ue] -- shared by per-sector
        # coverage's edge restriction and overshoot.
        dist_to_site = np.linalg.norm(ut_loc[:, None, :, :2] - self.bs_xy[None, :, None, :], axis=-1)
        power_db_full = core["power_db"]  # [batch, sector, ue] -- every sector's own power, not just serving

        # Per-sector coverage (whole population, or edge-restricted if edge_percentile given)
        if threshold_db is not None:
            per_sector_coverage = np.zeros(num_sectors)
            for i in range(num_sectors):
                served_by_i = serving_idx == i
                if not served_by_i.any():
                    continue
                sinr_i = sinr_db[served_by_i]
                if edge_percentile is not None:
                    dist_i = dist_to_site[:, i, :][served_by_i]
                    edge_threshold = np.percentile(dist_i, edge_percentile)
                    edge_mask = dist_i >= edge_threshold
                    sinr_i = sinr_i[edge_mask]
                if sinr_i.size > 0:
                    per_sector_coverage[i] = np.mean(sinr_i > threshold_db)
            result["per_sector_coverage"] = per_sector_coverage

        # Overshoot -- shared by Adaptive Legacy and DRL, no edge restriction.
        if distance_threshold_m is not None and overlap_margin_db is not None:
            overshoot = np.zeros(num_sectors)
            for i in range(num_sectors):
                served_elsewhere = serving_idx != i
                power_gap_db = best_power_db - power_db_full[:, i, :]
                comparable = served_elsewhere & (power_gap_db <= overlap_margin_db)
                is_far = dist_to_site[:, i, :] > distance_threshold_m
                overshoot[i] = np.mean(comparable & is_far)
            result["overshoot"] = overshoot

        return result

    # ---------------------------------------------------------- pooling

    def start_interval(self):
        """Reset accumulators for a new tilt-control interval -- no
        retention beyond the interval currently being pooled."""
        self._table_chunks = []
        self._adaptive_r0_chunks = []
        self._drl_r0_chunks = []
        self._no_tilt_chunks = []
        self._ut_loc_r0_chunks = []

    def pool_measurement(self, state: LargeScaleState, adaptive_tilt_deg, drl_tilt_deg,
                         downtilt_sweep_deg, ut_loc_r0=None):
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
        """
        self._table_chunks.append(self.compute_tilt_sector_ue_rx_power(state, downtilt_sweep_deg))

        adaptive_tilt_list = adaptive_tilt_deg.tolist() if hasattr(adaptive_tilt_deg, "tolist") else list(adaptive_tilt_deg)
        for etilt, tilt in zip(self.sector_etilts, adaptive_tilt_list):
            etilt.set_tilt(tilt)
        adaptive_power_w = self.compute_tilt_sector_ue_rx_power(state)  # [batch, sector, ue]

        drl_tilt_list = drl_tilt_deg.tolist() if hasattr(drl_tilt_deg, "tolist") else list(drl_tilt_deg)
        for etilt, tilt in zip(self.sector_etilts, drl_tilt_list):
            etilt.set_tilt(tilt)
        drl_power_w = self.compute_tilt_sector_ue_rx_power(state)

        if ut_loc_r0 is not None:
            self._adaptive_r0_chunks.append(adaptive_power_w[0])  # -> [sector, ue]
            self._drl_r0_chunks.append(drl_power_w[0])
            self._ut_loc_r0_chunks.append(ut_loc_r0)

        for etilt in self.sector_etilts:
            etilt.set_tilt(0.0)  # No Tilt baseline: fixed boresight
        no_tilt_power_w = self.compute_tilt_sector_ue_rx_power(state)
        self._no_tilt_chunks.append(no_tilt_power_w)

    def pooled_interval(self) -> dict:
        """Concatenates this interval's accumulated measurement draws into
        the pooled result. Call start_interval() again before the next one.
        """
        return {
            "power_table": np.concatenate(self._table_chunks, axis=1),
            "adaptive_power_w_r0": np.concatenate(self._adaptive_r0_chunks, axis=1),
            "drl_power_w_r0": np.concatenate(self._drl_r0_chunks, axis=1),
            "no_tilt_power_w": np.concatenate(self._no_tilt_chunks, axis=0),
            "ut_loc_r0": np.concatenate(self._ut_loc_r0_chunks, axis=0),
        }
