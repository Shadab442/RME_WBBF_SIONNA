"""Per-sector spatial grid accumulator for the local RL state/reward: 
for every cell in a sector's own observation region X_s, 
tracks smoothed occupancy rho_x and coverage c_x (Kriged where unvisited). 
The same per-cell (visit, covered) counts this window also directly give the 
sector's own local reward -- aggregate over its own cells, no separate accumulator needed.

Two interchangeable radio_map_method options for filling in coverage at
UNVISITED cells:
  "sinr_direct"      -- Krige the observed SINR [dB] directly (a continuous
                        physical quantity, unlike a bounded coverage
                        fraction), then threshold the prediction.
  "rsrp_per_sector"  -- Krige each of the num_bs sectors' own RSRP [dB] map
                        separately, then reconstruct SINR from ALL sectors'
                        predictions.
"""

import math

import numpy as np

from algorithm.cgr import CGRRadioMapEstimator
from algorithm.kriging import OrdinaryKriging
from helpers.utils import get_logger

logger = get_logger(__name__)


class SpatialGridEstimator:
    """
    :ivar visit_count: [num_bs, num_grid_cells] this turn's UE-occupancy
        count -- always reset to 0 for every sector at the end of every
        compute() call (exactly one measurement is taken per turn; see
        helpers/regional_info_provider.py's module docstring for what a
        "turn" is under sync vs async scheduling).
    :ivar covered_count: [num_bs, num_grid_cells] this turn's covered-UE
        count, same reset.
    """

    def __init__(self, num_bs: int, num_grid_cells: int, grid_points: np.ndarray,
                length_scale: float, coverage_threshold_db: float,
                radio_map_method: str = "sinr_direct", noise_power_w: float = None,
                variance: float = 1.0,
                topology=None, radio_map_estimator: str = "kriging",
                cgr_num_gaussian: int = 12, cgr_ridge_lambda: float = 1e-4,
                cgr_gaussian_config: dict = None, cgr_baseline_method: str = "global_mean",
                cgr_baseline_length_scale_m: float = 50.0):
        """
        :param grid_points: [num_bs, num_grid_cells, 2] fixed world (x, y)
            cell centers, from CellularTopology.build_sector_rhombus_grid --
            Kriging's/CGR's target/known coordinates.
        :param length_scale, variance: OrdinaryKriging's RBF covariance
            parameters for filling in coverage at unvisited cells.
        :param coverage_threshold_db: SINR threshold -- both methods Krige a
            continuous quantity (SINR or RSRP) and threshold it into a
            binary coverage prediction at unvisited cells.
        :param radio_map_method: "sinr_direct" or "rsrp_per_sector" -- see
            module docstring.
        :param noise_power_w: required for "rsrp_per_sector" (SINR
            reconstruction needs the same noise floor KpiManager uses).
        :param topology: a CellularTopology -- required when
            radio_map_estimator="cgr" (for each sector's own bs_loc/
            azimuth); unused for "kriging".
        :param radio_map_estimator: "kriging" (default, existing baseline,
            always available) or "cgr" (algorithm/cgr.py's
            CGRRadioMapEstimator, ONLY used inside the rsrp_per_sector
            branch's per-transmitting-sector prediction -- sinr_direct
            always uses Kriging, since there's no single sector to
            attribute an already-combined SINR field to).
        :param cgr_num_gaussian, cgr_ridge_lambda, cgr_gaussian_config,
            cgr_baseline_method, cgr_baseline_length_scale_m:
            CGRRadioMapEstimator hyperparameters, used only when
            radio_map_estimator="cgr" -- see its own __init__ docstring.
        """
        assert radio_map_method in ("sinr_direct", "rsrp_per_sector"), \
            "radio_map_method must be 'sinr_direct' or 'rsrp_per_sector'"
        assert radio_map_method != "rsrp_per_sector" or noise_power_w is not None, \
            "rsrp_per_sector needs noise_power_w to reconstruct SINR"
        assert radio_map_estimator in ("kriging", "cgr"), \
            "radio_map_estimator must be 'kriging' or 'cgr'"
        assert radio_map_estimator != "cgr" or topology is not None, \
            "radio_map_estimator='cgr' needs topology (for bs_loc/bs_orientations)"

        self.num_bs = num_bs
        self.num_grid_cells = num_grid_cells
        self.grid_points = grid_points
        self.coverage_threshold_db = coverage_threshold_db
        self.radio_map_method = radio_map_method
        self.noise_power_w = noise_power_w
        self.radio_map_estimator = radio_map_estimator
        # Cells within min_bs_ut_dist of ANY site are never estimated
        # (Kriging or CGR) -- a real UE can never be dropped there (see
        # helpers/ue_drop.py's own min_dist_from_site enforcement), so
        # they're NEVER visited and Krige/CGR would be extrapolating a
        # value for a location no measurement will ever exist for. Left
        # at the same -1 "not yet estimated" sentinel `compute()` already
        # uses, not force-computed. Defaults to "every cell estimable"
        # when topology isn't given (kriging-only callers that don't care
        # about this refinement, e.g. tests) -- no behavior change for them.
        if topology is not None:
            site_loc = topology.site_loc.detach().cpu().numpy()  # [num_cells, 2]
            dist_to_nearest_site = np.linalg.norm(
                grid_points[:, :, None, :] - site_loc[None, None, :, :], axis=-1).min(axis=-1)
            self.estimable_cell_mask = dist_to_nearest_site >= float(topology.min_bs_ut_dist)
        else:
            self.estimable_cell_mask = np.ones((num_bs, num_grid_cells), dtype=bool)
        # Kriging stays available as a baseline regardless of
        # radio_map_estimator -- see module/class docstring.
        self.kriging = OrdinaryKriging(length_scale, variance)
        if radio_map_estimator == "cgr":
            self.bs_loc = topology.bs_loc[0, :, :].detach().cpu().numpy()  # [num_bs, 3]
            self.bs_azimuths_rad = topology.bs_orientations[0, :, 0].detach().cpu().numpy()  # [num_bs]
            self.cgr = CGRRadioMapEstimator(num_gaussian=cgr_num_gaussian, ridge_lambda=cgr_ridge_lambda,
                                           gaussian_config=cgr_gaussian_config,
                                           baseline_method=cgr_baseline_method,
                                           baseline_length_scale_m=cgr_baseline_length_scale_m)

        self.visit_count = np.zeros((num_bs, num_grid_cells))
        self.covered_count = np.zeros((num_bs, num_grid_cells))

        if radio_map_method == "sinr_direct":
            self.sinr_db_sum = np.zeros((num_bs, num_grid_cells))
        else:
            # [transmitting sector, geographic grid sector, cell]
            self.rsrp_db_sum = np.zeros((num_bs, num_bs, num_grid_cells))

    def accumulate(self, sector_idx: np.ndarray, cell_idx: np.ndarray, covered_ue: np.ndarray,
                   sinr_db: np.ndarray = None, power_db_all_sectors: np.ndarray = None) -> None:
        """Accumulate one measurement slot's occupancy/coverage. Each UE
        contributes one count to the geographic grid cell it occupies --
        multiple UEs in the same cell are each counted individually.

        :param sector_idx, cell_idx: [num_ue] geographic grid assignment,
            from CellularTopology.assign_to_sector_rhombus_grid -- -1 where
            a UE falls outside every sector's grid (excluded below).
        :param covered_ue: [num_ue] bool, this slot's per-UE coverage indicator.
        :param sinr_db: [num_ue] float, this slot's per-UE serving SINR --
            required for radio_map_method="sinr_direct".
        :param power_db_all_sectors: [num_bs, num_ue] float, every sector's
            own received power at every UE -- required for
            radio_map_method="rsrp_per_sector".
        """
        logger.function("SpatialGridEstimator.accumulate start")
        assert self.radio_map_method != "sinr_direct" or sinr_db is not None, \
            "sinr_direct needs sinr_db"
        assert self.radio_map_method != "rsrp_per_sector" or power_db_all_sectors is not None, \
            "rsrp_per_sector needs power_db_all_sectors"

        resolved = (sector_idx >= 0) & (cell_idx >= 0) & (cell_idx < self.num_grid_cells)
        np.add.at(self.visit_count, (sector_idx[resolved], cell_idx[resolved]), 1.0)
        np.add.at(self.covered_count, (sector_idx[resolved], cell_idx[resolved]),
                  covered_ue[resolved].astype(float))
        if self.radio_map_method == "sinr_direct":
            np.add.at(self.sinr_db_sum, (sector_idx[resolved], cell_idx[resolved]), sinr_db[resolved])
        else:
            for b in range(self.num_bs):
                np.add.at(self.rsrp_db_sum[b], (sector_idx[resolved], cell_idx[resolved]),
                         power_db_all_sectors[b, resolved])
        logger.debug("SpatialGridEstimator.accumulate: resolved=%d/%d", int(resolved.sum()), resolved.size)
        logger.function("SpatialGridEstimator.accumulate end")

    def _cgr_context(self, b: int, i: int, own_tilt_deg: float) -> dict:
        """Builds algorithm/cgr.py's context dict for predicting
        transmitting sector b's own RSRP field over geographic sector i's
        grid -- own geometry only (see CGRRadioMapEstimator.
        _initialize_gaussians's own docstring for why neighboring tilts
        are never needed here).

        :param own_tilt_deg: sector b's OWN current electrical downtilt [deg].
        """
        cell_min = self.grid_points[i].min(axis=0)
        cell_max = self.grid_points[i].max(axis=0)
        receiver_height = 1.5
        return {
            "tx_position": self.bs_loc[b].tolist(),
            "own_azimuth_deg": math.degrees(float(self.bs_azimuths_rad[b])),
            "own_tilt_deg": float(own_tilt_deg),
            "receiver_height": receiver_height,
            # Tight bounding box of geo-sector i's own FULL grid (known
            # ahead of time, not just the measured points) -- z-row set
            # tight around receiver_height, since every known/target point
            # CGRRadioMapEstimator builds sits on that one height plane.
            "env_dims": [[float(cell_min[0]), float(cell_max[0])],
                        [float(cell_min[1]), float(cell_max[1])],
                        [receiver_height - 1.0, receiver_height + 1.0]],
        }

    def infer_region_probabilities(self, ue_sector: np.ndarray, ue_measured_rsrp_db: np.ndarray,
                                   cell_to_region: np.ndarray, num_regions: int,
                                   sigma_db: float, radio_map_estimator: str = "kriging",
                                   current_tilts_deg: np.ndarray = None) -> np.ndarray:
        """Region-membership probabilities for UEs whose exact location
        isn't known (non-anchor UEs -- see SimulationEngine.
        location_reporting_mask) -- inferred by comparing each UE's own
        measured RSRP AT its serving sector against THIS TURN's anchor-fit
        RSRP field (Kriging or CGR) over that sector's own fine grid, never
        the UE's real position (which is exactly what's unknown here).

        Must be called AFTER accumulate() for this turn (uses this turn's
        own self.visit_count/self.rsrp_db_sum -- the SAME known-point data
        compute()'s own coverage-filling uses) and BEFORE it's reset.

        :param ue_sector: [num_ue] int -- each UE's own geographic/serving
            sector (CellularTopology.assign_to_sector_rhombus_grid's own
            sector_idx). This IS known even for a non-reporting UE --
            which sector serves a UE is ordinary connectivity, not the
            precise position that's actually unknown.
        :param ue_measured_rsrp_db: [num_ue] float -- each UE's own real
            measured RSRP at its serving sector -- never used to reveal
            WHERE in that sector, only compared against the field.
        :param cell_to_region: [num_grid_cells] int -- which of
            num_regions each fine cell belongs to (OccupancyEstimator.
            cell_to_region / RegionalInfoProvider.cell_to_region).
        :param sigma_db: Gaussian-kernel bandwidth for the measured-vs-
            predicted match -- calibrate empirically against this
            scenario's own actual Kriging/CGR residual scale, not guessed.
        :param radio_map_estimator: "kriging" (default -- production's own
            choice, a self-contained fit needing no extra context) or
            "cgr" (this instance's own self.cgr -- only for debug/
            validation tooling, e.g. scripts/debug_pipeline_audit.py's
            phase 5, comparing both estimators' own region-inference
            quality; requires current_tilts_deg).
        :param current_tilts_deg: [num_bs] -- required (asserted) only for
            radio_map_estimator="cgr" (CGR's own tilt-aware Gaussian
            placement needs the transmitting sector's current tilt).
        :output: [num_ue, num_regions], each row summing to 1 -- uniform
            for a UE whose own sector got fewer than 2 anchor points this
            turn (nothing to fit from yet).
        """
        assert radio_map_estimator != "cgr" or current_tilts_deg is not None, \
            "radio_map_estimator='cgr' needs current_tilts_deg"
        num_ue = len(ue_sector)
        probs = np.full((num_ue, num_regions), 1.0 / num_regions)
        for s in np.unique(ue_sector):
            known = self.visit_count[s] > 0
            if known.sum() < 2:
                continue
            known_xy = self.grid_points[s, known]
            known_values = self.rsrp_db_sum[s, s, known] / self.visit_count[s, known]
            target_cells = np.where(self.estimable_cell_mask[s])[0]
            if radio_map_estimator == "cgr":
                context = self._cgr_context(s, s, current_tilts_deg[s])
                predicted = self.cgr.predict(known_xy, known_values, self.grid_points[s, target_cells],
                                             context=context)
            else:
                predicted = self.kriging.predict(known_xy, known_values, self.grid_points[s, target_cells])

            ue_mask = ue_sector == s
            measured = ue_measured_rsrp_db[ue_mask][:, None]  # [n_ue_s, 1]
            likelihood = np.exp(-((measured - predicted[None, :]) ** 2) / (2.0 * sigma_db ** 2))
            total = likelihood.sum(axis=1, keepdims=True)
            cell_probs = np.where(total > 0, likelihood / np.where(total > 0, total, 1.0),
                                  1.0 / target_cells.size)

            region_of_target = cell_to_region[target_cells]
            region_probs = np.zeros((int(ue_mask.sum()), num_regions))
            for r in range(num_regions):
                region_probs[:, r] = cell_probs[:, region_of_target == r].sum(axis=1)
            probs[ue_mask] = region_probs
        return probs

    def reset(self) -> None:
        """Clears this turn's accumulated (visit, covered, RSRP/SINR sum)
        state with NO prediction math -- for a caller that only ever needs
        infer_region_probabilities' Kriging fit (never compute()'s own
        per-cell coverage prediction, e.g. SimulationEngine's production
        instance since 2026-09-20 -- see RegionalInfoProvider's own module
        docstring), so it isn't stuck paying compute()'s full Kriging/CGR
        coverage-prediction cost every interval just to reset for the next
        one. compute() and predict_full_rsrp_sinr are unaffected -- e.g.
        scripts/debug_pipeline_audit.py's own standalone Kriging/CGR
        validation still uses those directly.
        """
        self.visit_count[:] = 0.0
        self.covered_count[:] = 0.0
        if self.radio_map_method == "sinr_direct":
            self.sinr_db_sum[:] = 0.0
        else:
            self.rsrp_db_sum[:] = 0.0

    def compute(self, closes: np.ndarray, current_tilts_deg: np.ndarray = None) -> tuple:
        """Resolves the just-accumulated turn: this turn's raw instantaneous
        per-cell occupancy (no smoothing across turns -- EMA removed
        2026-09-20, see helpers/regional_info_provider.py's module
        docstring for why; this return value is unused by every current
        caller anyway, see the `_` discard at both call sites), per-cell
        coverage (Kriged/CGR-filled where unvisited), and each sector's own
        local reward. Only `closes` sectors' reward persists (see
        helpers/regional_info_provider.py's module docstring for what a
        "turn"/`closes` means); raw per-turn counters always reset.

        :param current_tilts_deg: [num_bs] this turn's live electrical
            downtilt per sector [deg] -- required (asserted) only when
            radio_map_estimator="cgr" (rsrp_per_sector's tilt-aware
            Gaussian placement needs each transmitting sector's OWN
            current tilt); ignored otherwise.
        :output: (occupancy [num_bs, num_grid_cells], coverage
            [num_bs, num_grid_cells], reward [num_bs] -- NaN for a sector
            with zero visits this turn, i.e. no local UEs to score).
        """
        logger.function("SpatialGridEstimator.compute start: closes=%d", int(closes.sum()))
        assert self.radio_map_estimator != "cgr" or current_tilts_deg is not None, \
            "radio_map_estimator='cgr' needs current_tilts_deg"

        # This turn's occupancy share of network-wide visits, raw.
        total_visits_network = float(self.visit_count.sum())
        denom = total_visits_network if total_visits_network > 0 else 1.0
        occupancy = self.visit_count / denom

        # c_x[k]: instantaneous coverage where visited (a real empirical
        # observation either way); where unvisited, Krige a continuous
        # physical quantity and threshold it -- see radio_map_method.
        visited = self.visit_count > 0
        with np.errstate(invalid="ignore"):
            coverage_instant = np.where(visited, self.covered_count / np.where(visited, self.visit_count, 1), -1.0)
        coverage = coverage_instant.copy()
        for i in np.where(closes)[0]:
            known = visited[i]
            if not known.any():
                logger.warning("SpatialGridEstimator.compute: sector=%d has zero visits this window "
                              "-- coverage left at -1 for every cell", i)
                continue
            # Only cells a real UE could actually occupy get Kriged/CGR-
            # predicted -- cells inside min_bs_ut_dist of every site stay
            # at coverage_instant's own -1 sentinel (see
            # self.estimable_cell_mask's own docstring above).
            unknown = ~known & self.estimable_cell_mask[i]
            if not unknown.any():
                continue
            if self.radio_map_method == "sinr_direct":
                mean_sinr_db = np.where(known, self.sinr_db_sum[i] / np.where(known, self.visit_count[i], 1), 0.0)
                predicted_sinr_db = self.kriging.predict(self.grid_points[i, known], mean_sinr_db[known],
                                                         self.grid_points[i, unknown])
                coverage[i, unknown] = (predicted_sinr_db > self.coverage_threshold_db).astype(float)
            else:
                predicted_rsrp_db = np.zeros((self.num_bs, int(unknown.sum())))
                for b in range(self.num_bs):
                    mean_rsrp_db_b = np.where(known, self.rsrp_db_sum[b, i] / np.where(known, self.visit_count[i], 1), 0.0)
                    if self.radio_map_estimator == "cgr":
                        context = self._cgr_context(b, i, current_tilts_deg[b])
                        predicted_rsrp_db[b] = self.cgr.predict(self.grid_points[i, known], mean_rsrp_db_b[known],
                                                               self.grid_points[i, unknown], context=context)
                    else:
                        predicted_rsrp_db[b] = self.kriging.predict(self.grid_points[i, known], mean_rsrp_db_b[known],
                                                                    self.grid_points[i, unknown])
                # RSRP-based attachment: the SERVER is whichever sector's
                # predicted power is strongest at each cell, not always the
                # geographic sector i (a neighbor can legitimately win a
                # boundary/overlap cell) -- matches KpiManager's own
                # argmax-power attachment rule.
                predicted_rsrp_w = np.power(10.0, predicted_rsrp_db / 10.0)
                serving_w = predicted_rsrp_w.max(axis=0)
                interference_w = predicted_rsrp_w.sum(axis=0) - serving_w
                predicted_sinr_db = 10.0 * np.log10(serving_w / (interference_w + self.noise_power_w))
                coverage[i, unknown] = (predicted_sinr_db > self.coverage_threshold_db).astype(float)

        # Local reward: this window's coverage aggregated over the sector's
        # OWN cells -- the same counts already accumulated above, not a
        # separate served-based accumulator.
        sector_visits = self.visit_count.sum(axis=1)
        with np.errstate(invalid="ignore"):
            reward = np.where(sector_visits > 0, self.covered_count.sum(axis=1) / np.where(sector_visits > 0, sector_visits, 1), np.nan)

        self.visit_count[:] = 0.0
        self.covered_count[:] = 0.0
        if self.radio_map_method == "sinr_direct":
            self.sinr_db_sum[:] = 0.0
        else:
            self.rsrp_db_sum[:] = 0.0
        logger.debug("SpatialGridEstimator.compute: mean occupancy=%.4f mean coverage=%.4f mean reward=%.4f",
                    float(occupancy[closes].mean()) if closes.any() else float("nan"),
                    float(np.nanmean(coverage[closes])) if closes.any() else float("nan"),
                    float(np.nanmean(reward[closes])) if closes.any() else float("nan"))
        logger.function("SpatialGridEstimator.compute end")
        return occupancy, coverage, reward

    def predict_full_rsrp_sinr(self, current_tilts_deg: np.ndarray = None) -> tuple:
        """Continuous RSRP/SINR field at EVERY cell (known + unknown),
        reusing compute()'s own per-sector-i prediction math -- known
        cells get their real accumulated mean, unknown cells get the
        Kriged/CGR prediction. Unlike compute(), this is READ-ONLY: it
        never resets visit_count/rsrp_db_sum/sinr_db_sum, so it can be
        called any number of times after accumulate() (e.g. once for a
        coverage map, once for an RSRP map) without disturbing the turn's
        own compute() call.

        Added for scripts/debug_pipeline_audit.py's ground-truth vs.
        anchor-only vs. RME-assisted comparison figures (2026-09-20) --
        compute() only ever returns the thresholded binary coverage
        decision, never the continuous field it's derived from.

        :param current_tilts_deg: same as compute()'s own param -- required
            only for radio_map_estimator="cgr".
        :output: (rsrp_db [num_bs, num_bs, num_grid_cells] -- rsrp_db[b, i, :]
            is sector b's own predicted RSRP over sector i's grid;
            sinr_db [num_bs, num_grid_cells] -- each sector's own serving-cell
            SINR, RSRP-based best-server attachment, same rule KpiManager
            uses).
        """
        assert self.radio_map_estimator != "cgr" or current_tilts_deg is not None, \
            "radio_map_estimator='cgr' needs current_tilts_deg"
        visited = self.visit_count > 0
        rsrp_db_full = np.zeros((self.num_bs, self.num_bs, self.num_grid_cells))
        for i in range(self.num_bs):
            known = visited[i]
            unknown = ~known
            for b in range(self.num_bs):
                mean_rsrp_db_b = np.where(known, self.rsrp_db_sum[b, i] / np.where(known, self.visit_count[i], 1), 0.0)
                rsrp_db_full[b, i, known] = mean_rsrp_db_b[known]
                if not unknown.any():
                    continue
                if self.radio_map_estimator == "cgr":
                    context = self._cgr_context(b, i, current_tilts_deg[b])
                    rsrp_db_full[b, i, unknown] = self.cgr.predict(
                        self.grid_points[i, known], mean_rsrp_db_b[known], self.grid_points[i, unknown], context=context)
                else:
                    rsrp_db_full[b, i, unknown] = self.kriging.predict(
                        self.grid_points[i, known], mean_rsrp_db_b[known], self.grid_points[i, unknown])
        rsrp_w_full = np.power(10.0, rsrp_db_full / 10.0)
        own_rsrp_w = np.stack([rsrp_w_full[i, i] for i in range(self.num_bs)])  # [num_bs, num_grid_cells]
        # total received power AT sector i's own grid, summed over every
        # TRANSMITTING sector b, is rsrp_w_full[:, i, :].sum(axis=0).
        interference_w = np.zeros((self.num_bs, self.num_grid_cells))
        for i in range(self.num_bs):
            total_w_i = rsrp_w_full[:, i, :].sum(axis=0)
            interference_w[i] = total_w_i - own_rsrp_w[i]
        sinr_db = 10.0 * np.log10(own_rsrp_w / (interference_w + self.noise_power_w))
        return rsrp_db_full, sinr_db
