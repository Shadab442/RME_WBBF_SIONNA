"""
CGR-based radio-map estimator.

Purpose
-------
Replace OrdinaryKriging for CURRENT radio-map spatial estimation.

At every measurement interval:
    1. Build/initialize Gaussian basis using current network configuration
       and available domain knowledge.
    2. Fit W_out using ONLY measurements from the current interval.
    3. Predict the radio metric at unvisited target locations.

No temporal radio-map prediction.
No measurements from previous tilt configurations are reused.

Expected radio quantity:
    Prefer a continuous metric such as SINR / SS-SINR / RSRP / received power.
    If binary coverage is needed, threshold the predicted continuous map later.
"""

import numpy as np
import torch

from algorithm.cgr_model import CGR
from helpers.utils import get_logger

logger = get_logger(__name__)


class CGRRadioMapEstimator:
    """
    Spatial radio-map estimator using fixed Gaussian basis functions
    + ridge-regression output weights.

    External purpose is analogous to OrdinaryKriging:
        known positions + known values -> target-position estimates

    Gaussian parameters are rebuilt each measurement interval using
    physical/domain knowledge of the CURRENT radio configuration.
    W_out is fitted from current measurements only.
    """

    def __init__(
        self,
        num_gaussian: int,
        ridge_lambda: float = 1e-4,
        device: str = "cpu",
        gaussian_config: dict = None,
        normalize_columns_db: bool = False,
        baseline_method: str = "global_mean",
        baseline_length_scale_m: float = 50.0,
    ):
        """
        :param normalize_columns_db: column-normalize Phi before the ridge
            fit (see fit_ls's own docstring). Defaults to False -- verified
            empirically via a ridge_lambda sweep (scripts/
            compare_cgr_vs_kriging.py-style harness) even WITH alpha_m=1
            for every Gaussian (no more amplitude-domain path-loss decay
            creating near-dead columns): normalize=True still produced
            catastrophic outliers (MAE ~80-350 dB) at every tested
            ridge_lambda, while normalize=False with ridge_lambda=1.0 gave
            MAE=10.01 dB / RMSE=13.07 dB / max prediction=35 dB (zero
            predictions above 40 dB) -- BETTER than Kriging's own baseline
            (MAE=10.75 / RMSE=14.07) on the same data. Something besides
            alpha (likely the beam_footprint scale/geometry) still
            produces near-dead columns in this basis; unconstrained ridge
            has no sparsity to catch the resulting amplification when
            un-normalizing.
        :param baseline_method: how fit_current_interval's own output
            baseline (what predict_targets adds back after the Gaussian-
            basis regression) is computed:
              "global_mean"       -- (default, existing behavior) a
                  single scalar, the flat mean of ALL known values in the
                  sector, added back identically everywhere regardless of
                  a target's distance from them.
              "distance_weighted" -- opt-in: a Gaussian-kernel-weighted
                  LOCAL average of known values instead (bandwidth=
                  baseline_length_scale_m), computed separately per known
                  point (leave-one-out, to detrend the fit target) and per
                  target point (to add back after prediction). Addresses a
                  diagnosed sampling-bias effect: sparse real measurements
                  (driven by HotspotDemandMobility) skew toward
                  better-covered locations than the wider unvisited-cell
                  population -- confirmed empirically (observed-cell
                  ground-truth SINR averaging 0.85-4.4 dB higher than
                  unobserved-cell ground truth, every interval tested). A
                  single GLOBAL mean baseline can't correct for this on a
                  per-target basis; a local one can partially compensate
                  by weighting nearby (more relevant) known points more.
        :param baseline_length_scale_m: Gaussian-kernel bandwidth [m] for
            baseline_method="distance_weighted" -- ignored otherwise.
        """
        self.device = torch.device(device)
        self.num_gaussian = int(num_gaussian)
        self.ridge_lambda = float(ridge_lambda)
        self.normalize_columns_db = normalize_columns_db
        assert baseline_method in ("global_mean", "distance_weighted"), \
            "baseline_method must be 'global_mean' or 'distance_weighted'"
        self.baseline_method = baseline_method
        self.baseline_length_scale_m = float(baseline_length_scale_m)

        # ---------------------------------------------------------
        # Reuse your existing CGR implementation.
        #
        # We are predicting ONE scalar radio value per spatial point,
        # so num_tx_ant=num_rx_ant=1 are merely dummy dimensions.
        #
        # IMPORTANT:
        # We will use predict_flat(), NOT predict_channel().
        # ---------------------------------------------------------
        # `gaussian_config` (if given) OVERRIDES on top of these defaults
        # (merged, not replaced) -- so a caller like SpatialGridEstimator
        # only needs to pass the tilt-aware-specific keys it cares about
        # (beamwidths/fractions/PLE), not replicate scale_value/num_gaussian/
        # alpha_min/etc, which CGR(...) requires present (via `[...]`, not
        # `.get()`) regardless of mean/scale/alpha method chosen.
        default_gaussian_config = {
            "num_gaussian": self.num_gaussian,

            # Existing CGR initialization options
            "scale_value": 10.0,

            # These can later be replaced by tilt-aware initialization
            "mean_init_method": "tx_path_grid",
            "alpha_init_method": "distance",
            "scale_init_method": "multiscale",
            "rotation_init_method": "propagation_aligned",

            # Example defaults -- tune to your simulation
            "data_fraction": 0.3,
            "path_fraction": 0.3,
            "grid_jitter_std": 0.05,

            "alpha_value": 1.0,
            "alpha_min": 1e-3,
            "alpha_max": 1.0,

            "scale_min": 1.0,
            "scale_max": 100.0,

            # The PHYSICAL power-law path-loss exponent eta -- doubled
            # below (see the comment right after this dict) before being
            # handed to _init_alpha's "distance" method, which halves
            # whatever it's given (correct for the reference model's own
            # amplitude-domain use case, not for our power-domain one).
            "path_loss_exponent": 3.0,

            # "tilt_aware_beam"/"beam_footprint" defaults (see
            # algorithm/cgr_model.py) -- horizontal/vertical beamwidth
            # default to TR 38.901 Table 7.3-1's own element HPBW (65 deg),
            # not exposed elsewhere in this repo as a named constant.
            "horizontal_beamwidth_deg": 65.0,
            "vertical_beamwidth_deg": 65.0,
            "bs_fraction": 0.15,
            "beam_fraction": 0.35,
            "longitudinal_scale_coeff": 1.0,
            # Deterministic-but-NOT-colocated spread radius for the bs/beam
            # buckets (fraction of the geo-sector's own grid diagonal) --
            # see algorithm/cgr_model.py's _sunflower_xy/_along_across_xy:
            # with grid_jitter_std=0.0, every Gaussian in a bucket would
            # otherwise land on the EXACT same point (identical mu/Sigma/R/
            # alpha -> identical, wasted Phi columns, worsening the exact
            # rank-deficiency this whole design fights).
            "bucket_spread_frac": 0.05,

            # alpha_init_method="distance"'s own default d_ref (~0.06m,
            # falling back to a WAVELENGTH-scale carrier_wavelength we
            # never supply) is a near-field reference distance sized for
            # the reference implementation's own (much smaller-scale)
            # original problem -- at this repo's real cellular distances
            # (tens to hundreds of meters), (d_ref/d)^(PLE/2) underflows
            # to alpha_min for EVERY Gaussian uniformly, erasing all
            # distance-based differentiation and leaving an ill-
            # conditioned ridge-LS fit (verified empirically: predictions
            # collapsed to near-zero everywhere except at the exact known
            # points). 35.0m ~= a typical min_bs_ut_dist for this
            # scenario -- keeps alpha in a well-conditioned, meaningfully
            # varying range across realistic sector distances.
            "d_ref": 35.0,
        }
        self.gaussian_config = {**default_gaussian_config, **(gaussian_config or {})}
        # _init_alpha's "distance" method computes
        # alpha = alpha_ref * (d_ref/d)**(path_loss_exponent/2) -- correct
        # when path_loss_exponent/2 is an AMPLITUDE-domain exponent (the
        # reference model's own original complex-channel-coefficient use
        # case, H relates to amplitude). Alpha here is a POWER-domain
        # prior (this Gaussian's own activation weight in a linear
        # combination predicting dB(power)), so double whatever PHYSICAL
        # power-law exponent was configured (default or override) here,
        # ONCE, after merging -- _init_alpha's own /2 then cancels it,
        # landing on the correct physical eta instead of eta/2. (Only
        # relevant to the V1 fallback path now -- the tilt-aware path
        # uses alpha_init_method="constant", alpha_m=1 for every Gaussian.)
        self.gaussian_config["path_loss_exponent"] = 2.0 * self.gaussian_config["path_loss_exponent"]

        # Captured so the tilt-aware fallback (see _initialize_gaussians)
        # can explicitly restore V1 behavior rather than trusting whatever
        # mean_init_method/scale_init_method a PREVIOUS call's tilt-aware
        # override left in self.gaussian_config.
        self._v1_mean_init_method = self.gaussian_config.get("mean_init_method", "tx_path_grid")
        self._v1_scale_init_method = self.gaussian_config.get("scale_init_method", "multiscale")
        self._v1_rotation_init_method = self.gaussian_config.get("rotation_init_method", "propagation_aligned")
        self._v1_alpha_init_method = self.gaussian_config.get("alpha_init_method", "distance")
        self._v1_grid_jitter_std = self.gaussian_config.get("grid_jitter_std", 0.05)

        model_params = {
            "gaussian_functions": self.gaussian_config,
            "ridge_lambda": self.ridge_lambda,
        }

        self.model = CGR(
            num_tx_ant=1,
            num_rx_ant=1,
            device=self.device,
            **model_params,
        )

    # =============================================================
    # DOMAIN-KNOWLEDGE INITIALIZATION
    # =============================================================

    def _initialize_gaussians(
        self,
        known_xyz: torch.Tensor,
        env_dims: torch.Tensor,
        tx_position: torch.Tensor,
        context: dict,
    ):
        """
        Initialize Gaussian basis for THIS measurement interval, for THIS
        transmitting sector's own RSRP field. CGR models each transmitting
        sector's own P_i(x) independently (own BS position/azimuth/tilt
        only -- never a neighboring sector's tilt); SINR is reconstructed
        afterward by combining every sector's own RSRP prediction (see
        helpers/spatial_grid_estimator.py's rsrp_per_sector branch, which
        already does exactly that combination) -- nothing here needs to
        know about other sectors at all.

        context:
            "tx_position"      -- required (this sector's own [3] world xyz).
            "own_azimuth_deg"  -- this sector's own boresight yaw [deg].
            "own_tilt_deg"     -- this sector's own current electrical
                                  downtilt [deg] (0=boresight/horizon, +=down).
            "receiver_height"  -- existing key, the grid-map evaluation
                                  height [m] (default 1.5).
            "env_dims"         -- existing key, [3,2] -- pass the geo
                                  sector's own TIGHT grid bounding box when
                                  available (see SpatialGridEstimator's own
                                  context builder), not the wider fallback
                                  built below from measured points alone.

        If "own_azimuth_deg"/"own_tilt_deg" are both present, uses the
        tilt-aware initialization (algorithm/cgr_model.py's
        "tilt_aware_beam"/"beam_footprint" methods). Otherwise this is a
        genuine "missing domain knowledge" case -- falls back to the V1
        generic physics-informed initialization (logged once), rather than
        silently guessing values for the tilt/azimuth-dependent methods.
        """

        wavelength = context.get("carrier_wavelength", None)
        has_tilt_info = "own_azimuth_deg" in context and "own_tilt_deg" in context

        if has_tilt_info:
            # Mutates the SAME dict object self.model reads init_config
            # from (self.model.gaussian_params IS self.gaussian_config --
            # see __init__'s model_params) -- init_gaussians() re-reads it
            # fresh every call, so no change to its own signature is
            # needed to thread these per-interval, per-sector values in.
            self.gaussian_config["mean_init_method"] = "tilt_aware_beam"
            self.gaussian_config["scale_init_method"] = "beam_footprint"
            # NOT "propagation_aligned" (V1's default) -- see
            # algorithm/cgr_model.py's "yaw_aligned" docstring: a 3D-
            # aligned rotation's arbitrary per-Gaussian twist can leak
            # real horizontal displacement onto beam_footprint's
            # deliberately-tight z-scale, catastrophically.
            self.gaussian_config["rotation_init_method"] = "yaw_aligned"
            # alpha_m = 1 for every Gaussian -- NOT alpha_init_method=
            # "distance" (a (d_ref/d)^(eta/2) AMPLITUDE-domain path-loss
            # prior). Domain knowledge belongs ONLY in the Gaussian
            # GEOMETRY (mu from BS+azimuth+tilt, Sigma from beamwidth+
            # tilt, R from sector direction) -- that's already fully
            # captured by mean/scale/rotation above. The actual RSRP
            # amplitude should come from the real measurements via the
            # ridge fit, not from a second, physically-mismatched prior
            # layered on top of it.
            self.gaussian_config["alpha_init_method"] = "constant"
            self.gaussian_config["alpha_value"] = 1.0
            self.gaussian_config["own_azimuth_deg"] = context["own_azimuth_deg"]
            self.gaussian_config["own_tilt_deg"] = context["own_tilt_deg"]
            # Deterministic placement: the SAME sector geometry + tilt
            # must produce the SAME mu/Sigma/R every call, to isolate the
            # regression fit itself as the thing under test rather than
            # letting the basis re-roll dice every interval. jitter_std=0
            # makes the bs/beam/grid buckets exact (see
            # algorithm/cgr_model.py's "tilt_aware_beam" for the "path"
            # bucket's own separate determinism fix, an evenly-spaced
            # linspace instead of a random interpolation parameter).
            self.gaussian_config["grid_jitter_std"] = 0.0
        else:
            self.gaussian_config["mean_init_method"] = self._v1_mean_init_method
            self.gaussian_config["scale_init_method"] = self._v1_scale_init_method
            self.gaussian_config["rotation_init_method"] = self._v1_rotation_init_method
            self.gaussian_config["alpha_init_method"] = self._v1_alpha_init_method
            self.gaussian_config["grid_jitter_std"] = self._v1_grid_jitter_std
            logger.warning("CGRRadioMapEstimator._initialize_gaussians: context is missing "
                           "own_azimuth_deg/own_tilt_deg -- falling back to the generic V1 "
                           "(%s/%s) initialization instead of tilt-aware placement",
                           self._v1_mean_init_method, self._v1_scale_init_method)

        # Avoid an underdetermined basis (M >= N): with N known points and
        # M Gaussians, M comfortably below N keeps ridge regression an
        # actual regression, not an interpolation puzzle with more
        # unknowns than equations -- e.g. 12 Gaussians fit against 6
        # measured cells is asking for exactly the kind of ill-
        # conditioned, wildly-extrapolating fit this whole investigation
        # has been fighting. Caps at half the known points (minimum 1),
        # adapting per call since N varies a lot (sparse sectors/intervals
        # can have very few real measurements).
        num_known = known_xyz.shape[0]
        effective_num_gaussian = max(1, min(self.num_gaussian, num_known // 2))
        self.model.init_gaussians(
            num_gaussian=effective_num_gaussian,
            env_dims=env_dims,
            tx_position=tx_position,
            wavelength=wavelength,
            train_positions=known_xyz,
            num_train=known_xyz.shape[0],
        )

    # =============================================================
    # FIT CURRENT INTERVAL
    # =============================================================

    def fit_current_interval(
        self,
        known_xy: np.ndarray,
        known_values: np.ndarray,
        context: dict,
    ):
        """
        Fit CGR using ONLY measurements from the current interval.

        known_xy:
            [N, 2]

        known_values:
            [N]
            e.g. SS-SINR / RSRP / received power

        context:
            current radio/network configuration
        """

        if known_xy.shape[0] == 0:
            raise ValueError(
                "CGRRadioMapEstimator requires at least one current measurement"
            )

        known_xy = np.asarray(known_xy, dtype=np.float32)
        known_values = np.asarray(known_values, dtype=np.float32)

        # ---------------------------------------------------------
        # Convert 2-D map coordinates to 3-D positions.
        # ---------------------------------------------------------
        #
        # TODO:
        # Replace receiver_height with the actual UE/grid-map height
        # available from your simulator.
        #
        receiver_height = float(context.get("receiver_height", 1.5))

        known_xyz = np.column_stack(
            [
                known_xy,
                np.full(known_xy.shape[0], receiver_height),
            ]
        )

        known_xyz = torch.tensor(
            known_xyz,
            dtype=torch.float32,
            device=self.device,
        )

        # ---------------------------------------------------------
        # Environment dimensions
        # ---------------------------------------------------------
        #
        # TODO:
        # Prefer the actual simulation-area boundaries if available.
        #
        if "env_dims" in context:
            env_dims = torch.tensor(
                context["env_dims"],
                dtype=torch.float32,
                device=self.device,
            )
        else:
            # Fallback from currently observed positions.
            # This is not ideal because targets may extend beyond them.
            xyz_np = known_xyz.detach().cpu().numpy()

            mins = xyz_np.min(axis=0)
            maxs = xyz_np.max(axis=0)

            margin = 10.0
            mins[:2] -= margin
            maxs[:2] += margin

            # avoid zero z-range
            mins[2] = receiver_height - 1.0
            maxs[2] = receiver_height + 1.0

            env_dims = torch.tensor(
                np.stack([mins, maxs], axis=1),
                dtype=torch.float32,
                device=self.device,
            )

        # ---------------------------------------------------------
        # TX position
        # ---------------------------------------------------------
        #
        # TODO:
        # Select the correct TX / serving sector depending on how
        # this estimator is instantiated.
        #
        if "tx_position" not in context:
            raise ValueError(
                "context['tx_position'] is required for "
                "physics-informed CGR initialization"
            )

        tx_position = torch.tensor(
            context["tx_position"],
            dtype=torch.float32,
            device=self.device,
        )

        # ---------------------------------------------------------
        # Initialize Gaussian basis from CURRENT physical context.
        # ---------------------------------------------------------
        self._initialize_gaussians(
            known_xyz=known_xyz,
            env_dims=env_dims,
            tx_position=tx_position,
            context=context,
        )

        # ---------------------------------------------------------
        # Fit W_out from CURRENT measurements only.
        #
        # The reference model's feature basis has NO bias/intercept
        # column, so ridge regression's implicit "shrink W_out toward 0"
        # prior is really a "shrink the OUTPUT toward 0" prior -- correct
        # for the reference model's original use case (complex MIMO
        # channel coefficients, naturally ~zero-mean), but wrong for RSRP
        # in dB, which is always strongly NEGATIVE (uncentered, any
        # poorly-constrained/underdetermined Gaussian gets regularized
        # toward an implausibly STRONG near-0-dB signal instead of the
        # true typical level -- verified empirically: stronger ridge made
        # predictions WORSE, pulling harder toward that wrong zero).
        #
        # Center on the known values' own mean; predict_targets adds it
        # back.
        #
        # [N] -> [N,1] is enough for CGR._flatten_channel().
        # ---------------------------------------------------------
        if self.baseline_method == "global_mean":
            self._value_baseline = float(known_values.mean())
            fit_values = known_values - self._value_baseline
        else:
            # distance_weighted: each known point's own baseline is a
            # Gaussian-kernel-weighted average of the OTHER known points
            # (leave-one-out -- a point can't use its own value to
            # detrend itself); predict_targets computes each TARGET
            # point's own baseline the same way, but over ALL known
            # points (no leave-one-out, since a target isn't one of them).
            self._known_xy_for_baseline = known_xy.copy()
            self._known_values_for_baseline = known_values.copy()
            local_baseline = self._weighted_local_baseline(known_xy, known_xy, known_values, leave_one_out=True)
            self._value_baseline = local_baseline  # per-known-point array, not a scalar
            fit_values = known_values - local_baseline
        targets = torch.tensor(
            fit_values[:, None],
            dtype=torch.float32,
            device=self.device,
        )

        self.model.fit_ls(rx_positions=known_xyz, channel_targets=targets, ridge_lambda=self.ridge_lambda,
                          normalize_columns=self.normalize_columns_db)

    def _weighted_local_baseline(self, query_xy: np.ndarray, known_xy: np.ndarray, known_values: np.ndarray,
                                 leave_one_out: bool) -> np.ndarray:
        """Gaussian-kernel-weighted average of `known_values` at each
        known_xy point, evaluated at every `query_xy` point (bandwidth=
        self.baseline_length_scale_m). `leave_one_out=True` zeroes each
        point's own self-weight (query_xy is known_xy, used while fitting
        -- a point can't use its own value to detrend itself);
        `leave_one_out=False` is for predicting at genuinely separate
        target points. Falls back to the plain (unweighted) mean of
        known_values wherever every weight underflows to ~0 (e.g. a
        single known point under leave-one-out).
        """
        dists = np.linalg.norm(query_xy[:, None, :] - known_xy[None, :, :], axis=-1)
        weights = np.exp(-0.5 * (dists / self.baseline_length_scale_m) ** 2)
        if leave_one_out:
            np.fill_diagonal(weights, 0.0)
        weight_sums = weights.sum(axis=1)
        has_weight = weight_sums > 0
        baseline = np.full(query_xy.shape[0], float(known_values.mean()), dtype=np.float64)
        baseline[has_weight] = (weights[has_weight] @ known_values) / weight_sums[has_weight]
        return baseline

    # =============================================================
    # PREDICT UNVISITED LOCATIONS
    # =============================================================

    def predict_targets(
        self,
        target_xy: np.ndarray,
        context: dict,
    ) -> np.ndarray:
        """
        Predict the fitted radio quantity at target spatial locations.

        target_xy:
            [M, 2]

        returns:
            [M]
        """

        target_xy = np.asarray(target_xy, dtype=np.float32)

        receiver_height = float(context.get("receiver_height", 1.5))

        target_xyz = np.column_stack(
            [
                target_xy,
                np.full(target_xy.shape[0], receiver_height),
            ]
        )

        target_xyz = torch.tensor(
            target_xyz,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():
            pred = self.model.predict_flat(target_xyz)
        pred = pred[:, 0].detach().cpu().numpy()

        # Add back the baseline subtracted in fit_current_interval.
        if self.baseline_method == "global_mean":
            return pred + self._value_baseline
        target_baseline = self._weighted_local_baseline(
            target_xy, self._known_xy_for_baseline, self._known_values_for_baseline, leave_one_out=False)
        return pred + target_baseline

    # =============================================================
    # INTERFACE
    # =============================================================

    def predict(
        self,
        known_xy: np.ndarray,
        known_values: np.ndarray,
        target_xy: np.ndarray,
        context: dict,
    ) -> np.ndarray:
        """
        Kriging-like interface:

            known measurements
                -> fit CURRENT CGR
                -> predict target locations
        """

        logger.function(
            "CGRRadioMapEstimator.predict start: n_known=%d n_target=%d",
            known_xy.shape[0],
            target_xy.shape[0],
        )

        self.fit_current_interval(
            known_xy=known_xy,
            known_values=known_values,
            context=context,
        )

        prediction = self.predict_targets(
            target_xy=target_xy,
            context=context,
        )

        logger.function("CGRRadioMapEstimator.predict end")

        return prediction