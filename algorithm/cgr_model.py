import warnings
import csv
from pathlib import Path
from typing import Optional,  Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from algorithm.cov_utils import build_rotation, build_covariance, build_covariance_inverse


class CGR(nn.Module):
    """
    Fixed Gaussian radio field + Two stage LS output layer.

    Input:
        rx_pos, env_dim, tx_pos
    Output:
        channel matrix H(p) in C^{Nt x Nr}

    Model:
        phi_i(p) = alpha_i * exp(-1/2 * (p-mu_i)^T Sigma_i^{-1} (p-mu_i))
        H_hat(p) = sum_i phi_i(p) * C_i
    """

    def __init__(self, num_tx_ant: int, num_rx_ant: int, device, **params):
        super().__init__()
        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant
        self.device = device

        gaussian_params = params["gaussian_functions"]
        self.gaussian_params = gaussian_params
        self.num_gaussian = int(gaussian_params["num_gaussian"])
        self.init_scale = torch.tensor(gaussian_params["scale_value"], device=device)
        self.ridge_lambda = float(params.get("ridge_lambda", 1e-4))

        # Fixed Gaussian parameters
        self.register_buffer("_gaussian_mean", torch.empty(0, 3, device=device))
        self.register_buffer("_rotation_quat", torch.empty(0, 4, device=device))
        self.register_buffer("_scaling", torch.empty(0, 3, device=device))
        self.register_buffer("_alpha", torch.empty(0, 1, device=device))
        # LS output layer
        # shape: [num_gaussian, 2 * Nt * Nr]
        self.register_buffer("_W_out", torch.empty(0, 0, device=device))

    @property
    def get_gaussian_mean(self):
        return self._gaussian_mean

    @property
    def get_scaling(self):
        return torch.exp(self._scaling).clamp(min=1e-8)

    @property
    def get_rotation(self):
        return F.normalize(self._rotation_quat, p=2, dim=-1)

    @property
    def get_base_activation(self):
        return self._alpha

    @property
    def get_W_out(self):
        return self._W_out

    def get_covariance(self, return_inverse=False, eps=1e-6):
        scaling = self.get_scaling
        rotation_q = self.get_rotation

        R = build_rotation(rotation_q)
        covariance = build_covariance(R, scaling)

        if return_inverse:
            inv_covariance = build_covariance_inverse(R, scaling, eps)
            if torch.isnan(inv_covariance).any() or torch.isinf(inv_covariance).any():
                warnings.warn("NaN or Inf detected in inverse covariance. Replacing with identity.")
                bad_indices = torch.isnan(inv_covariance).any(dim=(1, 2)) | torch.isinf(inv_covariance).any(dim=(1, 2))
                identity = torch.eye(3, device=self.device, dtype=inv_covariance.dtype).expand(bad_indices.sum(), -1, -1)
                inv_covariance[bad_indices] = identity
            return covariance, inv_covariance
        else:
            return covariance

    def _init_means(self, num_gaussian, init_config, env_dims=None, tx_position=None, train_positions=None):
        method = init_config.get("mean_init_method")
        method = str(method).lower()
        # Mean Initialization method 1: uniform
        if method == "random":
            if env_dims is None:
                warnings.warn("mean_init_method='uniform_env' requires env_dims; falling back to [-1, 1].")
                return torch.rand(num_gaussian, 3, device=self.device) * 2 - 1
            env_min = env_dims[:, 0].to(self.device)
            env_max = env_dims[:, 1].to(self.device)
            return torch.rand(num_gaussian, 3, device=self.device) * (env_max - env_min) + env_min

        #  Mean Initialization method 2: grid jitter (uniform in the grid space)
        if method == "grid":
            env_min = env_dims[:, 0].to(self.device)
            env_max = env_dims[:, 1].to(self.device)
            side = max(1, math.ceil(num_gaussian ** (1.0 / 3.0)))
            axes = [torch.linspace(env_min[d], env_max[d], steps=side, device=self.device) for d in range(3)]
            grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)[:num_gaussian]
            jitter_scale = float(init_config.get("grid_jitter_std", 0.05))
            jitter = (env_max - env_min) * jitter_scale * torch.randn_like(grid)
            return (grid + jitter).clamp(min=env_min, max=env_max)

        if method == "tx_path_grid":
            """Physics-informed mean placement using training data and propagation paths.
            Strategy:
              - data_fraction: centers near/at training positions (capture local structure)
              - path_fraction: centers along TXâdata rays extended beyond data (extrapolation)
              - remaining: grid coverage for unexplored regions
            """
            env_min = env_dims[:, 0].to(self.device)
            env_max = env_dims[:, 1].to(self.device)
            tx = tx_position.to(self.device).float().reshape(1, 3)
            train_pos = train_positions.to(self.device).float()
            n_train = train_pos.shape[0]

            data_frac = float(init_config.get("data_fraction", 0.3))
            path_frac = float(init_config.get("path_fraction", 0.3))
            n_data = max(1, int(num_gaussian * data_frac))
            n_path = max(1, int(num_gaussian * path_frac))
            n_grid = num_gaussian - n_data - n_path

            # (a) Centers near training positions with jitter
            jitter_std = float(init_config.get("grid_jitter_std", 0.1))
            idx = torch.randint(0, n_train, (n_data,), device=self.device)
            env_range = env_max - env_min
            data_means = train_pos[idx] + jitter_std * env_range * torch.randn(n_data, 3, device=self.device)

            # (b) Centers along TXâdata propagation rays at varying distances
            #     Parameterize as: p(t) = TX + t * (data_i - TX), t in [0.5, 2.0]
            #     t > 1 extends beyond data point (extrapolation for test coverage)
            ray_idx = torch.randint(0, n_train, (n_path,), device=self.device)
            t_min = float(init_config.get("ray_t_min", 0.3))
            t_max = float(init_config.get("ray_t_max", 2.0))
            t = t_min + (t_max - t_min) * torch.rand(n_path, 1, device=self.device)
            ray_dir = train_pos[ray_idx] - tx
            path_means = tx + t * ray_dir

            # (c) Grid coverage for remaining
            grid_means = self._init_means(n_grid, {**init_config, "mean_init_method": "grid"},
                                          env_dims=env_dims, tx_position=tx_position)

            means = torch.cat([data_means, path_means, grid_means], dim=0)[:num_gaussian]
            return means.clamp(min=env_min, max=env_max)

        if method == "tilt_aware_beam":
            """Physics-informed mean placement using the antenna's OWN
            known tilt/azimuth -- unlike tx_path_grid, this does not need
            training data to infer WHERE the beam points, since tilt and
            azimuth already fully determine it.

            All means are placed at the QUERY height (env_dims' own z --
            see algorithm/cgr.py's context builder, which sets it tight
            around receiver_height), not the antenna's real height: every
            measurement/prediction position fed to this model sits on that
            one fixed height plane (see CGRRadioMapEstimator.
            fit_current_interval/predict_targets), so placing means there
            too keeps every mean and every query point on the SAME z --
            avoiding a spurious z-mismatch _init_scales would otherwise
            have to compensate for.

            init_config keys (all required for this method, set by
            algorithm/cgr.py's _initialize_gaussians from `context`):
              own_tilt_deg    -- this sector's current electrical downtilt
                                 [deg], 0=boresight/horizon, +=down.
              own_azimuth_deg -- this sector's boresight yaw [deg].
              bs_fraction, beam_fraction, path_fraction -- mean-count split
                                 (remainder goes to generic grid coverage).
            """
            env_min = env_dims[:, 0].to(self.device)
            env_max = env_dims[:, 1].to(self.device)
            tx = tx_position.to(self.device).float().reshape(3)
            h_rx = float((env_min[2] + env_max[2]) / 2.0)
            tx_ground = torch.stack([tx[0], tx[1], torch.tensor(h_rx, device=self.device, dtype=tx.dtype)])

            tilt_deg = float(init_config.get("own_tilt_deg", 0.0))
            az_deg = float(init_config.get("own_azimuth_deg", 0.0))
            az_rad = az_deg * math.pi / 180.0
            tilt_rad = tilt_deg * math.pi / 180.0
            delta_h = float(tx[2].item()) - h_rx
            env_diag = float((env_max - env_min)[:2].norm().item())

            tan_tilt = math.tan(tilt_rad)
            horizontal_distance = (delta_h / tan_tilt) if abs(tan_tilt) > 1e-6 else env_diag
            horizontal_distance = max(0.0, min(horizontal_distance, env_diag))

            beam_center = torch.stack([
                tx[0] + horizontal_distance * math.cos(az_rad),
                tx[1] + horizontal_distance * math.sin(az_rad),
                torch.tensor(h_rx, device=self.device, dtype=tx.dtype),
            ])

            bs_frac = float(init_config.get("bs_fraction", 0.15))
            beam_frac = float(init_config.get("beam_fraction", 0.35))
            path_frac = float(init_config.get("path_fraction", 0.2))
            n_bs = max(1, int(num_gaussian * bs_frac))
            n_beam = max(1, int(num_gaussian * beam_frac))
            n_path = max(1, int(num_gaussian * path_frac))
            n_grid = max(0, num_gaussian - n_bs - n_beam - n_path)

            z_col = torch.full((1,), h_rx, device=self.device, dtype=tx.dtype)
            spread_frac = float(init_config.get("bucket_spread_frac", 0.05))
            spread_radius = spread_frac * env_diag

            def _sunflower_xy(center_xy, n, radius):
                # n DETERMINISTIC, non-colocated offsets (a fixed
                # "sunflower" point pattern -- no RNG at all, so the same
                # center/n/radius always gives the same distinct
                # positions). grid_jitter_std=0.0 alone would place every
                # point in a bucket at the EXACT same spot -- deterministic
                # means reproducible, not identical: n_gaussian identical
                # columns of Phi waste basis capacity and worsen the exact
                # rank-deficiency this whole design is fighting.
                if n == 1:
                    return torch.cat([center_xy.unsqueeze(0), z_col.expand(1, 1)], dim=-1)
                golden_angle = math.pi * (3.0 - math.sqrt(5.0))
                k = torch.arange(n, device=self.device, dtype=center_xy.dtype)
                r = radius * torch.sqrt((k + 0.5) / n)
                theta = k * golden_angle
                offsets = torch.stack([r * torch.cos(theta), r * torch.sin(theta)], dim=-1)
                xy = center_xy.unsqueeze(0) + offsets
                return torch.cat([xy, z_col.expand(n, 1)], dim=-1)

            def _along_across_xy(center_xy, n, radius, axis_rad):
                # n DETERMINISTIC offsets alternating along/across the
                # given axis (the beam's own propagation direction) at
                # increasing distances -- matches beam_footprint's own
                # longitudinal/lateral scale axes, unlike an isotropic
                # scatter, and always includes the exact center itself.
                if n == 1:
                    return torch.cat([center_xy.unsqueeze(0), z_col.expand(1, 1)], dim=-1)
                along = torch.tensor([math.cos(axis_rad), math.sin(axis_rad)], device=self.device, dtype=center_xy.dtype)
                perp = torch.tensor([-math.sin(axis_rad), math.cos(axis_rad)], device=self.device, dtype=center_xy.dtype)
                fracs = torch.linspace(-1.0, 1.0, n, device=self.device, dtype=center_xy.dtype)
                directions = torch.stack([along if k % 2 == 0 else perp for k in range(n)], dim=0)
                offsets = radius * fracs.unsqueeze(1) * directions
                xy = center_xy.unsqueeze(0) + offsets
                return torch.cat([xy, z_col.expand(n, 1)], dim=-1)

            bs_means = _sunflower_xy(tx_ground[:2], n_bs, spread_radius)
            beam_means = _along_across_xy(beam_center[:2], n_beam, spread_radius, az_rad)

            # Evenly-spaced (not random) interpolation parameter -- same
            # sector geometry + tilt must give the same mu every call
            # (see algorithm/cgr.py's grid_jitter_std=0.0 override, which
            # handles the bs/beam/grid buckets' own determinism; this one
            # needs a direct fix since it isn't jitter-scaled at all).
            t = torch.linspace(0.0, 1.0, n_path, device=self.device).unsqueeze(1) if n_path > 1 \
                else torch.full((n_path, 1), 0.5, device=self.device)
            path_dir = (beam_center - tx_ground).unsqueeze(0)
            path_means = tx_ground.unsqueeze(0) + t * path_dir  # z-component of path_dir is 0 -- stays at h_rx too

            # Degenerate z-range (a linspace between two EQUAL endpoints is
            # a constant array) so the generic "grid" method's remainder
            # bucket also lands exactly on z=h_rx, not merely close to it.
            grid_env_dims = env_dims.clone()
            grid_env_dims[2, 0] = h_rx
            grid_env_dims[2, 1] = h_rx
            grid_means = self._init_means(n_grid, {**init_config, "mean_init_method": "grid"},
                                          env_dims=grid_env_dims) if n_grid > 0 else torch.empty(0, 3, device=self.device)

            means = torch.cat([bs_means, beam_means, path_means, grid_means], dim=0)[:num_gaussian]
            if means.shape[0] < num_gaussian:
                pad = self._init_means(num_gaussian - means.shape[0], {**init_config, "mean_init_method": "grid"},
                                       env_dims=grid_env_dims)
                means = torch.cat([means, pad], dim=0)
            # Deliberately NOT clamped to env_dims (unlike "grid"/"tx_path_grid"):
            # env_dims here is the QUERY region's own grid (e.g. geographic
            # sector i's), but tx/beam_center can legitimately sit far
            # outside it -- predicting a DIFFERENT sector b's own RSRP
            # field over sector i's grid (rsrp_per_sector's interference
            # terms) is the common case, not the exception. Clamping would
            # collapse the bs/beam/path buckets onto whichever box corner
            # is nearest tx, wasting most of the Gaussian budget on one
            # degenerate point AND decoupling _init_alpha's distance-from-
            # tx term (computed from the REAL, unclamped tx) from where
            # the means actually ended up -- verified empirically via
            # scripts/compare_cgr_vs_kriging.py: clamping made CGR predict
            # near-zero coverage almost everywhere (every sector's
            # near-degenerate feature response made SINR reconstruction's
            # interference term blow up), only the "grid" bucket -- which
            # DOES stay confined to env_dims, being separately generic
            # coverage of the query region -- needs to.
            return means


    def _init_alpha(self, num_gaussian, init_config, means=None, tx_position=None, wavelength=None):
        method = str(init_config.get("alpha_init_method", "constant")).lower()
        alpha_min = float(init_config.get("alpha_min", 1e-3))
        alpha_max = float(init_config.get("alpha_max", 1.0))
        if method == "constant":
            alpha_value = float(init_config.get("alpha_value"))
            return torch.full((num_gaussian, 1), alpha_value, device=self.device)
        if method == "random":
            return torch.rand(num_gaussian, 1, device=self.device) * (alpha_max - alpha_min) + alpha_min
        if method == "lognormal":
            mu = float(init_config.get("alpha_lognormal_mean", math.log(max(alpha_min, 1e-8))))
            sigma = float(init_config.get("alpha_lognormal_std", 0.5))
            a = torch.exp(mu + sigma * torch.randn(num_gaussian, 1, device=self.device))
            return a.clamp(min=alpha_min, max=alpha_max)
        if method == "distance":
            """Physics-based alpha using standard wireless path loss models.

            Models the received signal amplitude decay with distance from TX.
            Based on the log-distance path loss model:
                PL(d) = PL(d0) + 10 * n * log10(d / d0)
            where n is the path loss exponent (PLE).

            The activation coefficient represents the relative amplitude:
                alpha_i = alpha_ref * (d_ref / d_i)^(n/2)

            The n/2 exponent converts from power-domain path loss to
            amplitude-domain (voltage), since channel H relates to amplitude.

            Typical PLE values:
                - Free space:      n = 2.0
                - Indoor LOS:      n = 1.6 - 1.8
                - Indoor NLOS:     n = 2.5 - 4.0
                - Outdoor urban:   n = 2.7 - 3.5
                - Outdoor suburban: n = 3.0 - 5.0

            With log-normal shadow fading perturbation (sigma_sf in dB):
                alpha_i *= 10^(X / 20), X ~ N(0, sigma_sf)
            """
            tx = tx_position.to(self.device).float().reshape(1, 3)
            d = torch.linalg.norm(means - tx, dim=-1, keepdim=True)  # [M, 1]

            ple = float(init_config.get("path_loss_exponent", 2.0))
            wl = wavelength if wavelength is not None else float(init_config.get("carrier_wavelength", 0.06))

            # Reference distance d0 = wavelength (near-field boundary)
            d_ref = float(init_config.get("d_ref", wl))
            d_safe = d.clamp(min=d_ref)

            # Amplitude: alpha = alpha_ref * (d_ref / d)^(n/2)
            alpha_ref = float(init_config.get("alpha_ref", 0.5))
            alpha = alpha_ref * (d_ref / d_safe).pow(ple / 2.0)

            return alpha.clamp(min=alpha_min, max=alpha_max)
        raise ValueError(f"Unsupported alpha_init_method='{method}'")


    def _init_scales(self, num_gaussian, init_config, means=None, tx_position=None,
                     env_dims=None, wavelength=None, num_train=None):
        method = str(init_config.get("scale_init_method", "constant")).lower()
        scale_min = float(init_config.get("scale_min", max(self.init_scale.item() * 0.5, 1e-6)))
        scale_max = float(init_config.get("scale_max", self.init_scale.item() * 2.0))
        if method == "constant":
            scale_value = float(init_config.get("scale_value", self.init_scale.item()))
            return torch.full((num_gaussian, 3), scale_value, device=self.device)
        if method == "log_uniform":
            u = torch.rand(num_gaussian, 3, device=self.device)
            return torch.exp(torch.log(torch.tensor(scale_min, device=self.device)) + u * math.log(scale_max / scale_min))
        if method == "multiscale":
            env_min_v = env_dims[:, 0].to(self.device)
            env_max_v = env_dims[:, 1].to(self.device)
            env_range = env_max_v - env_min_v
            env_diag = env_range.norm().item()

            # Lower bound: inter-sample spacing (data density limit)
            env_vol = env_range.prod().item()
            delta_min = (env_vol / max(num_train, 1)) ** (1.0 / 3.0)

            # Configurable fraction overrides
            frac_min = float(init_config.get("scale_env_frac_min", delta_min / env_diag))
            frac_max = float(init_config.get("scale_env_frac_max", 2.0))

            s_min = max(env_diag * frac_min, 1e-6)
            s_max = env_diag * frac_max

            # Log-uniform sampling for multi-scale diversity
            u = torch.rand(num_gaussian, 3, device=self.device)
            scales = torch.exp(math.log(s_min) + u * math.log(s_max / max(s_min, 1e-8)))
            return scales.clamp(min=scale_min)

        if method == "beam_footprint":
            """Anisotropic per-mean scale derived from the antenna's known
            tilt/beamwidth geometry (see algorithm/cgr.py's context
            builder) -- not a generic multiscale draw. See
            _init_means's "tilt_aware_beam" branch for why every mean
            shares the SAME z as every query point (z-axis scale below is
            just a small fixed constant as a result, not beamwidth-derived
            at all).

            Longitudinal (propagation-direction) scale comes from where
            the vertical beam's near/far -3dB edges cross the UE plane:
                d_near = delta_h / tan(theta + beta_v/2)
                d_far  = delta_h / tan(theta - beta_v/2)   [theta > beta_v/2]
                scale_x = longitudinal_scale_coeff * |d_far - d_near| / 2
            When theta <= beta_v/2 (this repo's own tilt sweep, 0-12 deg,
            is ALWAYS in this regime against the default 65 deg beamwidth)
            the beam's shallow edge never crosses the ground at a finite
            distance, so d_far clamps to the environment's own bounding
            diagonal -- this naturally produces a LARGE longitudinal
            footprint at low tilt, which is the physically correct
            conclusion for a wide vertical beamwidth at a shallow downtilt,
            not an artifact to suppress.

            Lateral scale is the horizontal-beamwidth footprint width at
            each mean's own range (unchanged from the original proposal):
                scale_y = r_m * tan(beta_h / 2)
            """
            env_min = env_dims[:, 0].to(self.device)
            env_max = env_dims[:, 1].to(self.device)
            env_diag = float((env_max - env_min)[:2].norm().item())
            h_rx = float((env_min[2] + env_max[2]) / 2.0)

            tx = tx_position.to(self.device).float().reshape(3)
            tx_xy = tx[:2]
            delta_h = float(tx[2].item()) - h_rx

            tilt_deg = float(init_config.get("own_tilt_deg", 0.0))
            theta = tilt_deg * math.pi / 180.0
            beta_v = float(init_config.get("vertical_beamwidth_deg", 65.0)) * math.pi / 180.0
            beta_h = float(init_config.get("horizontal_beamwidth_deg", 65.0)) * math.pi / 180.0
            longitudinal_coeff = float(init_config.get("longitudinal_scale_coeff", 1.0))

            def _ground_range(angle):
                tan_a = math.tan(angle)
                return max(0.0, min(delta_h / tan_a, env_diag)) if abs(tan_a) > 1e-6 else env_diag

            d_near = _ground_range(theta + beta_v / 2.0)
            d_far = _ground_range(theta - beta_v / 2.0) if theta > beta_v / 2.0 else env_diag
            scale_x_val = longitudinal_coeff * abs(d_far - d_near) / 2.0

            r_m = torch.linalg.norm(means[:, :2] - tx_xy.unsqueeze(0), dim=-1)  # [num_gaussian]
            scale_x = torch.full((num_gaussian,), scale_x_val, device=self.device)
            scale_y = r_m * math.tan(beta_h / 2.0)
            scale_z = torch.full((num_gaussian,), scale_min, device=self.device)

            scales = torch.stack([scale_x, scale_y, scale_z], dim=-1)
            return scales.clamp(min=scale_min, max=scale_max)

        raise ValueError(f"Unsupported scale_init_method='{method}'")


    def _quaternion_from_two_vectors(self, src, dst):
        src = F.normalize(src, p=2, dim=-1)
        dst = F.normalize(dst, p=2, dim=-1)
        dot = (src * dst).sum(dim=-1, keepdim=True)
        cross = torch.cross(src, dst, dim=-1)
        q = torch.cat([1.0 + dot, cross], dim=-1)
        opposite = dot.squeeze(-1) < -0.9999
        if opposite.any():
            axis = torch.zeros((opposite.sum(), 3), device=self.device)
            axis[:, 0] = 1.0
            q[opposite] = torch.cat([torch.zeros((opposite.sum(), 1), device=self.device), axis], dim=-1)
        return F.normalize(q, p=2, dim=-1)


    def _init_rotations(self, num_gaussian, init_config, means=None, tx_position=None):
        method = str(init_config.get("rotation_init_method", "identity")).lower()
        if method == "identity":
            q = torch.zeros((num_gaussian, 4), device=self.device)
            q[:, 0] = 1.0
            return q
        if method == "random":
            q = torch.randn(num_gaussian, 4, device=self.device)
            return F.normalize(q, p=2, dim=-1)
        if method == "propagation_aligned":
            """TX-aligned rotation with controlled random perturbation.

            Physical motivation:
            In wireless propagation, the channel's spatial correlation is
            anisotropic â it varies faster perpendicular to the propagation
            direction than along it (due to phase accumulation along the path).
            Aligning Gaussians along the TXâcenter direction captures this
            anisotropy. The perturbation models multipath scattering:
              - Indoor environments: rich scattering â larger perturbation
                (concentration kappa ~ 2-5, nearly random)
              - Outdoor/LOS: directional propagation â smaller perturbation
                (concentration kappa ~ 10-50, nearly aligned)

            The perturbation follows a von Mises-Fisher-like distribution on SO(3),
            implemented via adding Gaussian noise to the aligned quaternion and
            renormalizing. The concentration parameter kappa controls the spread:
              kappa â â: no perturbation (pure TX alignment)
              kappa â 0: uniform random (isotropic scattering)
            """
            if tx_position is None or means is None:
                warnings.warn("propagation_aligned_perturbed requires tx_position and means; "
                              "falling back to random_uniform_so3.")
                return self._init_rotations(num_gaussian, {**init_config, "rotation_init_method": "random"})

            # Start with TX-aligned quaternions
            src = torch.tensor([1.0, 0.0, 0.0], device=self.device).view(1, 3).expand(num_gaussian, -1)
            dst = tx_position.to(self.device).float().reshape(1, 3) - means
            q_aligned = self._quaternion_from_two_vectors(src, dst)

            # Add perturbation: noise scale = 1/kappa
            kappa = float(init_config.get("rotation_concentration", 10.0))
            noise_scale = 1.0 / max(kappa, 1e-3)
            noise = noise_scale * torch.randn(num_gaussian, 4, device=self.device)
            q_perturbed = q_aligned + noise
            return F.normalize(q_perturbed, p=2, dim=-1)

        if method == "yaw_aligned":
            """Pure yaw (rotation about the GLOBAL z-axis only) toward the
            HORIZONTAL tx->mean azimuth -- unlike "propagation_aligned"
            (which aligns local-x to the TRUE 3D tx-mean direction,
            including height difference), this keeps local-z EXACTLY
            equal to global-z, with no twist mixing x/y into z at all.

            Required for "beam_footprint"'s own scale semantics
            (longitudinal/lateral purely in the horizontal plane, z a
            small fixed constant since every mean/query point shares one
            height -- see _init_means's "tilt_aware_beam"): with
            "propagation_aligned"'s arbitrary per-Gaussian twist, real
            horizontal displacement can leak onto the deliberately-tight
            z axis for SOME Gaussians (whichever direction the twist
            happens to land on), producing a huge, wildly inconsistent
            Mahalanobis penalty for what should have been a mild lateral
            offset -- verified empirically (predictions collapsed to
            near-zero: scripts/compare_cgr_vs_kriging.py) before this fix.
            """
            if tx_position is None or means is None:
                warnings.warn("yaw_aligned requires tx_position and means; falling back to identity.")
                return self._init_rotations(num_gaussian, {**init_config, "rotation_init_method": "identity"})
            tx = tx_position.to(self.device).float().reshape(1, 3)
            delta_xy = (tx[:, :2] - means[:, :2])  # horizontal-only direction, mean -> tx
            azimuth = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])
            half = azimuth / 2.0
            zeros = torch.zeros_like(half)
            return torch.stack([torch.cos(half), zeros, zeros, torch.sin(half)], dim=-1)

        raise ValueError(f"Unsupported rotation_init_method='{method}'")

    def init_gaussians(self, num_gaussian=None, env_dims=None, tx_position=None, wavelength=None, train_positions=None,
                       num_train=None):
        """Initialize Gaussian parameters.

        Args:
            num_gaussian: Number of Gaussian basis functions.
            env_dims: Environment boundaries [3, 2] with [min, max] per axis.
            tx_position: Transmitter position [3].
            wavelength: Carrier wavelength in meters (for physics-based methods).
            train_positions: Training sample positions [N_train, 3] (for data-aware methods).
            num_train: Number of training samples (for density-aware methods).
        """
        if num_gaussian is None:
            num_gaussian = self.num_gaussian

        init_config = self.gaussian_params

        gaussian_mean = self._init_means(num_gaussian, init_config, env_dims=env_dims, tx_position=tx_position,
                                         train_positions=train_positions)
        alpha = self._init_alpha(num_gaussian, init_config, means=gaussian_mean, tx_position=tx_position,
                                 wavelength=wavelength)
        scaling = self._init_scales(num_gaussian, init_config, means=gaussian_mean, tx_position=tx_position,
                                    env_dims=env_dims, wavelength=wavelength, num_train=num_train,)
        rotation_quat = self._init_rotations(num_gaussian, init_config, means=gaussian_mean, tx_position=tx_position)


        scaling_log = torch.log(scaling.clamp_min(1e-8))
        self._gaussian_mean = gaussian_mean
        self._scaling = scaling_log
        self._rotation_quat = rotation_quat
        self._alpha = alpha
        self.init_metadata = {
            "num_gaussian": int(num_gaussian),
            "tx_position": tx_position.detach().cpu().tolist() if isinstance(tx_position, torch.Tensor) else tx_position,
            "wavelength": float(wavelength) if wavelength is not None else None,
            "mean_init_method": init_config.get("mean_init_method", "uniform_env"),
            "scale_init_method": init_config.get("scale_init_method", "constant"),
            "rotation_init_method": init_config.get("rotation_init_method", "identity"),
            "alpha_init_method": init_config.get("alpha_init_method", "constant"),
        }

        print(f"CGR initialized with {self._gaussian_mean.shape[0]} fixed Gaussians.")

    def init_gaussians_from_csv(self, csv_path: Union[str, Path], num_gaussian: Optional[int] = None,
                                sort_by: str = "gaussian_id", descending: bool = False,):
        """Initialize Gaussian parameters from a saved Gaussian-analysis CSV.

        Required columns:
            mu_x, mu_y, mu_z,
            scale_x, scale_y, scale_z,
            quat_w, quat_x, quat_y, quat_z,
            base_activation
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Gaussian CSV file not found: {csv_path}")

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) == 0:
            raise ValueError(f"Gaussian CSV has no data rows: {csv_path}")

        required_cols = {
            "mu_x", "mu_y", "mu_z",
            "scale_x", "scale_y", "scale_z",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "base_activation",
        }
        missing = required_cols - set(rows[0].keys())
        if missing:
            raise ValueError(f"Gaussian CSV missing required columns: {sorted(missing)}")

        if sort_by not in rows[0]:
            raise ValueError(f"sort_by='{sort_by}' not found in CSV columns")

        def _to_float(v, default=0.0):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(default)

        rows.sort(key=lambda r: _to_float(r.get(sort_by, 0.0)), reverse=descending)
        if num_gaussian is not None:
            rows = rows[:int(num_gaussian)]

        means = torch.tensor([[_to_float(r["mu_x"]), _to_float(r["mu_y"]), _to_float(r["mu_z"])] for r in rows],
                             device=self.device, dtype=torch.float32,)
        scaling = torch.tensor([[_to_float(r["scale_x"]), _to_float(r["scale_y"]), _to_float(r["scale_z"])] for r in rows],
                               device=self.device, dtype=torch.float32).clamp_min(1e-8)
        rotation = torch.tensor([[_to_float(r["quat_w"]), _to_float(r["quat_x"]), _to_float(r["quat_y"]), _to_float(r["quat_z"])] for r in rows],
                                device=self.device, dtype=torch.float32,)
        rotation = F.normalize(rotation, p=2, dim=-1)
        fixed_alpha = torch.tensor( [[_to_float(r["base_activation"], self.alpha_value)] for r in rows],
                                    device=self.device, dtype=torch.float32,)

        self._gaussian_mean = means
        self._scaling = scaling
        self._rotation_quat = rotation
        self._alpha = fixed_alpha

        print(f"CGR initialized {means.shape[0]} fixed Gaussians from CSV: {csv_path}")

    def compute_feature_matrix(self, rx_positions, eps=1e-10, num_gaussian=None):
        """
        Compute Gaussian feature matrix Phi.

        Args:
            rx_positions: [B, 3]
            num_gaussian: optionally keep only the first ``num_gaussian`` basis functions

        Returns:
            Phi: [B, M] or [B, num_gaussian]
        """

        gauss_means = self.get_gaussian_mean
        _, gauss_inv_covs = self.get_covariance(return_inverse=True, eps=eps)

        if num_gaussian is not None:
            gauss_means = gauss_means[:num_gaussian]
            gauss_inv_covs = gauss_inv_covs[:num_gaussian]
            base_activation = self.get_base_activation[:num_gaussian]
        else:
            base_activation = self.get_base_activation

        rx_pos_expanded = rx_positions.unsqueeze(1)       # [B, 1, 3]
        gauss_means_expanded = gauss_means.unsqueeze(0)   # [1, M, 3]
        d_vec = rx_pos_expanded - gauss_means_expanded    # [B, M, 3]

        # Mahalanobis distance
        # result: [B, M]
        exp_val = torch.einsum("bmd,mdk,bmk->bm", d_vec, gauss_inv_covs, d_vec)
        pdf_weight = torch.exp(-0.5 * torch.clamp(exp_val, max=50.0))

        Phi = base_activation.squeeze(-1).unsqueeze(0) * pdf_weight
        return Phi

    def _flatten_channel(self, channel: torch.Tensor) -> torch.Tensor:
        """
        Convert channel targets to real-valued matrix [B, D].
        Supports complex input.
        """
        channel = channel.to(self.device)

        if torch.is_complex(channel):
            real = channel.real.reshape(channel.shape[0], -1)
            imag = channel.imag.reshape(channel.shape[0], -1)
            return torch.cat([real, imag], dim=-1).float()

        return channel.reshape(channel.shape[0], -1).float()

    def _solve_ridge_ls(self, Phi, Y, ridge_lambda):
        """
        Solve ridge-regularized least squares:
            W = argmin_W ||Phi W - Y||_F^2 + ridge_lambda * ||W||_F^2
        Closed-form solution:
            W = (Phi^T Phi + ridge_lambda I)^(-1) Phi^T Y
        Args:
            Phi: Design matrix of shape [N, M],
                 where N is the number of samples and M is the number of basis functions.
            Y: Target matrix of shape [N, D] or target vector of shape [N].
            ridge_lambda: Non-negative ridge regularization coefficient.
        Returns:
            W: Solution tensor of shape [M, D] if Y is 2D, or [M, 1] if Y is reshaped from 1D.
        """
        if ridge_lambda < 0:
            raise ValueError(f"ridge_lambda must be non-negative, got {ridge_lambda}")

        num_basis = Phi.shape[1]
        identity = torch.eye(num_basis, device=self.device, dtype=Phi.dtype)
        A = Phi.T @ Phi + ridge_lambda * identity
        B = Phi.T @ Y
        W_out = torch.linalg.solve(A, B)

        return W_out

    # TODO: refer to the paper "Configuring RNN Weights for MIMO-OFDM Receive Processing: Informing RNN With Domain Knowledge and improve the details of two-stage learning"
    def fit_two_stage_ls(self, rx_positions, channel_targets, stage1_num_gaussian, ridge_lambda=None, stage2_lr=1e-2,
                         stage2_iters=500, stage2_weight_decay=0.0, stage2_tolerance=1e-8, stage2_refine_optimizer: bool = True,
                         return_history: bool = True, val_rx_positions=None, val_channel_targets=None,
                         selection_criterion='target_correlation', val_check_interval=50):
        """
        Two-stage Gaussian basis fitting informed by domain knowledge.
            - Keep top-M1 eigenvectors U_M and project to reduced features Phi_red = Phi_full @ U_M
            - Solve ridge LS in reduced space for W_red
        Inspired by "Configuring RNN Weights for MIMO-OFDM Receive Processing: Informing RNN With Domain Knowledge":
        use domain knowledge to obtain a good closed-form initialization, then refine with gradient descent.

        Stage 1 (Closed-form in selected Gaussian subspace):
            - Score each Gaussian by how useful it is for predicting the channel
            - Select top M1 most important Gaussians directly
            - Solve ridge LS using only these M1 Gaussian columns of Phi
            - Initialize full weight vector: selected positions get LS solution,
              remaining positions get zeros

        Stage 2 (Gradient descent refinement on all M Gaussians):
            - Starting from the physically meaningful Stage 1 initialization,
              optimize all M weights via Adam
            - If validation data is provided, track validation loss and keep
              the best weights (early stopping on validation)

        Args:
            selection_criterion: How to score Gaussians for selection.
                'target_correlation' - ||Phi[:,i]^T Y||_F (how correlated each
                    Gaussian's activation is with the channel target)
                'activation_energy' - ||Phi[:,i]||^2 (how much each Gaussian
                    activates across training positions)
                'combined' - product of normalized energy and correlation
            val_rx_positions: Optional [N_val, 3] validation positions
            val_channel_targets: Optional [N_val, Nt, Nr] validation channels
            val_check_interval: Check validation loss every N iterations in Stage 2

        """
        M = self.get_gaussian_mean.shape[0]
        M1 = int(stage1_num_gaussian)

        if M1 <= 0:
            raise ValueError(f"stage1_num_gaussian must be positive, got {M1}")
        if M1 > M:
            warnings.warn(f"stage1_num_gaussian={M1} exceeds initialized Gaussian count {M}, choosen {M/2}")
            M1 = int (M // 2)

        lam = self.ridge_lambda if ridge_lambda is None else float(ridge_lambda)
        Y = self._flatten_channel(channel_targets)          # [N, D]
        D = Y.shape[1]

        # Build full feature matrix
        phi_full = self.compute_feature_matrix(rx_positions) # [N, M]

        # ============================================================
        # Stage 1: Direct Gaussian Selection + Closed-form LS
        # ============================================================
        # Score each Gaussian by its relevance
        if selection_criterion == 'activation_energy':
            scores = phi_full.pow(2).sum(dim=0)  # [M]
        elif selection_criterion == 'target_correlation':
            projections = phi_full.T @ Y  # [M, D]
            scores = projections.norm(dim=1)  # [M]
        elif selection_criterion == 'combined':
            energy = phi_full.pow(2).sum(dim=0)
            projections = phi_full.T @ Y
            correlation = projections.norm(dim=1)
            energy_n = energy / (energy.max() + 1e-10)
            corr_n = correlation / (correlation.max() + 1e-10)
            scores = energy_n * corr_n
        else:
            raise ValueError(f"Unknown selection_criterion: {selection_criterion}")

        # Select top M1 Gaussians
        _, selected_idx = torch.topk(scores, M1)
        selected_idx, _ = torch.sort(selected_idx)           # sorted for consistency

        # Solve ridge LS using only the selected Gaussian columns
        phi_selected = phi_full[:, selected_idx]              # [N, M1]
        w_selected = self._solve_ridge_ls(phi_selected, Y, lam)  # [M1, D]

        # Initialize full weight vector: selected get LS solution, rest zero
        w_full_init = torch.zeros(M, D, device=self.device, dtype=Y.dtype)
        w_full_init[selected_idx] = w_selected

        # Cache for analysis / debugging
        self._selected_indices = selected_idx.detach()
        self._selection_scores = scores.detach()
        self._reduced_dim = M1
        w_out = w_full_init.clone()

        loss_history = []

        with torch.no_grad():
            init_residual = phi_full @ w_out - Y
            init_loss = init_residual.pow(2).mean() + lam * w_out.pow(2).mean()
            loss_history.append(float(init_loss))

        # ============================================================
        # Stage 2: Gradient Descent Refinement on All M Gaussians
        # ============================================================

        # Prepare validation data if provided
        has_val = (val_rx_positions is not None and val_channel_targets is not None)
        if has_val:
            val_Y = self._flatten_channel(val_channel_targets)
            phi_val = self.compute_feature_matrix(val_rx_positions)
            best_val_loss = float('inf')
            best_w_out = w_out.detach().clone()

        if stage2_refine_optimizer and stage2_iters > 0:
            w_param = nn.Parameter(w_out.detach().clone())
            optimizer = torch.optim.Adam([w_param], lr=stage2_lr, weight_decay=stage2_weight_decay)
            prev_loss = None

            for iteration in range(stage2_iters):
                optimizer.zero_grad()
                residual = phi_full @ w_param - Y
                data_loss = residual.pow(2).mean()
                ridge_loss = lam * w_param.pow(2).mean()
                loss = data_loss + ridge_loss
                loss.backward()
                optimizer.step()

                loss_value = float(loss.detach())
                loss_history.append(loss_value)

                # Track best weights based on validation loss
                if has_val and (iteration + 1) % val_check_interval == 0:
                    with torch.no_grad():
                        val_pred = phi_val @ w_param
                        val_residual = val_pred - val_Y
                        val_loss = val_residual.pow(2).mean()
                        if val_loss < best_val_loss:
                            best_val_loss = float(val_loss)
                            best_w_out = w_param.detach().clone()

                # if prev_loss is not None and abs(prev_loss - loss_value) <= stage2_tolerance:
                #     break
                prev_loss = loss_value
            w_out = w_param.detach()

        # Use best validation weights if available
        if has_val:
            with torch.no_grad():
                val_loss_final = (phi_val @ w_out - val_Y).pow(2).mean()
                if val_loss_final < best_val_loss:
                    best_w_out = w_out.clone()
            w_out = best_w_out

        self._W_out = w_out.detach()

        result = {
            "stage1_weights": w_full_init.detach(),
            "stage2_weights": self._W_out.detach(),
            "reduced_weights": w_selected.detach(),
            "selected_indices": selected_idx.detach(),
            "reduced_dim": M1,
            "full_num_gaussian": int(M),
            "loss_history": loss_history,
        }
        return result if return_history else self._W_out


    def _normalize_feature_columns(self, Phi: torch.Tensor, eps: float = 1e-12) -> tuple:
        """
        Column-normalizes Phi to unit L2 norm per Gaussian:
            s_m = sqrt(sum_n Phi_nm^2 + eps),  Phi_tilde_nm = Phi_nm / s_m

        Ridge regularization (lambda * ||W||^2) penalizes every column of
        W equally in absolute terms, which is only fair if every column
        of Phi is on a comparable scale -- NOT true here (per-Gaussian
        activation across a typical known-point set spans roughly 1e-12
        to 1e-2, ten orders of magnitude, since alpha/scale vary a lot
        per Gaussian), a major source of the ill-conditioning behind
        several earlier bugs in this fit.

        :output: (Phi_tilde [B, M], s [M]) -- s must be divided back out
            of the recovered W (see fit_ls) so the result stays compatible
            with the UNNORMALIZED Phi predict_flat uses.
        """
        s = torch.sqrt(torch.sum(Phi ** 2, dim=0) + eps)  # [M]
        return Phi / s.unsqueeze(0), s

    def fit_ls(self, rx_positions: torch.Tensor, channel_targets: torch.Tensor, ridge_lambda: Optional[float] = None,
              normalize_columns: bool = False):
        """
        Solve LS output layer:
            W = (Phi^T Phi + lambda I)^(-1) Phi^T Y

        :param normalize_columns: column-normalize Phi before solving (see
            _normalize_feature_columns), un-normalizing W_out back after.
            CAUTION under an alpha_init_method that produces near-dead
            (near-zero-everywhere) columns (e.g. "distance"'s amplitude-
            domain path-loss decay): un-normalizing an UNCONSTRAINED
            weight by dividing back through a tiny column norm can
            amplify an already poorly-determined weight by 1000x+, with
            no sparsity/shrinkage to catch it -- verified empirically to
            spike MAE from ~13 dB to 220 dB in that
            configuration. Safe to enable with alpha_init_method=
            "constant" (no columns going artificially near-zero from an
            amplitude-domain decay) -- re-verify empirically (MAE/RMSE
            against ground truth, not just coverage match) before trusting
            it in any other alpha configuration.

        rx_positions:   [B, 3]
        channel_targets:[B, Nt, Nr] complex or any [B, ...]
        """
        lam = self.ridge_lambda if ridge_lambda is None else ridge_lambda

        Phi = self.compute_feature_matrix(rx_positions)   # [B, M]
        Y = self._flatten_channel(channel_targets)        # [B, D]

        M = Phi.shape[1]
        I = torch.eye(M, device=self.device, dtype=Phi.dtype)

        if normalize_columns:
            Phi_tilde, s = self._normalize_feature_columns(Phi)
            A = Phi_tilde.T @ Phi_tilde + lam * I
            B = Phi_tilde.T @ Y
            w_tilde = torch.linalg.solve(A, B)
            self._W_out = w_tilde / s.unsqueeze(1)
        else:
            A = Phi.T @ Phi + lam * I
            B = Phi.T @ Y
            self._W_out = torch.linalg.solve(A, B)        # [M, D]

        return self._W_out

    def predict_flat(self, rx_positions: torch.Tensor) -> torch.Tensor:
        if self._W_out.numel() == 0:
            raise RuntimeError("LS output layer not fitted. Call fit_ls() first.")
        Phi = self.compute_feature_matrix(rx_positions)   # [B, M]
        return Phi @ self._W_out                          # [B, D]

    def predict_channel(self, rx_positions):
        pred_flat = self.predict_flat(rx_positions)
        batch_size = pred_flat.shape[0]
        half = self.num_tx_ant * self.num_rx_ant

        real = pred_flat[:, :half].reshape(batch_size, self.num_tx_ant, self.num_rx_ant)
        imag = pred_flat[:, half:].reshape(batch_size, self.num_tx_ant, self.num_rx_ant)
        channel_pred = torch.complex(real, imag)

        return channel_pred

    def save(self, filepath, cfg=None):
        state_dict = {
            "gaussian_mean": self._gaussian_mean.detach().cpu(),
            "rotation": self._rotation_quat.detach().cpu(),
            "scaling": self._scaling.detach().cpu(),
            "alpha": self._alpha.detach().cpu(),
            "W_out": self._W_out.detach().cpu(),
            "init_metadata": self.init_metadata,
            "config_dict": cfg,
        }
        torch.save(state_dict, filepath)

    @torch.no_grad()
    def _gaussian_contribution_stats(self, nt, nr):
        """Return per-Gaussian contribution statistics from LS weights.

        Returns:
            (c_i_l2, c_i_mean_abs), each with shape [M]
        """
        nt = int(nt or self.num_tx_ant)
        nr = int(nr or self.num_rx_ant)

        expected_dim = 2 * nt * nr
        if self._W_out.shape[1] != expected_dim:
            raise ValueError(f"W_out feature dimension mismatch: expected {expected_dim}, got {self._W_out.shape[1]}")

        weights = self._W_out  # [M, 2*Nt*Nr]
        half = nt * nr
        real = weights[:, :half].reshape(-1, nt, nr)
        imag = weights[:, half:].reshape(-1, nt, nr)
        contrib_cplx = torch.complex(real, imag)

        c_i_l2 = weights.norm(dim=1)
        c_i_mean_abs = contrib_cplx.abs().mean(dim=(1, 2))
        return c_i_l2, c_i_mean_abs

    @torch.no_grad()
    def save_gaussian_statistics_csv(self, csv_path, nt, nr, eps: float = 1e-10):
        """Save per-Gaussian statistics to a CSV file."""
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        nt = int(nt or self.num_tx_ant)
        nr = int(nr or self.num_rx_ant)

        means = self.get_gaussian_mean
        covariance, inv_covariance = self.get_covariance(return_inverse=True, eps=eps)
        scaling = self.get_scaling
        rotation = self.get_rotation
        base_activation = self.get_base_activation
        c_i_l2, c_i_mean_abs = self._gaussian_contribution_stats(nt=nt, nr=nr)

        fieldnames = [
            "gaussian_id", "mu_x", "mu_y", "mu_z",
            "scale_x", "scale_y", "scale_z",
            "quat_w", "quat_x", "quat_y", "quat_z",
            "base_activation",
            "cov_xx", "cov_xy", "cov_xz", "cov_yx", "cov_yy", "cov_yz", "cov_zx", "cov_zy", "cov_zz",
            "inv_cov_xx", "inv_cov_xy", "inv_cov_xz", "inv_cov_yx", "inv_cov_yy", "inv_cov_yz", "inv_cov_zx", "inv_cov_zy", "inv_cov_zz",
            "cov_trace", "cov_det", "C_i_l2", "C_i_mean_abs"
        ]

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for idx in range(means.shape[0]):
                cov = covariance[idx]
                inv_cov = inv_covariance[idx]
                writer.writerow({
                    "gaussian_id": int(idx),
                    "mu_x": float(means[idx, 0]), "mu_y": float(means[idx, 1]), "mu_z": float(means[idx, 2]),
                    "scale_x": float(scaling[idx, 0]), "scale_y": float(scaling[idx, 1]), "scale_z": float(scaling[idx, 2]),
                    "quat_w": float(rotation[idx, 0]), "quat_x": float(rotation[idx, 1]),
                    "quat_y": float(rotation[idx, 2]), "quat_z": float(rotation[idx, 3]),
                    "base_activation": float(base_activation[idx, 0]),
                    # "base_activation_logit": float(base_logits[idx, 0]),
                    "cov_xx": float(cov[0, 0]), "cov_xy": float(cov[0, 1]), "cov_xz": float(cov[0, 2]),
                    "cov_yx": float(cov[1, 0]), "cov_yy": float(cov[1, 1]), "cov_yz": float(cov[1, 2]),
                    "cov_zx": float(cov[2, 0]), "cov_zy": float(cov[2, 1]), "cov_zz": float(cov[2, 2]),
                    "inv_cov_xx": float(inv_cov[0, 0]), "inv_cov_xy": float(inv_cov[0, 1]), "inv_cov_xz": float(inv_cov[0, 2]),
                    "inv_cov_yx": float(inv_cov[1, 0]), "inv_cov_yy": float(inv_cov[1, 1]), "inv_cov_yz": float(inv_cov[1, 2]),
                    "inv_cov_zx": float(inv_cov[2, 0]), "inv_cov_zy": float(inv_cov[2, 1]), "inv_cov_zz": float(inv_cov[2, 2]),
                    "cov_trace": float(torch.trace(cov)),
                    "cov_det": float(torch.det(cov)),
                    "C_i_l2": float(c_i_l2[idx]),
                    "C_i_mean_abs": float(c_i_mean_abs[idx]),
                })

        return csv_path

    @torch.no_grad()
    def plot_gaussian_3d(self, figure_path: Union[str, Path], tx_position: torch.Tensor,
                         nt: Optional[int] = None, nr: Optional[int] = None,
                         max_gaussians: int = 20) -> Path:
        """Draw trained Gaussians in 3D with color mapped to contribution strength C_i."""
        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)

        nt = int(nt or self.num_tx_ant)
        nr = int(nr or self.num_rx_ant)

        means = self.get_gaussian_mean.detach().cpu()
        covariance = self.get_covariance(return_inverse=False).detach().cpu()


        _, c_i_mean_abs = self._gaussian_contribution_stats(nt=nt, nr=nr)
        c_i = c_i_mean_abs.detach().cpu()


        n = means.shape[0]
        if n == 0:
            raise ValueError("Cannot plot Gaussian 3D visualization because no Gaussians are initialized.")

        num_show = min(max_gaussians, n)
        top_idx = torch.topk(c_i, k=num_show, largest=True).indices
        means = means[top_idx]
        covariance = covariance[top_idx]
        c_i = c_i[top_idx]

        norm = Normalize(vmin=float(c_i.min()), vmax=float(c_i.max()) + 1e-12)
        cmap = plt.cm.viridis

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(means[:, 0], means[:, 1], means[:, 2], c=c_i.numpy(), cmap=cmap, s=40)

        u = np.linspace(0, 2 * np.pi, 24)
        v = np.linspace(0, np.pi, 12)
        sphere_x = np.outer(np.cos(u), np.sin(v))
        sphere_y = np.outer(np.sin(u), np.sin(v))
        sphere_z = np.outer(np.ones_like(u), np.cos(v))

        for idx in range(num_show):
            eigvals, eigvecs = torch.linalg.eigh(covariance[idx])
            radii = torch.sqrt(torch.clamp(eigvals, min=1e-12)).numpy()
            ellipsoid = np.stack([sphere_x * radii[0], sphere_y * radii[1], sphere_z * radii[2]], axis=-1)
            ellipsoid = ellipsoid @ eigvecs.numpy().T

            x = ellipsoid[:, :, 0] + means[idx, 0].item()
            y = ellipsoid[:, :, 1] + means[idx, 1].item()
            z = ellipsoid[:, :, 2] + means[idx, 2].item()

            color = cmap(norm(float(c_i[idx])))
            ax.plot_wireframe(x, y, z, rstride=2, cstride=2, color=color, linewidth=0.4, alpha=0.35)

        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(c_i.numpy())
        cbar = fig.colorbar(mappable, ax=ax, pad=0.12, shrink=0.8)
        cbar.set_label("C_i (mean |complex contribution|)")

        ax.set_title(f"Trained Gaussian Functions (Top {num_show} by C_i)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(figure_path, dpi=220)
        plt.close(fig)

        return figure_path

    @classmethod
    def load(cls, filepath, device: torch.device, **model_cfg):
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        state_dict = torch.load(str(filepath), map_location=device, weights_only=False)

        if "config_dict" in state_dict and state_dict["config_dict"] is not None:
            config_dict = state_dict["config_dict"]
            data_params = config_dict["data_params"]
            model_params = config_dict["model_params"]
            model = cls(
                num_tx_ant=data_params["num_tx_ant"],
                num_rx_ant=data_params["num_rx_ant"],
                device=device,
                **model_params
            )
        else:
            model = cls(device=device, **model_cfg)

        model._gaussian_mean = state_dict["gaussian_mean"].to(device)
        model._rotation_quat = state_dict["rotation"].to(device)
        model._scaling = state_dict["scaling"].to(device)
        model._alpha = state_dict["alpha"].to(device)
        model._W_out = state_dict["W_out"].to(device)
        model.init_metadata = state_dict.get("init_metadata", {})

        print(f"nGRFLS loaded from {filepath}.")
        print(f"Loaded model has {model.get_gaussian_mean.shape[0]} Gaussians.")
        return model
