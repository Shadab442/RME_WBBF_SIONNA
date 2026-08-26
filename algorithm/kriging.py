"""Ordinary kriging spatial interpolator -- the best linear unbiased
predictor of a spatial field from scattered observations, under a
Gaussian/RBF covariance model. Mathematically the same model as Gaussian
Process regression with an RBF kernel (matches the RBF kernel choice from
the earlier radio-map SINR estimation design discussion).
"""

import numpy as np


class OrdinaryKriging:
    """:param length_scale: RBF covariance length scale [m] -- how far
        apart two points can be before their values are treated as
        essentially uncorrelated. V1 default, not yet empirically tuned.
    :param variance: covariance at zero distance (the field's own variance).
    :param nugget: small diagonal regularization -- avoids a singular
        covariance matrix when two observations coincide or are very close.
    """

    def __init__(self, length_scale: float, variance: float = 1.0, nugget: float = 1e-6):
        self.length_scale = length_scale
        self.variance = variance
        self.nugget = nugget

    def _covariance(self, dist: np.ndarray) -> np.ndarray:
        return self.variance * np.exp(-0.5 * (dist / self.length_scale) ** 2)

    def predict(self, known_xy: np.ndarray, known_values: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        """:param known_xy: [n, 2] observed locations.
        :param known_values: [n] observed values at those locations.
        :param target_xy: [m, 2] locations to predict at.
        :output: [m] predicted values -- the unbiased, minimum-variance
            linear predictor. Degrades gracefully: n=1 just returns that
            one value everywhere (a constant predictor). n=0 is the
            caller's responsibility to avoid -- there's nothing to
            interpolate from.
        """
        n = known_xy.shape[0]
        if n == 0:
            raise ValueError("OrdinaryKriging.predict needs at least one observation")

        # Covariance among known points, regularized
        d_known = np.linalg.norm(known_xy[:, None, :] - known_xy[None, :, :], axis=-1)
        c_known = self._covariance(d_known) + self.nugget * np.eye(n)

        # Augmented system enforcing unbiasedness (weights sum to 1)
        a = np.ones((n + 1, n + 1))
        a[:n, :n] = c_known
        a[n, n] = 0.0

        # Covariance between known points and targets
        d_target = np.linalg.norm(known_xy[:, None, :] - target_xy[None, :, :], axis=-1)  # [n, m]
        b = np.ones((n + 1, target_xy.shape[0]))
        b[:n, :] = self._covariance(d_target)

        weights = np.linalg.solve(a, b)  # [n+1, m]
        return known_values @ weights[:n, :]
