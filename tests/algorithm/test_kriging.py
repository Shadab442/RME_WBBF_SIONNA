"""Step-by-step verification of the real OrdinaryKriging implementation.

Coverage:
- RBF covariance values, variance/length-scale effects, shapes, and symmetry.
- Pairwise Euclidean distances passed to covariance, diagonal-only nugget,
  augmented unbiasedness constraint, and known-to-target covariance orientation.
- Actual solve residual, weights summing to one, and final weighted prediction.
- Independent two-point analytical weights; exact interpolation without nugget;
  constant/affine-value reproduction, permutation and geometric invariance.
- Single/no observations, empty targets, duplicate locations with regularization,
  and caller-input preservation.

Spies observe covariance/solve inputs and outputs while the real methods run.
The independent two-point oracle uses a closed-form formula, not np.linalg.solve.
This does not establish that an RBF model fits a particular measured radio field.
Run from the repository root: python tests/algorithm/test_kriging.py
"""
import logging
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from algorithm.kriging import OrdinaryKriging


class CovarianceTests(unittest.TestCase):
    def test_known_rbf_values_and_shape(self):
        # At distances 0, ell, 2*ell the normalized covariance is 1, exp(-.5), exp(-2).
        k = OrdinaryKriging(2., variance=3.)
        distances = np.array([[0.,2.],[4.,20.]])
        expected = 3*np.array([[1.,math.exp(-.5)],[math.exp(-2),math.exp(-50)]])
        actual = k._covariance(distances)
        np.testing.assert_allclose(actual, expected, rtol=1e-13)
        self.assertEqual(actual.shape, distances.shape)
        self.assertAlmostEqual(float(k._covariance(np.array(2.))),3*math.exp(-.5))

    def test_variance_scaling_length_scale_and_monotonicity(self):
        # More distant locations correlate less; a longer length scale correlates more.
        distances = np.array([0.,1.,2.,4.,8.])
        base = OrdinaryKriging(2.)._covariance(distances)
        self.assertTrue(np.all(np.diff(base)<0))
        self.assertTrue(np.all(base>0))
        np.testing.assert_allclose(OrdinaryKriging(2.,variance=5.)._covariance(distances),5*base)
        self.assertTrue(np.all(OrdinaryKriging(4.)._covariance(distances[1:])>base[1:]))
        # Nugget is added to the observation matrix, not inside the RBF function.
        np.testing.assert_array_equal(OrdinaryKriging(2.,nugget=9.)._covariance(distances),base)

    def test_covariance_matrix_symmetry_and_positive_definiteness(self):
        # Distinct locations under this positive-variance RBF give a symmetric positive-definite matrix.
        xy = np.array([[0.,0.],[1.,2.],[4.,1.]])
        distance = np.sqrt(((xy[:,None]-xy[None,:])**2).sum(axis=-1))
        covariance = OrdinaryKriging(2.,variance=3.)._covariance(distance)
        np.testing.assert_allclose(covariance,covariance.T)
        np.testing.assert_allclose(np.diag(covariance),3.)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance)>0))


class PredictionStepsTests(unittest.TestCase):
    def test_distances_augmented_system_weights_and_output(self):
        # A 3-4-5 triangle makes all expected known distances independently recognizable.
        k = OrdinaryKriging(2.,variance=3.,nugget=.2)
        xy = np.array([[0.,0.],[3.,0.],[0.,4.]])
        values = np.array([2.,-1.,8.])
        targets = np.array([[0.,0.],[3.,4.]])
        known_dist = np.array([[0.,3.,4.],[3.,0.,5.],[4.,5.,0.]])
        target_dist = np.array([[0.,5.],[3.,4.],[4.,3.]])
        solved = []
        real_solve = np.linalg.solve

        def record_solve(a,b):
            result = real_solve(a,b)
            solved.append((a.copy(),b.copy(),result.copy()))
            return result

        # Observe the real calculation without substituting covariance or solver results.
        with patch.object(k,'_covariance',wraps=k._covariance) as covariance_spy:
            with patch('numpy.linalg.solve',side_effect=record_solve) as solve_spy:
                prediction = k.predict(xy,values,targets)
        self.assertEqual(covariance_spy.call_count,2)
        np.testing.assert_allclose(covariance_spy.call_args_list[0].args[0],known_dist)
        np.testing.assert_allclose(covariance_spy.call_args_list[1].args[0],target_dist)
        solve_spy.assert_called_once()
        a,b,solution = solved[0]

        # Verify RBF entries independently with scalar math; nugget belongs only on the known diagonal.
        expected_c = np.array([[3*math.exp(-d*d/8) for d in row] for row in known_dist])
        np.testing.assert_allclose(a[:3,:3],expected_c+.2*np.eye(3))
        np.testing.assert_array_equal(a[3,:],[1,1,1,0])
        np.testing.assert_array_equal(a[:,3],[1,1,1,0])
        expected_cross = np.array([[3*math.exp(-d*d/8) for d in row] for row in target_dist])
        np.testing.assert_allclose(b[:3],expected_cross)
        np.testing.assert_array_equal(b[3],np.ones(2))
        self.assertEqual(a.shape,(4,4))
        self.assertEqual(b.shape,(4,2))

        # The real solve must satisfy both the covariance equations and sum(weights)=1.
        np.testing.assert_allclose(a@solution,b,atol=1e-12)
        np.testing.assert_allclose(solution[:3].sum(axis=0),np.ones(2),atol=1e-12)
        # Only observation weights contribute; the last row is the Lagrange multiplier.
        expected_prediction = np.array([sum(values[i]*solution[i,j] for i in range(3)) for j in range(2)])
        np.testing.assert_allclose(prediction,expected_prediction)
        self.assertEqual(prediction.shape,(2,))

    def test_two_point_closed_form_prediction(self):
        # For equal diagonal c, off-diagonal k, and cross-covariances b0,b1:
        # w0 = 1/2 + (b0-b1)/(2*(c-k)); w1 = 1-w0.
        for nugget in (0.,.3):
            with self.subTest(nugget=nugget):
                k = OrdinaryKriging(2.,variance=3.,nugget=nugget)
                xs = [0.,.5,1.,2.,5.]
                targets = np.column_stack([xs,np.zeros(len(xs))])
                expected = []
                for x in xs:
                    b0 = 3*math.exp(-x*x/8)
                    b1 = 3*math.exp(-(x-2)**2/8)
                    w0 = .5+(b0-b1)/(2*(3+nugget-3*math.exp(-.5)))
                    expected.append(2*w0+8*(1-w0))
                actual = k.predict(np.array([[0.,0.],[2.,0.]]),np.array([2.,8.]),targets)
                np.testing.assert_allclose(actual,expected,atol=1e-12)
                self.assertAlmostEqual(actual[2],5.)  # Midpoint symmetry.


class PredictionPropertiesTests(unittest.TestCase):
    def setUp(self):
        self.xy = np.array([[0.,0.],[2.,0.],[0.,3.]])
        self.values = np.array([2.,-1.,8.])
        self.targets = np.array([[.5,1.],[4.,2.]])
        self.k = OrdinaryKriging(2.)

    def test_exact_interpolation_without_nugget(self):
        # With no observation regularization, predictions at distinct known locations reproduce the data.
        actual = OrdinaryKriging(2.,nugget=0.).predict(self.xy,self.values,self.xy)
        np.testing.assert_allclose(actual,self.values,atol=1e-12)

    def test_constant_and_affine_value_reproduction(self):
        # Unbiasedness preserves a constant field, even far from observations.
        targets = np.vstack([self.targets,[[1000.,1000.]]])
        np.testing.assert_allclose(self.k.predict(self.xy,np.full(3,7.),targets),7.,atol=1e-12)
        # Scaling and shifting the observed values must scale/shift the prediction identically.
        baseline = self.k.predict(self.xy,self.values,targets)
        np.testing.assert_allclose(self.k.predict(self.xy,-2*self.values+5,targets),-2*baseline+5,atol=1e-12)

    def test_observation_and_target_permutation(self):
        # Reordering observations must not affect estimates; reordering targets only reorders output.
        expected = self.k.predict(self.xy,self.values,self.targets)
        order = [2,0,1]
        np.testing.assert_allclose(self.k.predict(self.xy[order],self.values[order],self.targets),expected)
        np.testing.assert_allclose(self.k.predict(self.xy,self.values,self.targets[::-1]),expected[::-1])

    def test_translation_rotation_and_input_preservation(self):
        # An isotropic distance-based model is unchanged by rigid coordinate transformations.
        originals = [x.copy() for x in (self.xy,self.values,self.targets)]
        expected = self.k.predict(self.xy,self.values,self.targets)
        rotation = np.array([[0.,-1.],[1.,0.]])
        np.testing.assert_allclose(self.k.predict(self.xy@rotation+10,self.values,self.targets@rotation+10),expected)
        for actual,original in zip((self.xy,self.values,self.targets),originals):
            np.testing.assert_array_equal(actual,original)

    def test_one_observation_is_constant(self):
        # Ordinary kriging's sum-to-one constraint makes a lone observation's weight exactly one.
        actual = self.k.predict(self.xy[:1],np.array([4.]),np.array([[0.,0.],[1000.,1000.]]))
        np.testing.assert_allclose(actual,[4.,4.])

    def test_no_observations_rejected(self):
        # Prediction without data is explicitly disallowed by the public contract.
        with self.assertRaises(ValueError):
            self.k.predict(np.empty((0,2)),np.empty(0),self.targets)

    def test_empty_targets_return_empty_vector(self):
        # Valid observations with no requested locations should yield shape (0,), not a scalar/error.
        actual = self.k.predict(self.xy,self.values,np.empty((0,2)))
        self.assertEqual(actual.shape,(0,))

    def test_duplicate_locations_regularized_and_symmetric(self):
        # Positive nugget makes identical locations solvable; identical covariance rows give equal weights.
        k = OrdinaryKriging(2.,nugget=.1)
        actual = k.predict(np.array([[0.,0.],[0.,0.]]),np.array([2.,8.]),self.targets)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual,[5.,5.],atol=1e-12)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
