import unittest
from support import np
from helpers.spatial_grid_estimator import SpatialGridEstimator, pool_observation_windows

def estimator(**kwargs):
    return SpatialGridEstimator(2, 4, np.array([[[0.,0.], [0.,1.], [1.,0.], [1.,1.]]]*2),
                                1., 0., **kwargs)

class SpatialContract(unittest.TestCase):
    def test_constructor_validation(self):
        for kwargs in ({'radio_map_method':'invalid'}, {'radio_map_method':'rsrp_per_sector'}):
            with self.assertRaises((AssertionError, ValueError)):
                estimator(**kwargs)

    def test_invalid_rows_excluded_and_selective_reset(self):
        e = estimator()
        e.accumulate(np.array([0, 1, -1, 0, 1]), np.array([0, 1, 0, -1, 4]),
                     np.array([True, False, True, True, True]), sinr_db=np.array([10., -10., 2., 2., 2.]))
        np.testing.assert_array_equal(e.visit_count.sum(axis=1), [1, 1])
        _, coverage, reward = e.compute(np.array([True, False]))
        np.testing.assert_array_equal(e.visit_count.sum(axis=1), [0, 1])
        np.testing.assert_array_equal(e.covered_count[0], np.zeros(4))
        self.assertTrue(np.isfinite(reward[0]))
        _, _, reward = e.compute(np.array([True, True]))
        self.assertTrue(np.isnan(reward[0]))
        self.assertTrue(np.isfinite(reward[1]))
        self.assertFalse(e.visit_count.any())

    def test_computed_coverage_range_and_uncomputed_sentinel(self):
        e = estimator()
        e.accumulate(np.array([0]), np.array([0]), np.array([True]), sinr_db=np.array([10.]))
        _, coverage, _ = e.compute(np.array([True, False]))
        self.assertTrue(np.all((coverage[0] >= 0) & (coverage[0] <= 1)),
                        f"Computed coverage must be in [0,1]; observed={coverage[0]}")
        np.testing.assert_array_equal(coverage[1], np.full(4, -1.))

    def test_occupancy_ema_convex_combination(self):
        e = estimator(occupancy_ema_alpha=.25)
        e.accumulate(np.array([0, 0]), np.array([0, 0]), np.ones(2, bool), sinr_db=np.ones(2)*10)
        old = e.compute(np.ones(2, bool))[0].copy()
        e.accumulate(np.array([0, 0]), np.array([1, 1]), np.ones(2, bool), sinr_db=np.ones(2)*10)
        new = e.compute(np.ones(2, bool))[0]
        # Equal sample counts let the first-window mass establish normalization.
        target = np.zeros(4)
        target[1] = old[0].sum()
        np.testing.assert_allclose(new[0], .25*target + .75*old[0])

    def test_window_pooling_sum_versus_mean(self):
        names = ['center','edge_beta0','edge_alpha0','edge_alphaN','edge_betaN',
                 'corner_C','corner_E1','corner_Far','corner_E2']
        windows = [{name: [(s, 0), (1-s, 2)] for name in names} for s in range(2)]
        occ = np.arange(8).reshape(2, 4)
        cov = occ/8.
        out_o, out_c, out_names = pool_observation_windows(occ, cov, windows)
        self.assertEqual(out_o.shape, (2, 9))
        self.assertEqual(out_c.shape, (2, 9))
        self.assertEqual(set(out_names), set(names))
        for s in range(2):
            for j, name in enumerate(out_names):
                members = windows[s][name]
                self.assertAlmostEqual(out_o[s,j], sum(occ[i,c] for i,c in members))
                self.assertAlmostEqual(out_c[s,j], np.mean([cov[i,c] for i,c in members]))

if __name__ == '__main__':
    unittest.main()
