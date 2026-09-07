import unittest
from support import np, manager, state
from helpers.kpi_manager import IntervalMeasurementPool

class KpiContract(unittest.TestCase):
    def test_power_is_repeatable_positive_and_sweep_equivalent(self):
        k, s = manager(), state()
        tilts = np.array([-3., 0., 8.])
        p = k.resolve_power_at_tilt(s, tilts)
        self.assertEqual(p.shape, (2, 3, 5))
        self.assertTrue(np.isfinite(p).all() and (p > 0).all())
        np.testing.assert_array_equal(p, k.resolve_power_at_tilt(s, tilts))
        k.resolve_power_at_tilt(s, np.full(3, 5.))
        np.testing.assert_array_equal(p, k.resolve_power_at_tilt(s, tilts))
        sweep = k.compute_tilt_sector_ue_rx_power(s, [-3., 0., 8.])
        expected = np.stack([k.resolve_power_at_tilt(s, np.full(3, t)) for t in tilts])
        np.testing.assert_array_equal(sweep, expected)
        np.testing.assert_array_equal(k.compute_tilt_sector_ue_rx_power(s), expected[-1])

    def test_attachment_threshold_counts_and_empty_sector(self):
        k = manager()
        power = np.array([[[9., 1., 4., 9.], [1., 8., 4., 1.], [.1, .1, .1, .1]]])*1e-9
        result = k.compute_ue_kpis(power, threshold_db=0., ut_loc=np.zeros((1, 4, 3)), return_counts=True)
        serving = power.argmax(axis=1)
        np.testing.assert_array_equal(result['serving_idx'], serving)
        signal = power.max(axis=1)
        np.testing.assert_allclose(result['rsrp_dbm'], 10*np.log10(signal)+30, atol=1e-5)
        self.assertEqual(result['sinr_db'].shape, (1, 4))
        self.assertAlmostEqual(result['coverage'], np.mean(result['sinr_db'] > 0.))
        self.assertEqual(np.sum(result['per_sector_served_count']), 4)
        self.assertTrue(np.all(result['per_sector_covered_count'] <= result['per_sector_served_count']))
        self.assertTrue(np.isnan(result['per_sector_coverage'][2]))
        for value in result['per_sector_coverage'][:2]:
            self.assertTrue(0 <= value <= 1)
        boundary = float(result['sinr_db'][0, 0])
        strict = k.compute_ue_kpis(power, threshold_db=boundary)
        self.assertEqual(strict['coverage'], np.mean(strict['sinr_db'] > boundary))

    def test_tilt_table_selection(self):
        k = manager()
        table = np.random.default_rng(42).uniform(1e-10, 1e-8, (3, 2, 3, 7))
        indices = np.array([2, 0, 1])
        selected = np.stack([table[indices[i], :, i, :] for i in range(3)], axis=1)
        expected = k.compute_ue_kpis(selected)
        actual = k.compute_ue_kpis(table, tilt_idx_per_sector=indices)
        for key in ('sinr_db', 'rsrp_dbm', 'serving_idx'):
            np.testing.assert_allclose(actual[key], expected[key])

    def test_interval_concatenation_and_reset(self):
        k, s = manager(), state(batch=1)
        pool = IntervalMeasurementPool(k, 3)
        pool.start_interval()
        positions = np.zeros((5, 3))
        first = pool.pool_measurement(s, np.zeros(3), np.ones(3), [0., 4.], positions)
        self.assertEqual(first.shape, (3, 5))
        one = {key: np.array(value, copy=True) for key, value in pool.pooled_interval().items()}
        pool.pool_measurement(s, np.zeros(3), np.ones(3), [0., 4.], positions)
        two = pool.pooled_interval()
        for key in ('power_table', 'adaptive_power_w_r0', 'drl_power_w_r0', 'no_tilt_power_w', 'ut_loc_r0'):
            axis = {'power_table': 1, 'no_tilt_power_w': 0, 'ut_loc_r0': 0}.get(key, -1)
            np.testing.assert_array_equal(two[key], np.concatenate([one[key], one[key]], axis=axis))
        pool.start_interval()
        try:
            empty = pool.pooled_interval()
        except (ValueError, RuntimeError):
            return
        self.assertTrue(all(v is None or np.asarray(v).size == 0 for v in empty.values()))

if __name__ == '__main__':
    unittest.main()
