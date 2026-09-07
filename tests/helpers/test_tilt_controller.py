import unittest
from unittest.mock import Mock
from support import np, manager
from helpers.tilt_controller import (GlobalTiltSelector, LocalTiltSelector,
    DynamicTiltController, AdaptiveLegacyTiltController, RLTiltController)

class ControllerContract(unittest.TestCase):
    def setUp(self):
        self.k = manager()
        self.table = np.random.default_rng(7).uniform(1e-11, 1e-8, (4, 1, 3, 40))

    def coverage(self, assignment):
        return self.k.compute_ue_kpis(self.table, tilt_idx_per_sector=assignment, threshold_db=0.)['coverage']

    def test_global_matches_exhaustive_uniform_search(self):
        selector = GlobalTiltSelector()
        result = selector.select(self.k, self.table, 0.)
        self.assertTrue(np.all(result == result[0]))
        self.assertEqual(self.coverage(result), max(self.coverage(np.full(3,i)) for i in range(4)))
        for warm in (np.zeros(3,int), np.full(3,3)):
            np.testing.assert_array_equal(result, selector.select(self.k, self.table, 0., warm_start=warm))

    def test_local_is_monotonic_and_bounded(self):
        selector = LocalTiltSelector(max_rounds=3)
        warm = np.array([0, 1, 2])
        result = selector.select(self.k, self.table, 0., warm_start=warm)
        self.assertGreaterEqual(self.coverage(result), self.coverage(warm))
        self.assertTrue(np.all(np.diff(selector.last_coverage_trace) >= -1e-12))
        self.assertLessEqual(selector.last_num_rounds, 3)
        self.assertTrue(np.all((result >= 0) & (result < 4)))

    def test_dynamic_retains_or_improves_previous_assignment(self):
        controller = DynamicTiltController(LocalTiltSelector(max_rounds=3))
        previous = controller.update(self.k, self.table, 0.).copy()
        self.table = self.table[::-1].copy()
        baseline = self.coverage(previous)
        current = controller.update(self.k, self.table, 0.)
        self.assertGreaterEqual(self.coverage(current), baseline)

    def test_adaptive_clips_and_resets_missing_data(self):
        controller = AdaptiveLegacyTiltController(3, 4., 0., 6., 500., 0., 3., 95., 2.)
        for _ in range(10):
            out = controller.update(np.array([1., 0., 1.]), np.array([0., 1., 0.]),
                                    has_data=np.array([True, True, False]))
            self.assertTrue(np.all((out >= 0.) & (out <= 6.)))
            self.assertEqual(out[2], 0.)

    def test_rl_unscheduled_state_and_evaluation_never_observes(self):
        policy = Mock()
        policy.act.return_value = np.ones(3, dtype=int)
        controller = RLTiltController(policy, 3)
        before = controller.tilt_idx.copy()
        # The very first update() call unconditionally initializes
        # _prev_observations from None to zeros, regardless of schedule --
        # only from the SECOND call onward does an unscheduled row's stored
        # state stay truly untouched (checked below).
        controller.update(np.zeros((3, 5)), np.zeros(3), training=False,
                          schedule=np.zeros(3, bool))
        np.testing.assert_array_equal(controller.tilt_idx, before)
        np.testing.assert_array_equal(controller._prev_observations, np.zeros((3, 5)))
        policy.observe.assert_not_called()

        previous = controller._prev_observations.copy()
        controller.update(np.ones((3, 5)), np.zeros(3), training=False,
                          schedule=np.zeros(3, bool))
        np.testing.assert_array_equal(controller.tilt_idx, before)
        np.testing.assert_array_equal(controller._prev_observations, previous)
        policy.observe.assert_not_called()

    def test_rl_initialized_unscheduled_rows_and_no_learning(self):
        policy = Mock()
        policy.act.return_value = np.ones(3, dtype=int)
        controller = RLTiltController(policy, 3)
        observations = np.arange(15, dtype=float).reshape(3, 5)
        # Establish a previous observation through the public method first.
        controller.update(observations, np.zeros(3), training=False,
                          schedule=np.zeros(3, bool))
        previous = controller._prev_observations.copy()
        tilts = controller.tilt_idx.copy()
        controller.update(observations + 10., np.ones(3), training=False,
                          schedule=np.zeros(3, bool))
        np.testing.assert_array_equal(controller._prev_observations, previous)
        np.testing.assert_array_equal(controller.tilt_idx, tilts)
        policy.observe.assert_not_called()

if __name__ == '__main__':
    unittest.main()
