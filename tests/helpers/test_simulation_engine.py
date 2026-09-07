"""Small end-to-end contract checks using the public example configuration."""
import copy
import unittest
from pathlib import Path
from support import np, numpy
import yaml
from helpers.simulation_engine import SimulationEngine

class EngineContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = yaml.safe_load((Path(__file__).resolve().parents[2]/'config.yaml').read_text())
        cfg['topology'].update(num_ut=6)
        cfg['mobility'].update(num_groups=3, cluster_mobility_mode='random', waypoint_type='random')
        cfg['simulation'].update(measurement_interval_s=1., tilt_control_interval_s=2.,
                                 num_tilt_control_intervals=2, steady_state_episodes=1,
                                 num_realizations_per_slot=1, max_realization_cuda=1)
        cfg['antenna']['downtilt_sweep_deg'] = [0., 4.]
        # sync: this test's 2-slot window is far too short for the real
        # 21-sector topology's async coloring (3 phases) to fit without
        # colliding adjacent sectors onto the same closing slot -- not
        # exercising async staggering here anyway.
        # region_block_size=1: spatial_grid_cell_size_m=100. gives grid_n=3
        # here, which config.yaml's own default region_block_size=2 doesn't
        # evenly divide. coverage_fill='zero': this test's 6 UEs are too
        # sparse over 21 sectors for config.yaml's own default "kriged" to
        # ever produce a completed snapshot for every sector -- these tests
        # exercise the training pipeline, not Kriging readiness (see
        # KrigedAsyncSafetyContract for that).
        cfg['algorithms']['drl'].update(policy_name='random', spatial_grid_cell_size_m=100.,
                                        region_block_size=1, coverage_fill='zero', async_schedule='sync')
        cfg['algorithms']['optimization']['run_causal'] = False
        cls.cfg = cfg

    def test_last_step_preserves_positions_and_fresh_channel_draws(self):
        engine = SimulationEngine(copy.deepcopy(self.cfg))
        self.assertEqual(engine.num_ut, 6)
        self.assertEqual(engine.num_bs, 21)
        self.assertEqual(engine.measurement_slots_per_interval, 2)
        before = [numpy(m.ut_loc).copy() for m in engine.mobility_list]
        first = engine.step_environment(is_last_step=True)
        second = engine.step_environment(is_last_step=True)
        self.assertEqual(len(first), 1)
        for m, pos in zip(engine.mobility_list, before):
            np.testing.assert_array_equal(numpy(m.ut_loc), pos)
        for a,b in zip(first,second):
            loss = numpy(a.total_pathloss_db)
            self.assertEqual(loss.shape, (1,21,6))
            self.assertTrue(np.isfinite(loss).all() and (loss > 0).all())
            self.assertFalse(np.array_equal(loss, numpy(b.total_pathloss_db)))
            theta = numpy(a.theta_lcs)
            self.assertEqual(theta.shape, loss.shape)
            self.assertTrue(np.all((theta >= 0) & (theta <= np.pi)))
            self.assertEqual(numpy(a.phi_lcs).shape, loss.shape)

    def test_short_simulation_outputs(self):
        engine = SimulationEngine(copy.deepcopy(self.cfg))
        result = engine.run_simulation()
        for name in ('oracle', 'adaptive_legacy', 'no_tilt', 'drl'):
            coverage = np.asarray(result['coverage_dynamic_local_'+name] if name == 'oracle'
                                  else result['coverage_'+name])
            self.assertEqual(coverage.shape, (2,))
            self.assertTrue(np.all((coverage >= 0) & (coverage <= 1)))
        self.assertTrue(np.isnan(result['coverage_dynamic_local_causal']).all())
        for key in ('drl_reward_history','drl_loss_per_interval','drl_policy'):
            self.assertIn(key, result)

if __name__ == '__main__':
    unittest.main()
