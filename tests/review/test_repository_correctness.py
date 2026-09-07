"""Targeted repository review regressions; failures identify unresolved contracts.

Run directly with the project's sionna-wbbf interpreter. All artifacts use /tmp;
production files and historical experiment outputs are not modified.
"""
import ast
import copy
import logging
import os
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/beamforming-review-mpl')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests/helpers'))
import numpy as np
import torch
from algorithm.kriging import OrdinaryKriging
from algorithm.wesn import EchoStateReservoir
from drl.replay_buffer import ReplayBuffer
from drl.dqn import Dqn
from helpers.mobility import ReferencePointGroupMobility
from helpers.tilt_controller import RLTiltController
from support import topology
from test_spatial_grid_estimator import estimator
import test_simulation_engine as engine_contract
from helpers.simulation_engine import SimulationEngine


def wesn_policy(**kwargs):
    options = dict(model='wesn', wesn={'units':4, 'win_len':2}, sequence_length=3,
                   batch_size=64, epsilon_start=0., epsilon_end=0.)
    options.update(kwargs)
    return Dqn(1, 1, 2, **options)


def function_from_script(relative_path, name):
    """Extract a pure function without triggering a script's top-level I/O."""
    path = ROOT/relative_path
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {'np':np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class DrlCorrectness(unittest.TestCase):
    def test_wesn_evaluation_eventually_uses_network_without_learning(self):
        policy = wesn_policy()
        calls = []
        handle = policy.networks[0].register_forward_pre_hook(lambda module,args: calls.append(args[0].clone()))
        controller = RLTiltController(policy, 1)
        for value in range(10):
            controller.update(np.array([[float(value)]]), np.ones(1), training=False)
        handle.remove()
        self.assertGreater(len(calls), 0, 'Evaluation never invoked the trained network after 10 real observations')
        self.assertEqual(len(policy.buffers[0]), 0, 'Evaluation must not add training transitions')

    def test_wesn_evaluation_window_advances(self):
        policy = wesn_policy()
        for value in (0.,1.):
            policy.buffers[0].add([value],0,1.,[value+1.],False)
        calls = []
        handle = policy.networks[0].register_forward_pre_hook(lambda module,args: calls.append(args[0].clone()))
        policy.act(np.array([[2.]]), training=False)
        policy.act(np.array([[3.]]), training=False)
        handle.remove()
        np.testing.assert_array_equal(calls[-1].numpy().reshape(-1), [1.,2.,3.])

    def test_recurrent_replay_does_not_cross_terminal(self):
        replay = ReplayBuffer(10, random.Random(0), sequence_length=3)
        replay.add([0.],0,1.,[1.],True)
        replay.add([100.],0,1.,[101.],False)
        replay.add([101.],0,1.,[102.],False)
        try:
            sample = replay.sample(1)
        except ValueError:  # No valid complete window yet is an acceptable outcome.
            return
        self.assertFalse(sample[-1][0,:-1].any(), 'An earlier terminal lies inside the sampled recurrent context')

    def test_recurrent_replay_does_not_bridge_missing_observations(self):
        policy = wesn_policy()
        controller = RLTiltController(policy,1)
        for value, valid in enumerate([True,True,False,True,True,True]):
            controller.update(np.array([[float(value)]]),np.ones(1),training=True,
                              has_data=np.array([valid]))
        try:
            obs, _, _, next_obs, _ = policy.buffers[0].sample(1)
        except ValueError:
            return
        np.testing.assert_array_equal(next_obs[:,:-1], obs[:,1:],
                                      err_msg='Dropped transitions were joined into an artificial contiguous history')

    def test_double_dqn_targets_and_terminal_mask(self):
        for terminal, expected_loss in ((False,.5),(True,0.)):
            with self.subTest(terminal=terminal):
                p=Dqn(1,1,2,model='mlp',hidden_sizes=[],batch_size=1,gamma=.5,learning_rate=0.)
                with torch.no_grad():
                    p.networks[0].layers[0].weight.zero_()
                    p.networks[0].layers[0].bias.copy_(torch.tensor([1.,3.]))
                    p.targets[0].layers[0].weight.zero_()
                    p.targets[0].layers[0].bias.copy_(torch.tensor([10.,2.]))
                p.buffers[0].add([0.],0,1.,[1.],terminal)
                p._learn(0)
                self.assertAlmostEqual(p.step_losses[-1],expected_loss,places=6)
                self.assertTrue(all(parameter.grad is None for parameter in p.targets[0].parameters()))

    def test_target_sync_cadence(self):
        p=Dqn(1,1,2,model='mlp',hidden_sizes=[],batch_size=1,target_update_steps=2)
        before=copy.deepcopy(p.targets[0].state_dict())
        for step in range(2):
            p.observe(np.ones((1,1)),np.array([0]),np.array([5.]),np.zeros((1,1)),False)
            if step==0:
                for key,val in before.items():
                    torch.testing.assert_close(p.targets[0].state_dict()[key],val)
        for key,val in p.networks[0].state_dict().items():
            torch.testing.assert_close(p.targets[0].state_dict()[key],val)

    def test_mlp_minimal_training_and_mask(self):
        p=Dqn(2,2,2,model='mlp',batch_size=1)
        p.observe(np.zeros((2,2)),np.zeros(2,int),np.ones(2),np.ones((2,2)),False,mask=np.array([True,False]))
        self.assertEqual([len(b) for b in p.buffers],[1,0])
        self.assertEqual(len(p.step_losses),1)
        self.assertTrue(np.isfinite(p.step_losses).all())


class NumericAndIntegrationCorrectness(unittest.TestCase):
    def test_reservoir_recurrence_and_frozen_parameters(self):
        torch.manual_seed(4)
        r=EchoStateReservoir(2,5,connectivity=1.,leaky=.3,spectral_radius=.8)
        inputs=torch.arange(12,dtype=torch.float32).reshape(2,3,2)/10
        state=np.zeros((2,5),dtype=np.float32)
        for t in range(3):
            state=.7*state+.3*np.tanh(inputs[:,t].numpy()@r.kernel.numpy()+state@r.recurrent_kernel.numpy().T+r.bias.numpy())
        np.testing.assert_allclose(r(inputs).numpy(),state,atol=1e-6)
        self.assertEqual(list(r.parameters()),[])
        self.assertAlmostEqual(float(torch.linalg.eigvals(r.recurrent_kernel).abs().max()),.8,places=5)

    def test_kriging_constants_and_known_points(self):
        k=OrdinaryKriging(1.,nugget=0.)
        xy=np.array([[0.,0.],[1.,0.],[0.,1.]])
        np.testing.assert_allclose(k.predict(xy,np.array([2.,4.,8.]),xy),[2.,4.,8.],atol=1e-7)
        np.testing.assert_allclose(k.predict(xy,np.full(3,7.),np.array([[10.,10.],[.2,.2]])),[7.,7.],atol=1e-7)

    def test_async_occupancy_startup_and_steady_state(self):
        e=estimator(occupancy_ema_alpha=1.)
        for slot in range(6):
            e.accumulate(np.array([0,1]),np.array([0,0]),np.ones(2,bool),sinr_db=np.ones(2))
            if slot>=1:
                closes=np.array([slot%2==1,slot%2==0])
                occupancy=e.compute(closes)[0]
                np.testing.assert_allclose(occupancy[closes].sum(axis=1),.5)

    def test_static_cluster_preserves_radius_at_boundary(self):
        t=topology(num_sites=1)
        m=ReferencePointGroupMobility(torch.tensor([[140.,0.,1.5]]),torch.tensor([0]),[(100.,0.)],40.,t,1.,2.,
            cluster_mobility_mode='periodic',num_waypoints=2,waypoint_hold_steps=1,
            waypoints=[[[100.,0.],[260.,0.]]],intra_cluster_mobility='static')
        m.step(1.)
        offsets=m.ut_loc[:,:2]-m.ref_xy[m.member_group_idx]
        self.assertTrue(torch.all(torch.linalg.norm(offsets,dim=1)<=40.+1e-5),
                        f'Member left its 40 m cluster disk: offset={offsets.tolist()}')

    def test_small_mlp_and_wesn_engine_runs_train(self):
        engine_contract.EngineContract.setUpClass()
        for model in ('mlp','wesn'):
            with self.subTest(model=model):
                cfg=copy.deepcopy(engine_contract.EngineContract.cfg)
                cfg['simulation']['num_tilt_control_intervals']=8
                cfg['algorithms']['drl']['policy_name']='dqn'
                cfg['algorithms']['drl']['dqn'].update(model=model,batch_size=2,
                    sequence_length=2,train_steps_per_interval=1,target_update_steps=2,
                    wesn={'units':4,'win_len':2})
                engine=SimulationEngine(cfg)
                result=engine.run_simulation()
                self.assertGreater(len(engine.drl_policy.step_losses),0)
                self.assertTrue(np.isfinite(engine.drl_policy.step_losses).all())
                self.assertTrue(np.all((result['coverage_drl']>=0)&(result['coverage_drl']<=1)))

    def test_pooled_sinr_cdf_flattens_realizations(self):
        ecdf=function_from_script('scripts/plots/plot_static_scenario_tilts_effect.py','ecdf')
        x,y=ecdf(np.array([[3.,1.,2.],[6.,4.,5.]]))
        np.testing.assert_array_equal(x,np.arange(1.,7.))
        np.testing.assert_allclose(y,np.arange(1.,7.)/6.)


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
