"""Deterministic Dqn checks. Run directly; no simulation or pytest required.

Uses controlled networks/RNGs to distinguish branches, and real replay buffers
for sampling and training-boundary checks. Does not modify production files.

Verification coverage:
- act(): masked defaults, forced exploration/exploitation, epsilon boundary and
  decay, evaluation, WESN history availability/order, and network input tensors.
- _learn(): actual replay-sampling call, float32 observation conversion, final
  WESN timestep selection, Double DQN targets, terminal masking, and mean loss.
- Learning updates: independently calculated gradients/SGD changes, frozen target
  parameters, minimum replay size, gradient-step count, and target-sync timing.
Known Q-values come from test networks; the real Dqn methods perform the work.
The loss spy records arguments while still executing the real PyTorch loss.
These checks do not establish convergence or verify episode/gap continuity and
ongoing WESN evaluation-history advancement.
"""
import logging
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from drl.dqn import Dqn
from drl.replay_buffer import ReplayBuffer


def policy(model='mlp', **overrides):
    options = dict(num_sectors=1, num_features=2, num_actions=3, model=model,
                   sequence_length=3, batch_size=2, hidden_sizes=(4,),
                   wesn={'units':4, 'win_len':2}, device='cpu',
                   epsilon_start=.5, epsilon_end=.1, target_update_steps=100)
    options.update(overrides)
    return Dqn(**options)


class RecordingNetwork(nn.Module):
    """Trainable lookup table; input's final feature-vector first value is row ID."""
    def __init__(self, rows):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(rows, dtype=torch.float32))
        self.calls = []

    def forward(self, inputs):
        self.calls.append((inputs.detach().clone(), torch.is_grad_enabled()))
        ids = inputs[:, -1, 0] if inputs.ndim == 3 else inputs[:, 0]
        return self.values[ids.long()]


def install(p, rows, target_rows=None):
    online = RecordingNetwork(rows)
    target = RecordingNetwork(target_rows if target_rows is not None else rows)
    p.networks[0], p.targets[0] = online, target
    p.optimizers[0] = torch.optim.SGD(online.parameters(), lr=.1)
    return online, target


def add_history(p, count):
    for i in range(count):
        p.buffers[0].add(np.array([i, 10+i]), 0, 0., np.array([i+1, 11+i]), 0.)


class ActVerification(unittest.TestCase):
    def test_mask_defaults_and_no_branch_execution(self):
        # Verify masked sectors return defaults (or zero) without evaluating a network or drawing randomness.
        p = policy(num_sectors=2)
        p.networks = [Mock(), Mock()]
        p.rng = Mock()
        obs = np.zeros((2,2))
        defaults = np.array([2,1])
        for default, expected in ((None,[0,0]), (defaults,[2,1])):
            result = p.act(obs, True, mask=[False,False], default_action=default)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(result.dtype, np.int64)
        np.testing.assert_array_equal(defaults, [2,1])
        p.rng.random.assert_not_called()
        p.rng.randrange.assert_not_called()
        for net in p.networks:
            net.assert_not_called()

    def test_exploration_precedes_network_and_history_checks(self):
        # Force draw < epsilon and verify the random action is used even with empty WESN history.
        for model in ('mlp','wesn'):
            with self.subTest(model=model):
                p = policy(model)
                p.networks[0] = Mock()
                p.rng = Mock(random=Mock(return_value=.49), randrange=Mock(return_value=2))
                np.testing.assert_array_equal(p.act(np.zeros((1,2)), True), [2])
                p.rng.randrange.assert_called_once_with(3)
                p.networks[0].assert_not_called()

    def test_exploitation_epsilon_boundary_and_evaluation(self):
        # Verify draw >= epsilon selects argmax; evaluation bypasses exploration entirely.
        for training, draw in ((True,.5),(True,.9),(False,0.)):
            with self.subTest(training=training, draw=draw):
                p = policy()
                net, _ = install(p, [[2,7,3]])
                p.rng = Mock(random=Mock(return_value=draw))
                obs = np.array([[0.,12.]], dtype=np.float64)
                np.testing.assert_array_equal(p.act(obs, training), [1])
                state, grad = net.calls[0]
                torch.testing.assert_close(state, torch.tensor([[0.,12.]]))
                self.assertEqual(state.device.type, 'cpu')
                self.assertFalse(grad)
                p.rng.randrange.assert_not_called()
                if not training:
                    p.rng.random.assert_not_called()
                np.testing.assert_array_equal(obs, [[0.,12.]])

    def test_wesn_insufficient_history_in_training_and_evaluation(self):
        # With T=3, zero or one past observation must trigger fallback without a network call.
        for training in (True,False):
            for count in (0,1):
                with self.subTest(training=training, count=count):
                    p = policy('wesn')
                    add_history(p, count)
                    p.networks[0] = Mock()
                    p.rng = Mock(random=Mock(return_value=.9), randrange=Mock(return_value=2))
                    np.testing.assert_array_equal(p.act(np.array([[2.,12.]]), training), [2])
                    p.rng.randrange.assert_called_once_with(3)
                    p.networks[0].assert_not_called()

    def test_wesn_exact_and_excess_history_order(self):
        # Verify only the latest T-1 observations precede the current observation in the network input.
        for count in (2,4):
            with self.subTest(count=count):
                p = policy('wesn')
                add_history(p, count)
                net, _ = install(p, [[2,7,3]]*5)
                result = p.act(np.array([[count,10+count]]), False)
                np.testing.assert_array_equal(result, [1])
                expected = torch.tensor([[[i,10+i] for i in range(count-2,count+1)]], dtype=torch.float32)
                torch.testing.assert_close(net.calls[0][0], expected)
                self.assertFalse(net.calls[0][1])

    def test_mixed_mask_and_variable_feature_widths(self):
        # Verify masked and active sectors are handled separately, preserving each input width.
        p = policy(num_sectors=2, num_features=[2,3])
        p.networks = [Mock(), RecordingNetwork([[2,7,3]])]
        result = p.act([np.zeros(2), np.array([0.,4.,5.])], False,
                       mask=[False,True], default_action=[2,0])
        np.testing.assert_array_equal(result, [2,1])
        p.networks[0].assert_not_called()
        self.assertEqual(p.networks[1].calls[0][0].shape, (1,3))

    def test_epsilon_decay_and_floor(self):
        # Compare exploration probability with hand-calculated decay values and its lower bound.
        p = policy(epsilon_start=.8, epsilon_end=.1, epsilon_decay_rate=.5)
        for steps, expected in ((0,.8),(1,.4),(2,.2),(3,.1),(20,.1)):
            p.sector_decision_intervals[0] = steps
            self.assertAlmostEqual(p._epsilon(0), expected)


class LearnVerification(unittest.TestCase):
    def test_tensor_processing_double_dqn_loss_and_gradients(self):
        # Verify real _learn() processing against known Q-values, targets, loss, and exact gradients.
        for model in ('mlp','wesn'):
            with self.subTest(model=model):
                p = policy(model, num_actions=2)
                online, target = install(p, [[2,9],[8,.5],[1,4],[5,2]],
                                         [[0,0],[0,0],[10,3],[6,9]])
                obs = np.array([[0,11],[1,12]], dtype=np.float64)
                nxt = np.array([[2,13],[3,14]], dtype=np.float64)
                actions, rewards, dones = np.array([0,1]), np.ones(2), np.array([0,1])
                if model == 'wesn':
                    obs = np.stack([obs+np.array([0,100]),obs+np.array([0,200]),obs], axis=1)
                    nxt = np.stack([nxt+np.array([0,100]),nxt+np.array([0,200]),nxt], axis=1)
                    actions = np.stack([1-actions,1-actions,actions],axis=1)
                    rewards = np.array([[100,200,1],[300,400,1]], dtype=np.float64)
                    dones = np.array([[1,1,0],[0,0,1]])
                p.buffers[0] = Mock(sample=Mock(return_value=(obs,actions,rewards,nxt,dones)))
                before = online.values.detach().clone()
                target_before = target.values.detach().clone()
                original_loss = torch.nn.functional.smooth_l1_loss
                with patch('torch.nn.functional.smooth_l1_loss', wraps=original_loss) as loss_spy:
                    p._learn(0)
                # Verify replay is sampled once and both networks receive the complete float32 inputs.
                p.buffers[0].sample.assert_called_once_with(2)
                torch.testing.assert_close(online.calls[0][0], torch.tensor(obs,dtype=torch.float32))
                torch.testing.assert_close(online.calls[1][0], torch.tensor(nxt,dtype=torch.float32))
                torch.testing.assert_close(target.calls[0][0], torch.tensor(nxt,dtype=torch.float32))
                self.assertEqual([c[1] for c in online.calls], [True,False])
                self.assertFalse(target.calls[0][1])
                # Capture the real loss inputs: online next-state argmax selects target value 3,
                # so the nonterminal target is 1 + 0.9*3 = 3.7; the terminal target is 1.
                chosen, bellman = loss_spy.call_args.args
                torch.testing.assert_close(chosen.detach(), torch.tensor([2.,.5]))
                torch.testing.assert_close(bellman, torch.tensor([3.7,1.]))
                self.assertFalse(bellman.requires_grad)
                self.assertEqual(chosen.device.type, 'cpu')
                self.assertEqual(chosen.dtype, torch.float32)
                # Mean Smooth L1: errors 1.7 and 0.5 give (1.2 + 0.125)/2 = 0.6625.
                self.assertAlmostEqual(p.step_losses[-1], .6625, places=6)
                # Mean Huber gradients: -1/2 and -0.5/2 on selected entries.
                expected_grad = torch.zeros_like(before)
                expected_grad[0,0], expected_grad[1,1] = -.5, -.25
                torch.testing.assert_close(online.values.grad, expected_grad)
                torch.testing.assert_close(online.values, before-.1*expected_grad)
                # The target branch must remain detached and unchanged by the optimizer.
                self.assertIsNone(target.values.grad)
                torch.testing.assert_close(target.values, target_before)

    def test_loss_zero_signed_boundary_errors_and_gamma_zero(self):
        # Verify both Smooth L1 branches and that gamma=0 removes all next-state contributions.
        p = policy(batch_size=5, gamma=0., num_actions=2)
        install(p, [[0,0]]*5, [[50,80]]*5)
        obs = np.column_stack([np.arange(5),np.zeros(5)])
        # Errors 0, +0.5, -0.5, +1, -2; mean (0+.125+.125+.5+1.5)/5.
        p.buffers[0] = Mock(sample=Mock(return_value=(obs,np.zeros(5,int),
                              np.array([0,.5,-.5,1,-2]),obs,np.zeros(5))))
        p._learn(0)
        self.assertAlmostEqual(p.step_losses[-1], .45, places=6)


class ReplayAndScheduleVerification(unittest.TestCase):
    def test_real_replay_alignment_order_and_distinct_windows(self):
        # Verify real replay preserves field alignment, evicts old entries, and samples distinct starts.
        for length in (None,3):
            with self.subTest(sequence_length=length):
                buffer = ReplayBuffer(5, random.Random(4), sequence_length=length)
                for i in range(7):
                    obs = np.array([i],float)
                    buffer.add(obs,i,100+i,obs+1,0.)
                    obs[:] = -999  # Storage must own a copy.
                batch = 5 if length is None else 3
                obs, actions, rewards, nxt, _ = buffer.sample(batch)
                ids = obs[...,0]
                np.testing.assert_array_equal(actions,ids)
                np.testing.assert_array_equal(rewards,ids+100)
                np.testing.assert_array_equal(nxt[...,0],ids+1)
                self.assertGreaterEqual(ids.min(),2)
                if length is None:
                    self.assertEqual(set(ids.tolist()),set(range(2,7)))
                else:
                    np.testing.assert_array_equal(np.diff(ids,axis=1),np.ones((3,2)))
                    self.assertEqual(set(ids[:,0].tolist()),{2,3,4})

    def test_training_threshold_and_number_of_gradient_steps(self):
        # Verify no early training, then exactly two real learning steps at the minimum replay size.
        for model, threshold in (('mlp',2),('wesn',4)):
            with self.subTest(model=model):
                p = policy(model, train_steps_per_interval=2)
                actual_learn = p._learn
                with patch.object(p,'_learn', wraps=actual_learn) as learn:
                    for i in range(threshold-1):
                        p.observe(np.array([[i,0.]]),[0],[1.],np.array([[i+1,0.]]),False)
                    # Before the threshold there are too few distinct samples/windows to learn.
                    learn.assert_not_called()
                    p.observe(np.array([[threshold-1,0.]]),[0],[1.],np.array([[threshold,0.]]),False)
                    self.assertEqual(learn.call_count,2)
                    self.assertEqual(len(p.step_losses),2)
                    self.assertTrue(np.isfinite(p.step_losses).all())

    def test_target_sync_counts_only_valid_transitions_and_follows_learning(self):
        # Verify target copying occurs after learning on each second valid transition, never on a masked call.
        p = policy(batch_size=1, target_update_steps=2)
        online, target = install(p, [[0,0,0]])
        obs = np.zeros((1,2))
        target_before = target.values.detach().clone()
        p.observe(obs,[0],[1.],obs,False)
        torch.testing.assert_close(target.values,target_before)
        self.assertFalse(torch.equal(online.values,target.values))
        p.observe(obs,[0],[1.],obs,False,mask=[False])
        self.assertEqual(p.sector_decision_intervals[0],1)
        self.assertEqual(len(p.buffers[0]),1)
        torch.testing.assert_close(target.values,target_before)
        p.observe(obs,[0],[1.],obs,False)
        self.assertEqual(p.sector_decision_intervals[0],2)
        torch.testing.assert_close(target.values,online.values)
        self.assertIsNone(target.values.grad)
        synced = target.values.detach().clone()
        p.observe(obs,[0],[1.],obs,False)
        torch.testing.assert_close(target.values,synced)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
