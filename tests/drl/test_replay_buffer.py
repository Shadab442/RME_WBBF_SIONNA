"""ReplayBuffer contract tests, especially WESN sequences.

Run: python tests/drl/test_replay_buffer.py
Tests call the real implementation. No production files are modified.
Positive batches sample without replacement; WESN uniqueness is by start index,
not by non-overlapping windows. recent(0) is expected to return zero observations.

Verification coverage:
- add(): field values/shapes/dtypes, copied storage, length, and oldest eviction.
- recent(): observation-only chronological output, requested/available history,
  zero-history behavior, and independence of returned arrays from storage.
- sample(): aligned fields and dtypes, valid distinct selections, rejection of
  insufficient data, reproducible dedicated randomness, and no storage mutation.
- WESN: [batch, time, features] observations, [batch, time] scalar fields,
  chronological contiguous windows, first/last valid starts, overlapping windows,
  minimum N = batch_size + sequence_length - 1, and behavior after eviction.
This verifies stored sequence ordering, not continuity across real episode resets
or missing-data gaps. The recent(0) assertion intentionally exposes that edge case
if the implementation returns the entire buffer.
"""
import logging
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from drl.replay_buffer import ReplayBuffer


def transition(i):
    # Every field encodes the ID, so misaligned samples cannot pass silently.
    return (np.array([i, i + .25], dtype=np.float32),
            np.int64(i + 10), np.float32(i + 100),
            np.array([i + 1, i + 1.25], dtype=np.float32), np.float32(i % 2))


def populated(n=7, capacity=20, sequence_length=None, seed=12):
    buffer = ReplayBuffer(capacity, random.Random(seed), sequence_length)
    for i in range(n):
        buffer.add(*transition(i))
    return buffer


class AddAndRecentTests(unittest.TestCase):
    def test_add_preserves_fields_shapes_and_dtypes(self):
        # Verify adding one transition preserves all five fields, their shapes, and numerical dtypes.
        buffer = populated(1)
        self.assertEqual(len(buffer), 1)
        actual = buffer.sample(1)
        # Sampling adds a batch axis but must preserve each stored field exactly.
        for field, expected in zip(actual, transition(0)):
            np.testing.assert_array_equal(field, np.expand_dims(expected, 0))
            self.assertEqual(field.dtype, np.asarray(expected).dtype)
        self.assertEqual(len(actual), 5)

    def test_add_owns_copies_of_all_fields(self):
        # Mutate every caller-owned input after add(); stored values must retain their original contents.
        buffer = populated(0)
        inputs = [np.array(x, copy=True) for x in transition(4)]
        expected = [x.copy() for x in inputs]
        buffer.add(*inputs)
        for x in inputs:
            x[...] = -999
        for actual, original in zip(buffer.sample(1), expected):
            np.testing.assert_array_equal(actual[0], original)

    def test_capacity_evicts_oldest_and_keeps_order(self):
        # Overflow capacity and verify only the three newest transitions remain in chronological order.
        buffer = populated(8, capacity=3)
        self.assertEqual(len(buffer), 3)
        np.testing.assert_array_equal(buffer.recent(3), [transition(i)[0] for i in (5,6,7)])

    def test_recent_returns_only_observations_oldest_first(self):
        # Verify recent(n) selects the latest n observations and does not consume transitions.
        buffer = populated()
        for n in (1,3,7):
            with self.subTest(n=n):
                np.testing.assert_array_equal(buffer.recent(n),
                                              [transition(i)[0] for i in range(7-n,7)])
        self.assertEqual(len(buffer), 7)

    def test_recent_more_than_available_returns_available_history(self):
        # Verify an oversized history request returns all available observations without padding.
        np.testing.assert_array_equal(populated(2).recent(5),
                                      [transition(i)[0] for i in range(2)])

    def test_recent_zero_returns_no_observations(self):
        # Verify the zero-history boundary returns an empty array with the observation feature dimension.
        # Important for DQN sequence_length=1, which requests T-1 history.
        result = populated(3).recent(0)
        self.assertEqual(result.shape, (0,2), 'recent(0) must not return the whole buffer')

    def test_recent_result_does_not_alias_storage(self):
        # Mutate the returned history and verify subsequent reads still return the original observations.
        buffer = populated(3)
        recent = buffer.recent(2)
        recent[:] = -999
        np.testing.assert_array_equal(buffer.recent(2), [transition(i)[0] for i in (1,2)])


class SampleTests(unittest.TestCase):
    def assert_fields(self, sample, ids):
        # Verify each sampled field encodes the same transition IDs, with correct feature axes and dtypes.
        self.assertEqual(len(sample), 5)
        obs, actions, rewards, successors, dones = sample
        np.testing.assert_array_equal(obs, np.stack([ids, ids+.25], axis=-1))
        np.testing.assert_array_equal(actions, ids+10)
        np.testing.assert_array_equal(rewards, ids+100)
        np.testing.assert_array_equal(successors, np.stack([ids+1, ids+1.25], axis=-1))
        np.testing.assert_array_equal(dones, ids % 2)
        for actual, dtype in zip(sample, (np.float32,np.int64,np.float32,np.float32,np.float32)):
            self.assertEqual(actual.dtype, dtype)

    def test_mlp_samples_aligned_distinct_transitions(self):
        # Verify ordinary batches contain the requested number of unique valid transitions.
        for batch_size in (1,4,7):
            with self.subTest(batch_size=batch_size):
                sample = populated().sample(batch_size)
                ids = sample[0][:,0]
                self.assertEqual(ids.shape, (batch_size,))
                self.assertEqual(len(set(ids)), batch_size)
                self.assertTrue(set(ids).issubset(set(range(7))))
                self.assert_fields(sample, ids)

    def test_wesn_controlled_starts_preserve_batch_and_time_axes(self):
        # Force nonchronological batch starts; each individual window must still be chronological and correctly shaped.
        buffer = populated(sequence_length=3)
        buffer.rng = Mock()
        buffer.rng.sample.return_value = [4,0,2]
        sample = buffer.sample(3)
        # Seven transitions admit exactly five length-three starts: indices 0 through 4.
        buffer.rng.sample.assert_called_once_with(range(5), 3)
        # Explicit independent expected windows; no production slicing reused.
        ids = np.array([[4,5,6],[0,1,2],[2,3,4]])
        self.assert_fields(sample, ids)
        self.assertEqual(sample[0].shape, (3,3,2))
        for field in (sample[1],sample[2],sample[4]):
            self.assertEqual(field.shape, (3,3))

    def test_wesn_all_valid_windows_including_first_and_last(self):
        # Sample every possible start to catch omitted edge windows and duplicated starts.
        sample = populated(sequence_length=3).sample(5)
        ids = sample[0][...,0]
        self.assertEqual(set(ids[:,0]), {0,1,2,3,4})
        np.testing.assert_array_equal(np.diff(ids,axis=1), np.ones((5,2)))
        self.assert_fields(sample, ids)
        # Overlap is allowed; distinct starts do not mean independent windows.
        self.assertLess(len(np.unique(ids)), ids.size)

    def test_wesn_minimum_history_for_distinct_batch(self):
        # Verify the failure/success boundary for sampling B distinct length-T windows.
        # B=2, T=3 requires N >= B+T-1 = 4, not merely N >= T.
        with self.assertRaises(ValueError):
            populated(3, sequence_length=3).sample(2)
        sample = populated(4, sequence_length=3).sample(2)
        self.assertEqual(set(sample[0][:,0,0]), {0,1})

    def test_wesn_one_window_and_sequence_length_one(self):
        # Verify singleton batch/time dimensions are retained rather than squeezed away.
        for n, length in ((3,3),(4,1)):
            with self.subTest(n=n, length=length):
                sample = populated(n, sequence_length=length).sample(1)
                self.assertEqual(sample[0].shape, (1,length,2))
                self.assert_fields(sample, sample[0][...,0])

    def test_wesn_windows_after_capacity_eviction(self):
        # Verify window indices refer to retained history after eviction, never to discarded transitions.
        sample = populated(10, capacity=5, sequence_length=3).sample(3)
        ids = sample[0][...,0]
        self.assertEqual(set(ids[:,0]), {5,6,7})
        np.testing.assert_array_equal(np.diff(ids,axis=1), np.ones((3,2)))
        self.assert_fields(sample, ids)

    def test_insufficient_history_and_oversized_batches_rejected(self):
        # Verify empty/short history and requests beyond the available distinct samples raise ValueError.
        for n, length, batch in ((0,None,1),(2,None,3),(0,3,1),(2,3,1),(5,3,4)):
            with self.subTest(n=n, sequence_length=length, batch=batch):
                with self.assertRaises(ValueError):
                    populated(n, sequence_length=length).sample(batch)

    def test_sample_is_non_destructive_and_returns_independent_arrays(self):
        # Verify sampling does not remove entries and modifying sampled arrays does not corrupt storage.
        for length in (None,3):
            with self.subTest(sequence_length=length):
                buffer = populated(sequence_length=length)
                before = buffer.recent(7).copy()
                sample = buffer.sample(2)
                for field in sample:
                    field[...] = -999
                self.assertEqual(len(buffer), 7)
                np.testing.assert_array_equal(buffer.recent(7), before)
                again = buffer.sample(2)
                self.assert_fields(again, again[0][...,0])

    def test_dedicated_rng_reproducibility_and_global_rng_isolation(self):
        # Verify equal dedicated seeds reproduce batches, without consuming or depending on global randomness.
        saved = random.getstate()
        try:
            for length in (None,3):
                a = populated(sequence_length=length)
                b = populated(sequence_length=length)
                for _ in range(3):
                    global_before = random.getstate()
                    first = a.sample(2)
                    self.assertEqual(random.getstate(), global_before)
                    random.random()  # Perturb global RNG; dedicated RNG must be unaffected.
                    second = b.sample(2)
                    for x,y in zip(first,second):
                        np.testing.assert_array_equal(x,y)
        finally:
            random.setstate(saved)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
