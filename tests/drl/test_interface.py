"""Observation-focused verification of the actual drl.interface functions.

Coverage:
- Variable observations: exact own-region/own-tilt/neighbor ordering, per-sector
  widths, individual neighbor slots, and no invented padding.
- Zero padding: fixed block offsets, zeros only in unused neighbor slots, and
  custom/default max_neighbors.
- Consistency between both representations and num_features_for(), including
  heterogeneous neighbor counts, zero-valued real features, and empty inputs.
- Inputs and outputs do not alias, and unknown state/padding options are rejected.
- Small companion checks: reward pass-through including NaN and tilt-index mapping.

Expected feature arrays are explicit, independently specified examples. These
checks call production functions, not copies of their implementation. They do
not verify upstream region aggregation or the usefulness of the RL observation.

Run from repository root: python tests/drl/test_interface.py
"""
import logging
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from drl.interface import (compute_region_state, num_features_for,
                           compute_reward, tilt_idx_to_deg)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        # Distinct numbers identify every feature block and sector; zeros are
        # real data too. own_occupancy/own_coverage are FIXED-width (2
        # own regions, identical for every sector) -- only the neighbor
        # portion varies per sector (2, 1, and 0 neighbors respectively).
        self.own_tilt = np.array([.5,.75,0.])
        self.own_occupancy = np.array([[.2,.3],[.6,.15],[.05,.09]])
        self.own_coverage = np.array([[.4,.7],[0.,.55],[.11,.22]])
        self.neighbor_tilts = [np.array([.1,.9]),np.array([.8]),np.array([])]
        self.neighbor_occupancy = [np.array([.11,.91]),np.array([.81]),np.array([])]
        self.neighbor_coverage = [np.array([.12,.92]),np.array([.82]),np.array([])]

    def state(self, mode, **kwargs):
        return compute_region_state(self.own_tilt,self.neighbor_tilts,self.own_occupancy,
                                    self.own_coverage,self.neighbor_occupancy,
                                    self.neighbor_coverage,mode,**kwargs)

    def test_variable_exact_feature_order_and_widths(self):
        # Own region values precede own tilt and separate neighbor values, without gaps.
        actual = self.state('variable')
        expected = [[.2,.3,.4,.7,.5,.11,.91,.12,.92,.1,.9],
                   [.6,.15,0.,.55,.75,.81,.82,.8],
                   [.05,.09,.11,.22,0.]]
        self.assertIsInstance(actual,list)
        self.assertEqual(len(actual),3)
        for row,want in zip(actual,expected):
            self.assertEqual(row.ndim,1)
            np.testing.assert_array_equal(row,want)
        self.assertEqual([len(row) for row in actual],[11,8,5])

    def test_zero_padding_exact_block_positions(self):
        # With 2 own regions and max_neighbors=2: occupancy 0:2, coverage 2:4,
        # own tilt 4, neighbor occupancy 5:7, coverage 7:9, tilt 9:11.
        actual = self.state('zero',max_neighbors=2)
        expected = [[.2,.3,.4,.7,.5,.11,.91,.12,.92,.1,.9],
                   [.6,.15,0.,.55,.75,.81,0.,.82,0.,.8,0.],
                   [.05,.09,.11,.22,0.,0.,0.,0.,0.,0.,0.]]
        self.assertIsInstance(actual,np.ndarray)
        self.assertEqual(actual.shape,(3,11))
        np.testing.assert_array_equal(actual,expected)

    def test_default_max_neighbors_feature_layout(self):
        # Default max_neighbors=4 -> width = 2*2+1+3*4 = 17. Blocks: occupancy
        # 0:2, coverage 2:4, own tilt 4, neighbor occupancy 5:9, coverage 9:13,
        # tilt 13:17.
        actual = self.state('zero')
        self.assertEqual(actual.shape,(3,17))
        np.testing.assert_array_equal(actual[0,:2],[.2,.3])
        np.testing.assert_array_equal(actual[0,2:4],[.4,.7])
        np.testing.assert_array_equal(actual[:,4],self.own_tilt)
        np.testing.assert_array_equal(actual[0,5:7],[.11,.91])
        for block in (actual[0,7:9],actual[0,11:13],actual[0,15:17]):
            np.testing.assert_array_equal(block,np.zeros_like(block))

    def test_full_capacity_preserves_every_feature(self):
        # When no neighbor slots are unused, variable and fixed representations must coincide.
        own_tilt = np.array([.6])
        own_occupancy = np.array([[0.,.5,1.]])
        own_coverage = np.array([[1.,.5,0.]])
        neighbor_tilts = [np.array([.1,.2,.3,.4])]
        neighbor_occupancy = [np.array([.1,.2,.3,.4])]
        neighbor_coverage = [np.array([.9,.8,.7,.6])]
        fixed = compute_region_state(own_tilt,neighbor_tilts,own_occupancy,own_coverage,
                                     neighbor_occupancy,neighbor_coverage,'zero')
        variable = compute_region_state(own_tilt,neighbor_tilts,own_occupancy,own_coverage,
                                        neighbor_occupancy,neighbor_coverage,'variable')
        self.assertEqual(fixed.shape,(1,19))
        np.testing.assert_array_equal(fixed[0],variable[0])
        np.testing.assert_array_equal(fixed[0,:3],own_occupancy[0])
        np.testing.assert_array_equal(fixed[0,3:6],own_coverage[0])
        np.testing.assert_array_equal(fixed[0,6],.6)
        np.testing.assert_array_equal(fixed[0,7:11],[.1,.2,.3,.4])
        np.testing.assert_array_equal(fixed[0,11:15],[.9,.8,.7,.6])
        np.testing.assert_array_equal(fixed[0,15:19],[.1,.2,.3,.4])

    def test_representations_agree_when_padding_removed(self):
        # Removing only unused neighbor slots must recover the complete variable observation.
        fixed = self.state('zero',max_neighbors=2)
        variable = self.state('variable')
        # Explicit indices retain genuine zero features (e.g. sector 1 coverage).
        for sector,indices in enumerate(([0,1,2,3,4,5,6,7,8,9,10],[0,1,2,3,4,5,7,9],[0,1,2,3,4])):
            np.testing.assert_array_equal(fixed[sector,indices],variable[sector])

    def test_real_zeros_and_neighbor_order_are_preserved(self):
        # Zeros must not be filtered out, and neighbors must not be sorted or averaged.
        actual = compute_region_state(np.array([0.]),[np.array([.9,0.,.2])],
                                      np.array([[0.,.4]]),np.array([[0.,1.]]),
                                      [np.array([.5,0.,.3])],[np.array([.6,0.,.1])],'variable')
        np.testing.assert_array_equal(actual[0],[0.,.4,0.,1.,0.,.5,0.,.3,.6,0.,.1,.9,0.,.2])

    def test_empty_sector_collection(self):
        # An empty collection still has a well-defined fixed-width batch shape.
        empty_own = np.zeros((0,2))
        self.assertEqual(compute_region_state(np.array([]),[],empty_own,empty_own,[],[],'variable'),[])
        self.assertEqual(compute_region_state(np.array([]),[],empty_own,empty_own,[],[],'zero').shape,(0,17))

    def test_no_regions_or_neighbors_retains_own_tilt(self):
        # Zero own regions and zero configured max_neighbors yield a one-feature observation, the own tilt.
        no_regions = np.zeros((1,0))
        actual = compute_region_state(np.array([.7]),[np.array([])],no_regions,no_regions,
                                      [np.array([])],[np.array([])],'zero',max_neighbors=0)
        np.testing.assert_array_equal(actual,[[.7]])

    def test_construction_preserves_inputs_and_output_is_independent(self):
        # Both modes must allocate observations without modifying or sharing caller storage.
        for mode in ('variable','zero'):
            with self.subTest(mode=mode):
                inputs = [self.own_tilt,self.own_occupancy,self.own_coverage,
                         *self.neighbor_tilts,*self.neighbor_occupancy,*self.neighbor_coverage]
                saved = [x.copy() for x in inputs]
                result = self.state(mode)
                for actual,original in zip(inputs,saved):
                    np.testing.assert_array_equal(actual,original)
                for row in result:
                    row[:] = -999
                for actual,original in zip(inputs,saved):
                    np.testing.assert_array_equal(actual,original)

    def test_unknown_padding_rejected(self):
        # Unsupported modes should not silently select another representation.
        with self.assertRaises(ValueError):
            self.state('unsupported')

    def test_feature_counts_match_actual_observations(self):
        # DQN input sizes must match constructed rows for every sector and both modes.
        for mode in ('variable','zero'):
            with self.subTest(mode=mode):
                counts = num_features_for('region_aligned',num_own_regions=2,
                                          num_neighbors_per_sector=[2,1,0],state_padding=mode,
                                          max_neighbors=2)
                actual = self.state(mode,max_neighbors=2)
                if mode == 'variable':
                    self.assertEqual(counts,[11,8,5])
                    self.assertEqual(counts,[len(row) for row in actual])
                else:
                    self.assertEqual(counts,11)
                    self.assertEqual(counts,actual.shape[1])


class FeatureCountAndCompanionTests(unittest.TestCase):
    def test_default_custom_and_empty_feature_counts(self):
        # Check default max_neighbors, custom max_neighbors, and an empty variable batch.
        self.assertEqual(num_features_for('region_aligned',num_own_regions=2,state_padding='zero'),17)
        self.assertEqual(
            num_features_for('region_aligned',num_own_regions=2,state_padding='zero',max_neighbors=2),11)
        self.assertEqual(
            num_features_for('region_aligned',num_own_regions=2,num_neighbors_per_sector=[],
                             state_padding='variable'),[])
        self.assertEqual(
            num_features_for('region_aligned',num_own_regions=2,num_neighbors_per_sector=[2,1,0],
                             state_padding='variable'),[11,8,5])

    def test_unknown_feature_type_or_padding_rejected(self):
        # Configuration typos must fail explicitly rather than produce incorrect network widths.
        for state,padding in (('unknown','zero'),('region_aligned','unknown')):
            with self.subTest(state=state,padding=padding):
                with self.assertRaises(ValueError):
                    num_features_for(state,num_own_regions=2,state_padding=padding)

    def test_reward_pass_through_preserves_nan(self):
        # Already-aggregated rewards remain unchanged, including no-data NaNs.
        coverage = np.array([0.,.6,1.,np.nan])
        np.testing.assert_array_equal(compute_reward('sector_neighbors',coverage),coverage)
        with self.assertRaises(ValueError):
            compute_reward('unknown',coverage)

    def test_tilt_index_mapping(self):
        # Nonuniform candidates reveal arithmetic-based mapping errors; repeated indices remain repeated.
        sweep = np.array([0.,2.,7.,15.])
        indices = np.array([3,0,2,2,1])
        np.testing.assert_array_equal(tilt_idx_to_deg(indices,sweep),[15.,0.,7.,7.,2.])
        np.testing.assert_array_equal(sweep,[0.,2.,7.,15.])
        np.testing.assert_array_equal(indices,[3,0,2,2,1])


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
