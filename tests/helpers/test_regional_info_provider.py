import unittest
from support import np, topology
from helpers.regional_info_provider import RegionalInfoProvider


class ConstructionAndGeometryContract(unittest.TestCase):
    """Own-region geometry is now a static, sector-independent block-pooling
    of the rhombus fine grid -- no per-sector search, so these checks are
    about the mapping itself and its integration with real topology
    neighbor_ids, not an expensive offset search like the design this
    replaces."""

    def test_grid_n_not_divisible_by_block_size_rejected(self):
        with self.assertRaises(ValueError):
            RegionalInfoProvider(np.array([[-1]]), grid_n=5, block_size=2)

    def test_unknown_coverage_fill_rejected(self):
        with self.assertRaises(ValueError):
            RegionalInfoProvider(np.array([[-1]]), grid_n=2, block_size=1, coverage_fill="bogus")

    def test_num_own_regions_and_cell_to_region_mapping(self):
        est = RegionalInfoProvider(np.array([[-1], [-1]]), grid_n=6, block_size=2)
        self.assertEqual(est.num_own_regions, 9)
        self.assertEqual(est.cell_to_region.shape, (36,))
        self.assertEqual(est.cell_to_region.min(), 0)
        self.assertEqual(est.cell_to_region.max(), 8)
        # cells (alpha,beta) in {0,1}x{0,1} -- cell_idx 0,1,6,7 in row-major
        # order over grid_n=6 -- all pool into the same top-left region.
        self.assertTrue(np.all(est.cell_to_region[[0, 1, 6, 7]] == 0))
        # the very next block over (beta 2,3) is a distinct region.
        self.assertEqual(est.cell_to_region[2], 1)

    def test_neighbor_count_matches_topology(self):
        topo = topology()
        topo.build_sector_rhombus_grid(50.0)
        est = RegionalInfoProvider(topo.neighbor_ids, grid_n=6, block_size=2)
        for s in range(topo.bs_loc.shape[1]):
            n_neighbors = int((topo.neighbor_ids[s] >= 0).sum())
            self.assertEqual(len(est.neighbor_sets[s]), n_neighbors)

    def test_num_own_regions_identical_for_every_sector(self):
        # Unlike the earlier per-sector search-based design, every sector's
        # own-region count is now the same by construction.
        topo = topology()
        topo.build_sector_rhombus_grid(50.0)
        est = RegionalInfoProvider(topo.neighbor_ids, grid_n=6, block_size=2)
        num_bs = topo.bs_loc.shape[1]
        own_occupancy, *_ = est.compute(np.ones(num_bs, dtype=bool))
        self.assertEqual(own_occupancy.shape, (num_bs, 9))


class OwnRegionStateContract(unittest.TestCase):
    def setUp(self):
        # grid_n=2, block_size=1 -> 4 own regions, cell_to_region == cell_idx directly.
        self.est = RegionalInfoProvider(np.array([[-1]]), grid_n=2, block_size=1, occupancy_ema_alpha=1.0)

    def test_unvisited_region_coverage_is_zero_not_negative_or_nan(self):
        self.est.accumulate(np.array([0]), np.array([0]), np.array([True]))
        _, own_coverage, _, _, _, state_ready = self.est.compute(np.array([True]))
        self.assertTrue(state_ready[0])
        self.assertTrue(np.all(np.isfinite(own_coverage[0])))
        self.assertTrue(np.all((own_coverage[0] >= 0) & (own_coverage[0] <= 1)))
        self.assertEqual(own_coverage[0][0], 1.0, "the one visited+covered region")
        self.assertTrue(np.all(own_coverage[0][1:] == 0.0), "unvisited regions read exactly 0.0")

    def test_occupancy_normalizes_by_network_wide_visits_since_last_reset(self):
        self.est.accumulate(np.array([0]), np.array([0]), np.array([True]))
        own_occupancy, *_ = self.est.compute(np.array([True]))
        self.assertAlmostEqual(float(own_occupancy[0].sum()), 1.0, places=6)


class NeighborCoarsePairContract(unittest.TestCase):
    """A sector's own coarse (occupancy, coverage) feedback fed to a
    neighbor is its own WHOLE-GRID aggregate -- independent of its own
    region breakdown, not a collapse of it (see module docstring)."""

    def test_neighbor_coarse_pair_is_neighbors_own_whole_grid_aggregate(self):
        est = RegionalInfoProvider(np.array([[1], [-1]]), grid_n=2, block_size=1, occupancy_ema_alpha=1.0)
        # sector 0: one UE in its own cell 0. sector 1: two UEs (cell 0
        # covered, cell 1 not) -- sector 1's whole-grid coverage is 0.5,
        # independent of which of its 4 regions each UE fell in.
        est.accumulate(np.array([0, 1, 1]), np.array([0, 0, 1]), np.array([True, True, False]))
        own_occupancy, _, neighbor_occupancy, neighbor_coverage, _, _ = est.compute(np.array([True, True]))
        self.assertAlmostEqual(neighbor_coverage[0][0], 0.5, places=6)
        # 3 total network visits since last reset; sector 1 made 2 of them.
        self.assertAlmostEqual(neighbor_occupancy[0][0], 2 / 3, places=6)
        self.assertAlmostEqual(float(own_occupancy[0].sum()), 1 / 3, places=6)


class RegionalRewardContract(unittest.TestCase):
    """Reward is accumulated directly over sector s's OWN window (in_xs =
    UEs geographically in [s] + its neighbor_ids), never pooled from a
    neighbor's own separately-timed last-completed window -- see
    RegionalInfoProvider's module docstring for why an earlier,
    neighbor-pooling design was a real temporal-misalignment bug."""

    def _est(self, neighbor_ids):
        return RegionalInfoProvider(np.array(neighbor_ids), grid_n=1, block_size=1)

    def test_reward_pools_over_own_window_using_neighbor_ids(self):
        # sector 0's X_s = {0,1}; sector 1's own X_s = {1} only (neighbor_ids
        # is per-owner, not symmetric here -- irrelevant to this test).
        e = self._est([[1], [-1]])
        e.accumulate(np.array([0, 1]), np.array([0, 0]), np.array([True, False]))
        *_, reward, _ = e.compute(np.array([True, False]))
        # sector 0's X_s={0,1} sees BOTH UEs (1 covered, 1 not) -> 1/2 = 0.5.
        self.assertAlmostEqual(reward[0], 0.5, places=6)

    def test_reward_available_without_waiting_on_any_other_sector(self):
        e = self._est([[1], [-1]])
        e.accumulate(np.array([0]), np.array([0]), np.array([True]))
        *_, reward, _ = e.compute(np.array([True, False]))
        # sector 0 closes; sector 1 has NEVER closed -- reward must still
        # be defined (no neighbor-completion dependency at all).
        self.assertAlmostEqual(reward[0], 1.0, places=6)

    def test_reward_reflects_only_sectors_own_current_window(self):
        e = self._est([[-1]])
        e.accumulate(np.array([0]), np.array([0]), np.array([False]))
        e.compute(np.array([True]))  # closes and resets; this window's reward (0.0) isn't checked here
        e.accumulate(np.array([0, 0]), np.array([0, 0]), np.array([True, True]))
        *_, reward, _ = e.compute(np.array([True]))
        # Only the SECOND window's 2 covered UEs count -- the first
        # window's uncovered UE must not bleed into this one.
        self.assertAlmostEqual(reward[0], 1.0, places=6)

    def test_isolated_sector_reward_matches_simple_ratio(self):
        e = self._est([[-1]])
        e.accumulate(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([True, True, False]))
        *_, reward, _ = e.compute(np.array([True]))
        self.assertAlmostEqual(reward[0], 2 / 3, places=6)

    def test_reward_nan_when_zero_visits_in_own_window(self):
        e = self._est([[-1], [-1]])
        e.accumulate(np.array([1]), np.array([0]), np.array([True]))  # UE geo in sector 1, outside X_s={0}
        *_, reward, _ = e.compute(np.array([True, False]))
        self.assertTrue(np.isnan(reward[0]))


class KrigedAsyncSafetyContract(unittest.TestCase):
    """A region (and a neighbor's whole grid) never spans more than one
    geographic sector now, so the async-safety property that used to guard
    a shared multi-sector region now guards a sector's own readiness and
    the coarse coverage/occupancy it hands to a neighbor: a non-closing
    (or zero-visit) sector's LAST completed snapshot is used, never its
    live/partial one, and dependents stay not-ready until it exists."""

    def _est(self):
        return RegionalInfoProvider(np.array([[1], [-1]]), grid_n=1, block_size=1, coverage_fill="kriged")

    def test_non_closing_neighbor_contributes_its_last_completed_snapshot_not_live_partial(self):
        est = self._est()

        # First call: both sectors close, with clean, fully-resolved coverage.
        _, own_coverage, _, neighbor_coverage, _, state_ready = est.compute(
            np.array([True, True]), coverage_per_cell=np.array([[0.8], [0.4]]))
        self.assertTrue(state_ready[0])
        self.assertAlmostEqual(own_coverage[0][0], 0.8, places=6)
        self.assertAlmostEqual(neighbor_coverage[0][0], 0.4, places=6)

        # Second call: sector 0 closes again; sector 1 does NOT -- its
        # "live" value here is a partial-window -1 sentinel that must NOT
        # leak into sector 0's neighbor-coarse feedback.
        _, own_coverage, _, neighbor_coverage, _, state_ready = est.compute(
            np.array([True, False]), coverage_per_cell=np.array([[0.9], [-1.0]]))
        self.assertTrue(state_ready[0])
        self.assertAlmostEqual(own_coverage[0][0], 0.9, places=6)
        # Sector 1 hasn't closed -- its contribution must still be its LAST
        # completed value (0.4), not this call's live -1 sentinel.
        self.assertAlmostEqual(neighbor_coverage[0][0], 0.4, places=6)

    def test_not_ready_until_neighbor_has_a_valid_snapshot(self):
        est = self._est()
        # sector 0 closes with a valid map; sector 1 has never closed.
        _, _, _, neighbor_coverage, _, state_ready = est.compute(
            np.array([True, False]), coverage_per_cell=np.array([[0.8], [-1.0]]))
        self.assertFalse(state_ready[0], "state depends on neighbor 1 too -- not ready until it has a valid snapshot")
        self.assertEqual(neighbor_coverage[0][0], 0.0, "not-ready neighbor coverage must read 0.0, not a sentinel")

    def test_zero_visit_close_is_not_mistaken_for_a_valid_completed_snapshot(self):
        est = self._est()
        # BOTH close, but sector 1 had zero visits -- SpatialGridEstimator's
        # own contract leaves its entire row at -1, which must NOT count as
        # "completed" even though sector 1 technically closed this call.
        _, _, _, neighbor_coverage, _, state_ready = est.compute(
            np.array([True, True]), coverage_per_cell=np.array([[0.8], [-1.0]]))
        self.assertFalse(state_ready[0], "sector 1's all-sentinel row must not count as a valid completed snapshot")
        self.assertEqual(neighbor_coverage[0][0], 0.0)


if __name__ == '__main__':
    unittest.main()
