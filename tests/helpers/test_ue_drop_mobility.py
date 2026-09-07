import unittest
from support import np, torch, topology, numpy
from helpers.ue_drop import UeDropper
from helpers.mobility import ReferencePointGroupMobility

class DropContract(unittest.TestCase):
    def setUp(self):
        self.topo = topology()
        self.drop = UeDropper(self.topo, device='cpu', generator=torch.Generator().manual_seed(71))

    def test_disk_uniform_area_and_angular_constraint(self):
        points = numpy(self.drop.disk_offset(20., 20000))
        r2 = np.sum(points**2, axis=1)/400.
        self.assertTrue(np.all((r2 >= 0) & (r2 <= 1.000001)))
        self.assertAlmostEqual(r2.mean(), .5, delta=.015)
        hist, _ = np.histogram(r2, bins=np.linspace(0,1,11))
        self.assertTrue(np.all(np.abs(hist-2000) < 200))
        wedge = numpy(self.drop.disk_offset(20., 2000, angle_center=0., angle_half_width=.2))
        self.assertTrue(np.all(np.abs(np.arctan2(wedge[:,1], wedge[:,0])) <= .20001))

    def test_valid_points_and_clustered_height(self):
        centers = self.drop.cluster_centers(8)
        self.assertEqual(len(centers), 8)
        points = self.drop.valid_points(torch.tensor(centers), 10., min_dist_from_site=35.)
        self.assertTrue(self.topo.is_within_coverage(points).all())
        sites = numpy(self.topo.bs_loc)[0, ::3, :2]
        self.assertTrue(np.all(np.linalg.norm(numpy(points)[:,None,:]-sites[None,:,:], axis=-1) >= 35.-1e-4))
        loc, groups = self.drop.clustered(centers, 3, 10., 1.5)
        self.assertEqual(tuple(loc.shape), (24,3))
        self.assertEqual(groups.dtype, torch.long)
        self.assertTrue(torch.all(loc[:,2] == 1.5))
        self.assertTrue(torch.all((groups >= 0) & (groups < 8)))
        np.testing.assert_array_equal(np.bincount(numpy(groups), minlength=8), np.full(8,3))

    def test_cluster_centers_respect_min_bs_ut_dist(self):
        # A cluster's own reference point is always its first waypoint
        # (ReferencePointGroupMobility) -- it must respect the same
        # exclusion zone as any other UE/waypoint, not just be inside
        # coverage.
        centers = numpy(torch.tensor(self.drop.cluster_centers(40)))
        sites = numpy(self.topo.bs_loc)[0, ::3, :2]
        nearest = np.linalg.norm(centers[:, None, :] - sites[None, :, :], axis=-1).min(axis=1)
        self.assertTrue(np.all(nearest >= float(self.topo.min_bs_ut_dist) - 1e-4),
                        f"cluster center(s) within min_bs_ut_dist of a site: min nearest={nearest.min():.2f}")

    def test_impossible_drop_warns(self):
        with self.assertLogs(level='WARNING') as logs:
            self.drop.valid_points(torch.tensor([[100., 0.]]), 1., max_rounds=2,
                                   min_dist_from_site=1e9)
        self.assertTrue(any('valid' in line.lower() for line in logs.output))

class MobilityContract(unittest.TestCase):
    def make(self, mode='random', **kwargs):
        t = topology()
        return ReferencePointGroupMobility(torch.tensor([[105.,0.,1.5],[95.,0.,1.5]]),
            torch.zeros(2, dtype=torch.long), [(100.,0.)], 20., t, 1., 2.,
            generator=torch.Generator().manual_seed(12), cluster_mobility_mode=mode,
            intra_cluster_mobility='static', **kwargs), t

    def test_static_offsets_coverage(self):
        m,t = self.make()
        offsets = m.ut_loc[:,:2] - m.ref_xy[0]
        for _ in range(30):
            positions = m.step(.1)
            self.assertTrue(t.is_within_coverage(positions[:,:2]).all())
            np.testing.assert_allclose(numpy(positions[:,:2]-m.ref_xy[0]), numpy(offsets), atol=1e-4)

    def test_markov_rejects_nonstochastic_matrix(self):
        with self.assertRaises((AssertionError, ValueError)):
            self.make('markov', num_waypoints=2, waypoint_hold_steps=2,
                      waypoint_transition_matrix=[[.1,.1],[.2,.2]])

    def test_periodic_hold_and_cycle(self):
        waypoints = torch.tensor([[[100.,0.],[140.,0.],[100.,40.]]])
        m,t = self.make('periodic', num_waypoints=3, waypoint_hold_steps=2, waypoints=waypoints)
        for step in range(1,13):
            positions = m.step(1.)
            np.testing.assert_allclose(numpy(m.ref_xy[0]), numpy(waypoints[0,(step//2)%3]), atol=1e-5)
            self.assertTrue(t.is_within_coverage(positions[:,:2]).all())

    def test_random_jump_holds_then_teleports(self):
        m,t = self.make('random', transition='jump', waypoint_hold_steps=3)
        positions = [m.ref_xy[0].clone()]
        for _ in range(40):
            m.step(1.)
            positions.append(m.ref_xy[0].clone())
        moves = [float(torch.linalg.norm(positions[i+1]-positions[i])) for i in range(len(positions)-1)]
        # Every move is either ~0 (holding) or a real jump (teleport) --
        # never a small partial step, since transition="jump" never
        # interpolates.
        self.assertTrue(all(mv < 1e-6 or mv > 1.0 for mv in moves))
        self.assertTrue(any(mv > 1.0 for mv in moves), "expected at least one jump in 40 steps")
        self.assertTrue(any(mv < 1e-6 for mv in moves), "expected at least one held (non-moving) step")

    def test_periodic_smooth_holds_then_walks_to_arrival(self):
        waypoints = torch.tensor([[[100.,0.],[140.,0.],[100.,40.]]])
        m,t = self.make('periodic', transition='smooth', num_waypoints=3, waypoint_hold_steps=2,
                        waypoints=waypoints, transition_duration_s=10.)
        # Holds at waypoint 0 for the first waypoint_hold_steps calls.
        for _ in range(2):
            m.step(1.)
            np.testing.assert_allclose(numpy(m.ref_xy[0]), numpy(waypoints[0,0]), atol=1e-5)
        # Then walks continuously (small, monotonically-decreasing distance
        # to the target) rather than teleporting -- speed is derived from
        # transition_duration_s, so a 40m gap at dt=1s arrives in EXACTLY
        # transition_duration_s=10 steps, not a random/variable count.
        prev_dist = float(torch.linalg.norm(m.ref_xy[0]-waypoints[0,1]))
        for step in range(1, 11):
            m.step(1.)
            dist = float(torch.linalg.norm(m.ref_xy[0]-waypoints[0,1]))
            self.assertLessEqual(dist, prev_dist + 1e-6)
            self.assertTrue(t.is_within_coverage(m.ref_xy[:1]).all())
            if step < 10:
                self.assertGreater(dist, 1e-5, f"arrived early at step {step}, expected exactly step 10")
            prev_dist = dist
        self.assertLess(dist, 1e-5, "did not arrive at exactly transition_duration_s=10 steps")

if __name__ == '__main__':
    unittest.main()
