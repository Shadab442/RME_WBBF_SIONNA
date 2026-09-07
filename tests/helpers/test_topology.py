import unittest
from support import np, torch, topology, numpy

class TopologyContract(unittest.TestCase):
    def test_layout_adjacency_and_padding(self):
        t = topology(batch_size=2)
        self.assertEqual(t.num_bs, t.num_cells * 3)
        self.assertEqual(tuple(t.bs_loc.shape), (2, t.num_bs, 3))
        loc = numpy(t.bs_loc).reshape(2, t.num_cells, 3, 3)
        np.testing.assert_allclose(loc, np.repeat(loc[:, :, :1], 3, axis=2))
        np.testing.assert_allclose(numpy(t.bs_orientations)[0, :, 0],
                                   np.tile(np.deg2rad([60, 180, 300]), t.num_cells), atol=1e-6)
        a = numpy(t.sector_adjacency)
        np.testing.assert_array_equal(a, a.T)
        self.assertFalse(np.diag(a).any())
        neighbors = numpy(t.neighbor_ids)
        self.assertEqual(neighbors.shape, (t.num_bs, t.max_neighbors))
        for s in range(t.num_bs):
            np.testing.assert_array_equal(np.sort(neighbors[s][neighbors[s] >= 0]), np.flatnonzero(a[s]))
            self.assertTrue(np.all(neighbors[s][neighbors[s] < 0] == -1))
            for sibling in range(s // 3 * 3, s // 3 * 3 + 3):
                if sibling != s:
                    self.assertTrue(a[s, sibling])

    def test_coverage_and_drop_radius(self):
        t = topology()
        sites = numpy(t.bs_loc)[0, ::3, :2]
        expected = np.linalg.norm(sites, axis=1).max() + (500./np.sqrt(3))
        self.assertAlmostEqual(float(t.default_drop_radius), expected, places=3)
        self.assertGreater(t.default_drop_radius, (500./np.sqrt(3)))
        self.assertTrue(np.isfinite(t.default_drop_radius))
        self.assertTrue(t.is_within_coverage(torch.tensor(sites)).all())
        angles = np.linspace(0, 2*np.pi, 360, endpoint=False)
        xy = t.default_drop_radius * np.column_stack([np.cos(angles), np.sin(angles)])
        self.assertFalse(t.is_within_coverage(torch.tensor(xy, dtype=torch.float32)).all())

    def test_truncation_disables_wraparound(self):
        t = topology(num_sites=2, batch_size=2)
        self.assertEqual((t.num_cells, t.num_bs), (2, 6))
        users = torch.tensor([[[0., 0., 1.5], [10000., -20000., 1.5]]]).repeat(2, 1, 1)
        expected = t.bs_loc[:, :, None, :].expand(2, 6, 2, 3)
        np.testing.assert_array_equal(numpy(t.mirror_bs_loc(users)), numpy(expected))

    def test_grid_coordinates_and_round_trip(self):
        t = topology(num_sites=1)
        points, n = t.build_sector_rhombus_grid((500./np.sqrt(3))/3.2)
        self.assertEqual(n, 4)
        self.assertEqual(points.shape, (3, n*n, 2))
        for s, yaw in enumerate(np.deg2rad([60, 180, 300])):
            e1 = np.array([np.cos(yaw-np.pi/3), np.sin(yaw-np.pi/3)])
            e2 = np.array([np.cos(yaw+np.pi/3), np.sin(yaw+np.pi/3)])
            expected = [numpy(t.bs_loc)[0, s, :2] + (500./np.sqrt(3))*((a+.5)*e1+(b+.5)*e2)/n
                        for a in range(n) for b in range(n)]
            np.testing.assert_allclose(points[s], expected, atol=1e-4)
            sectors, cells = t.assign_to_sector_rhombus_grid(points[s])
            np.testing.assert_array_equal(sectors, np.full(n*n, s))
            np.testing.assert_array_equal(cells, np.arange(n*n))
        sectors, cells = t.assign_to_sector_rhombus_grid(np.array([[1e7, 1e7]]))
        np.testing.assert_array_equal(sectors, [-1])
        np.testing.assert_array_equal(cells, [-1])

    def test_observation_window_membership(self):
        t = topology()
        size = (500./np.sqrt(3))/4
        _, n = t.build_sector_rhombus_grid(size)
        windows = t.compute_observation_windows(size)
        names = {'center', 'edge_beta0', 'edge_alpha0', 'edge_alphaN', 'edge_betaN',
                 'corner_C', 'corner_E1', 'corner_Far', 'corner_E2'}
        self.assertEqual(len(windows), t.num_bs)
        for s, window in enumerate(windows):
            self.assertEqual(set(window), names)
            self.assertEqual(len(window['center']), 4)
            self.assertTrue(all(i == s for i, _ in window['center']))
            self.assertEqual(len(window['corner_C']), 3)
            self.assertEqual({i for i, _ in window['corner_C']}, set(range(s//3*3, s//3*3+3)))
            for name, members in window.items():
                if name.startswith('edge_'):
                    self.assertIn(len(members), (2, 4))
                if name.startswith('corner_'):
                    self.assertTrue(1 <= len(members) <= 6)
                self.assertTrue(all(0 <= i < t.num_bs and 0 <= c < n*n for i, c in members))

    def test_hexagon_vertex_to_edge_radius_ratio(self):
        t = topology(num_sites=1)
        angles = torch.linspace(0., 2*np.pi, 721)[:-1]
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        center = t.bs_loc[0, 0, :2]
        lower, upper = torch.zeros(720), torch.full((720,), 500.)
        for _ in range(24):
            middle = (lower + upper)/2
            inside = t.is_within_coverage(center + middle[:,None]*directions)
            lower = torch.where(inside, middle, lower)
            upper = torch.where(inside, upper, middle)
        self.assertAlmostEqual(float(lower.max()/lower.min()), 1/np.cos(np.pi/6), delta=1e-4)

    def test_center_ue_needs_no_mirror(self):
        t = topology()
        result = t.mirror_bs_loc(torch.tensor([[[0.,0.,1.5]]]))
        np.testing.assert_allclose(numpy(result[:,:,0,:]), numpy(t.bs_loc), atol=1e-4)

if __name__ == '__main__':
    unittest.main()
