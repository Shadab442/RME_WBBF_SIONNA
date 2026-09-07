import unittest
from support import np, torch, array, ElectricalDowntilt, numpy

class ElectricalContract(unittest.TestCase):
    def test_rectangular_peak_and_tilt_sign(self):
        for tilt in (-12., 0., 12.):
            with self.subTest(tilt=tilt):
                e = ElectricalDowntilt(array(), 3.5e9, tilt)
                theta = torch.deg2rad(torch.linspace(50., 130., 1601))
                af = numpy(e.array_factor(theta))
                self.assertTrue(np.all(af >= 0))
                self.assertAlmostEqual(float(af.max()), 8., places=4)
                self.assertAlmostEqual(float(torch.rad2deg(theta[np.argmax(af)])), 90.+tilt, delta=.051)
                gain = numpy(e.gain_pattern(theta, torch.zeros_like(theta)))
                self.assertEqual(gain.shape, tuple(theta.shape))
                self.assertTrue(np.isfinite(gain).all())
                self.assertTrue((gain >= 0).all())
                self.assertEqual(np.argmax(gain), np.argmax(af))
                e.set_tilt(tilt)
                np.testing.assert_array_equal(numpy(e.array_factor(theta)), af)
                e.set_tilt(tilt)
                np.testing.assert_array_equal(numpy(e.array_factor(theta)), af)

    def test_invalid_arrays_rejected(self):
        for a in (array(cols=2), array(polarization='dual')):
            with self.assertRaises((AssertionError, ValueError)):
                ElectricalDowntilt(a, 3.5e9)

if __name__ == '__main__':
    unittest.main()
