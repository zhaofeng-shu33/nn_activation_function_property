import unittest
import numpy as np
import tensorflow as tf
import optimization
from optimization import get_spherical_coordinate
from optimization import generate_uniform_sample
import verification_z_norm
import m_not_small

class TestFunc(unittest.TestCase):
    def test_get_spherical_coordinate(self):
        arr = get_spherical_coordinate()
        self.assertAlmostEqual(np.linalg.norm(arr), 1.0)
    def test_generate_uniform_sample(self):
        x_t, y_t = generate_uniform_sample()
        self.assertEqual(x_t.shape.as_list(), [3, 2])
        self.assertEqual(y_t.shape.as_list(), [3, 1])
        sess = tf.Session()
        with sess.as_default():
            x = x_t.eval()
            x_norm = x.T @ x
            self.assertAlmostEqual(x_norm[0, 0], 1.0)
            self.assertAlmostEqual(x_norm[1, 1], 1.0)
            self.assertAlmostEqual(x_norm[0, 1], 0.0)
            self.assertAlmostEqual(x_norm[1, 0], 0.0)
            y = y_t.eval()
            self.assertAlmostEqual(np.linalg.norm(y), 1.0)

class TestMN(unittest.TestCase):
    def test_n_ij(self):
        m_not_small.n = 10
        m_not_small.k = 8
        old_value = m_not_small.compute_N(2, 4, False)
        new_value = m_not_small.compute_N(2, 4, True)
        self.assertAlmostEqual(old_value, new_value)
    def test_m_ij(self):
        m_not_small.n = 10
        m_not_small.k = 8
        old_value = m_not_small.compute_M_without_r(2, 4, False)
        new_value = m_not_small.compute_M_without_r(2, 4, True)
        self.assertAlmostEqual(old_value, new_value)
    def test_m_whole(self):
        m_not_small.n = 10
        m_not_small.k = 8
        M = m_not_small.construct_M_without_r(2)
        self.assertEqual(M.shape, (3, 3))
        self.assertAlmostEqual(M[0, 0], 10)
        self.assertAlmostEqual(M[0, 2], 2/3)
        self.assertAlmostEqual(M[2, 2], -1/21)

class TestPoly(unittest.TestCase):
    def test_two_poly(self):
        a2 = optimization.n ** 2 / optimization.k
        a = np.array([-1, 0, a2]) / np.sqrt(2 * optimization.n)
        z = np.array([0, -1, 1, 3])
        value_1 = verification_z_norm.poly2(z)
        value_2 = verification_z_norm.poly(z, a)
        for i in range(4):
            self.assertAlmostEqual(value_1[i], value_2[i])

if __name__ == '__main__':
    unittest.main()
