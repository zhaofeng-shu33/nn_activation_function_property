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
    def test_generate_uniform_sample(self):
        optimization.n = 3
        optimization.k = 2
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

class TestMN(unittest.TestCase):
    def test_n_ij(self):
        optimization.n = 10
        optimization.k = 8
        old_value = m_not_small.compute_N(2, 4, False)
        new_value = m_not_small.compute_N(2, 4, True)
        self.assertAlmostEqual(old_value, new_value)
    def test_m_ij(self):
        optimization.n = 10
        optimization.k = 8
        old_value = m_not_small.compute_M_without_r(2, 4, False)
        new_value = m_not_small.compute_M_without_r(2, 4, True)
        self.assertAlmostEqual(old_value, new_value)
    def test_m_whole(self):
        optimization.n = 10
        optimization.k = 8
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
    def test_two_poly_derivative(self):
        a2 = optimization.n ** 2 / optimization.k
        a = np.array([-1, 0, a2]) / np.sqrt(2 * optimization.n)
        z = np.array([0, -1, 1, 3])
        value_1 = verification_z_norm.derivative_poly2(z)
        value_2 = verification_z_norm.derivative_poly(z, a)
        for i in range(4):
            self.assertAlmostEqual(value_1[i, i], value_2[i, i])
    def test_coeff_epsilon_2_theoretical(self):
        a2 = optimization.n ** 2 / optimization.k
        q = np.array([-1, 0, a2]) / np.sqrt(2 * optimization.n)
        C_xi_2 = verification_z_norm.get_coeff_epsilon_2_theoretical(q)
        self.assertAlmostEqual(C_xi_2, -(1 - optimization.k / optimization.n))

class TestEmpiricalNormalize(unittest.TestCase):
    def test_empirical_normalize(self):
        q_norm = verification_z_norm.empirical_normalize(1,
                 np.array([-1, 0, 1]))
        print(q_norm)

class TestIntegrate(unittest.TestCase):
    def test_numerical_integrate(self):
        integral_value = m_not_small.numerical_integration(6, 3, 0, 0, 0)
        self.assertAlmostEqual(integral_value, 1.0)
        integral_value_2 = m_not_small.numerical_integration(6, 3, 0, 0, 2)
        self.assertAlmostEqual(integral_value_2, 0.0375)
        integral_value_3 = m_not_small.numerical_integration(6, 3, 1, 0, 2)
        self.assertAlmostEqual(integral_value_3, 0.01875)
        integral_value_4 = m_not_small.numerical_integration(6, 3, 1, 1, 2)
        self.assertAlmostEqual(integral_value_4, 3/(160*7)*(2+15/12))
   
if __name__ == '__main__':
    unittest.main()
