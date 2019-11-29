import unittest
import numpy as np
import tensorflow as tf
from optimization import get_spherical_coordinate
from optimization import generate_uniform_sample

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

if __name__ == '__main__':
    unittest.main()
