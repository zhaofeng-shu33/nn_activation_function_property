import unittest
from optimization import get_spherical_coordinate
from optimization import generate_uniform_sample

class TestFunc(unittest.TestCase):
    def test_get_spherical_coordinate(self):
        arr = get_spherical_coordinate()
        self.assertAlmostEqual(np.linalg.norm(arr), 1.0)
    def test_generate_uniform_sample(self):
        x_t, y_t = generate_uniform_sample()
        self.assertEqual(x_t.shape, (3,2))
        self.assertEqual(y_t.shape, (3,1))
        
if __name__ == '__main__':
    unittest.main()