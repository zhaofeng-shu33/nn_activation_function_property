import unittest
import numpy as np

from quasi_linear_model import QuasiLinearRegression

class TestQuasiLinearModel(unittest.TestCase):
    def test_get_model_parameter(self):
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float)
        y = np.array([3, 6, 8], dtype=np.float)
        model = QuasiLinearRegression()
        model.fit(x, y)

if __name__ == '__main__':
    unittest.main()
