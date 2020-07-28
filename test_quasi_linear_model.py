import unittest
import numpy as np

from quasi_linear_model import QuasiLinearRegression

class TestQuasiLinearModel(unittest.TestCase):
    def test_get_model_parameter(self):
        x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float)
        y = np.array([3, 6, 8], dtype=np.float)
        model = QuasiLinearRegression()
        model.fit(x, y)
        x_test = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float)
        y_predict = model.predict(x_test)
        print(y_predict)

if __name__ == '__main__':
    unittest.main()
