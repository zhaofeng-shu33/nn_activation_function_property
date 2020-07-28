import unittest
import numpy as np

from sklearn import linear_model

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

    def test_quasi_fit_predict(self):
        x = np.array([[1, 2], [3, 4.1], [5, 6], [6.9, 8.1]], dtype=np.float)
        y = np.array([3, 6, 8, 11.8], dtype=np.float)
        model = QuasiLinearRegression(epsilon=0.05)
        model._quasi_fit(x, y)
        reg = linear_model.LinearRegression()
        reg.fit(x, y)
        x_test = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float)
        y_predict = model._quasi_predict(x_test)
        y_linear_predict = reg.predict(x_test)
        print(y_predict, y_linear_predict)

if __name__ == '__main__':
    unittest.main()
