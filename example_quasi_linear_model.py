# comparison of quasi linear model
# with linear model (Least Square)
import numpy as np

from sklearn import linear_model
from sklearn.metrics import mean_squared_error

from quasi_linear_model import QuasiLinearRegression

def get_linear_prediction_error(X, Y):
    reg = linear_model.LinearRegression()
    reg.fit(X, Y)
    Y_pred_linear = reg.predict(X)
    return mean_squared_error(Y, Y_pred_linear)

def get_quasi_linear_prediction_error(X, Y):
    reg = QuasiLinearRegression()
    reg.fit(X, Y)
    Y_pred_quasilinear = reg.predict(X)
    return mean_squared_error(Y, Y_pred_quasilinear)
if __name__ == "__main__":
    X = [[1,2],[3,4],[5,6],[7,8]]
    Y = [3, 6, 8, 12]
    error_linear = get_linear_prediction_error(X, Y)
    error_quasi_linear = get_quasi_linear_prediction_error(X, Y)
    print(error_linear, error_quasi_linear)