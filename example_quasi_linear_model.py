# comparison of quasi linear model
# with linear model (Least Square)
import numpy as np
import argparse

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

def get_quasi_approximation_linear_prediction_error(X, Y):
    reg = QuasiLinearRegression()
    reg._quasi_fit(X, Y)
    Y_pred_quasilinear = reg._quasi_predict(X)
    return mean_squared_error(Y, Y_pred_quasilinear)

def artificial_dataset(sample_size=10):
    # generate data by formula y = z + 0.05 * z^3
    # z = 3 * x_1 + 4 * x_2 - 2
    # where (x_1, x_2) is generated from uniform random number
    np.random.seed(12)
    X = np.random.random([sample_size, 2])
    Z = X @ np.array([3, 4]) - 2
    Y = Z + 0.05 * np.power(Z, 3.0)
    return (X, Y)

def toy_dataset():
    X = [[1, 2], [3, 4.1], [5, 6], [6.9, 8.1]]
    Y = [3, 6, 8, 11.8]
    return (X, Y)

def compare_two_method(X, Y):
    error_linear = get_linear_prediction_error(X, Y)
    error_quasi_linear = get_quasi_linear_prediction_error(X, Y)
    error_quasi_approximation_linear = get_quasi_approximation_linear_prediction_error(X, Y)
    print(error_linear, error_quasi_linear, error_quasi_approximation_linear)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['toy', 'artificial'], default='toy')
    args = parser.parse_args()
    if args.dataset == 'toy':
        X, Y = toy_dataset()
    else:
        X, Y = artificial_dataset()
    compare_two_method(X, Y)
