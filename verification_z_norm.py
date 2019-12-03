import numpy as np
import argparse
import optimization
from optimization import get_orthogonal_coordinate, get_spherical_coordinate
# python3 verification_z_norm.py --n 180 --k 120
def generate_z_instance(x, y):
    A = x @ x.T
    return A @ y

def poly(z):
    r = optimization.k * 1.0 / optimization.n
    return (optimization.n * z * z / r - 1) / np.sqrt(2)

def derivative_poly(z):
    r = optimization.k * 1.0 / optimization.n
    diagonal_terms = np.sqrt(2) * optimization.n * z / r
    return np.diag(diagonal_terms)

def sigma(z):
    return z + optimization.epsilon * poly(z)

def get_w_hat(x, y):
    z = generate_z_instance(x, y)
    w_hat = x.T @ (derivative_poly(z) @ (y-z) - poly(z))
    return w_hat

def get_w_estimate(x, y):
    w_bar = x.T @ y
    w = w_bar + optimization.epsilon * get_w_hat(x, y)
    return w 

def get_coeff_epsilon_2(x, y, w_hat):
    z = generate_z_instance(x, y)
    part_1 = np.linalg.norm(x @ w_hat + poly(z)) ** 2
    part_2 = derivative_poly(z) @ x @ w_hat
    part_2 = part_2.T @ (y - z)
    return part_1 - 2 * part_2

def evaluate_coefficient_epsilon_2(num_times):
    total_value = 0
    for i in range(num_times):
        y = get_spherical_coordinate()
        x = get_orthogonal_coordinate()
        w_hat = get_w_hat(x, y)
        total_value += get_coeff_epsilon_2(x, y, w_hat)
    return total_value / num_times

def evaluate_w_hat_estimate(num_times):
    total_value = 0
    for i in range(num_times):
        y = get_spherical_coordinate()
        x = get_orthogonal_coordinate()
        w = get_w_estimate(x, y)
        total_value += np.linalg.norm(y - sigma(x @ w)) ** 2
    return total_value / num_times

def get_average(num_times):
    total_value = 0
    for i in range(num_times):
        y = get_spherical_coordinate()
        x = get_orthogonal_coordinate()
        z = generate_z_instance(x, y)
        xi_z = poly(z)
        total_value += np.linalg.norm(xi_z) ** 2 
    return total_value / num_times

def get_matrix_average(num_times):
    n = optimization.n
    A_total = np.zeros([n, n])
    for i in range(num_times):
        x = get_orthogonal_coordinate()
        A = x @ x.T
        A_total += (A * A)
    return A_total / num_times

def get_crossover_average(num_times):
    # E[x_11^2 * x_12^2]
    total_value = 0
    for i in range(num_times):
        x = get_orthogonal_coordinate()
        current_value = (x[0, 0]**2) * (x[0, 1]**2)
        total_value += current_value
    return total_value / num_times

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_times', type=int, default=100)
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--epsilon', type=float, default=0.05)
    parser.add_argument('--verify_A_2', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--get_crossover', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--w_hat_estimate', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--coefficient_epsilon_2', type=bool, default=False, const=True, nargs='?')

    args = parser.parse_args()
    optimization.n = args.n
    optimization.k = args.k
    optimization.epsilon = args.epsilon
    if args.coefficient_epsilon_2:
        result = evaluate_coefficient_epsilon_2(args.sample_times)
    if args.w_hat_estimate:
        result = evaluate_w_hat_estimate(args.sample_times)
        print(result)
    elif args.get_crossover:
        result = get_crossover_average(args.sample_times)
        print(result)
    elif args.verify_A_2:
        result = get_matrix_average(args.sample_times)
        #print(result)
        print(np.average(np.diag(result)))
    else:
        result = get_average(args.sample_times)
        print(result / optimization.n)

