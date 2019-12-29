import numpy as np
import argparse
import optimization
from optimization import get_orthogonal_coordinate, get_spherical_coordinate
# verify the norm = 1 constraint: python3 verification_z_norm.py --n 180 --k 120
# consistent with commit ebf3c05 of  https://gitee.com/freewind201301/non-linear-activation-function
activate = None
derivative_activate = None

def generate_z_instance(x, y):
    A = x @ x.T
    return A @ y

def poly2(z):
    r = optimization.k * 1.0 / optimization.n
    return (optimization.n * z * z / r - 1) / np.sqrt(2 * optimization.n)

def derivative_poly2(z):
    r = optimization.k * 1.0 / optimization.n
    diagonal_terms = np.sqrt(2) * np.sqrt(optimization.n) * z / r
    return np.diag(diagonal_terms)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def derivative_sigmoid(z):
    return np.exp(z) / ((1 + np.exp(z)) ** 2)

def sigma(z):
    return z + optimization.epsilon * activate(z)

def get_w_hat(x, y):
    z = generate_z_instance(x, y)
    w_hat = x.T @ (derivative_activate(z) @ (y-z) - activate(z))
    return w_hat

def get_w_estimate(x, y):
    w_bar = x.T @ y
    w = w_bar + optimization.epsilon * get_w_hat(x, y)
    return w 

def get_coeff_epsilon_2(x, y, w_hat):
    z = generate_z_instance(x, y)
    part_1 = np.linalg.norm(x @ w_hat + activate(z)) ** 2
    part_2 = derivative_activate(z) @ x @ w_hat
    part_2 = part_2.T @ (y - z)
    return part_1 - 2 * part_2

def get_coeff_epsilon_2_theoretical(x, y):
    z = generate_z_instance(x, y)
    xi_z = activate(z)
    I_1 = np.linalg.norm(xi_z) ** 2
    I_2 = np.linalg.norm(x.T @ xi_z) ** 2
    # I_3 = 

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
        xi_z = activate(z)
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
    parser.add_argument('--w_hat_estimate', type=bool, default=False, const=True, nargs='?', help="non linear mse")
    parser.add_argument('--coefficient_epsilon_2', type=bool, default=False, const=True, nargs='?')
    parser.add_argument('--activate', default='poly2')
    np.random.seed(0)
    args = parser.parse_args()
    exec('activate = ' + args.activate)
    exec('derivative_activate = derivative_' + args.activate)

    optimization.n = args.n
    optimization.k = args.k
    optimization.epsilon = args.epsilon
    if args.coefficient_epsilon_2:
        result = evaluate_coefficient_epsilon_2(args.sample_times)
        print(result)
    elif args.w_hat_estimate:
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
        print(result)

