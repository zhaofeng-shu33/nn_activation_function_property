import argparse
import numpy as np
import optimization
import m_not_small
import verification_z_norm

# see https://en.wikipedia.org/wiki/Hermite_polynomials#Definition
Hermite = [[-1, 0, 1],
    [0, -3, 0, 1],
    [3, 0, -6, 0, 1],
    [0, 15, 0, -10, 0, 1]]
    
a = None
def activate(z):
    return verification_z_norm.poly(z, a)
    
def derivative_activate(z):
    return verification_z_norm.derivative_poly(z, a)

verification_z_norm.activate = activate
verification_z_norm.derivative_activate = derivative_activate
    
def set_poly_coefficient(p):
    global a
    a = np.zeros(len(p))
    for i in range(len(p)):
        a[i] = np.power(optimization.n, -0.5 + i) * p[i] / np.power(optimization.k, i/2)

def get_empirical_result(sample_times):
    result_list = []
    for p in Hermite:
        set_poly_coefficient(p)
        empirical_result = verification_z_norm.evaluate_coefficient_epsilon_2(sample_times)
        result_list.append(empirical_result)
    return result_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_times', type=int, default=100)
    parser.add_argument('--n', type=int, default=180)
    parser.add_argument('--k', type=int, default=120)
    args = parser.parse_args()
    empirical_result_list = get_empirical_result(args.sample_times)
    print(empirical_result_list)