import argparse
import numpy as np
import optimization
import m_not_small
import verification_z_norm
a = None
def activate(z):
    return verification_z_norm.poly(z, a)
    
def derivative_activate(z):
    return verification_z_norm.derivative_poly(z, a)

def normalize(v):
    m = len(v) - 1
    N = m_not_small.construct_N(m, theoretical=True)
    v_const = v @ N @ v
    return v / np.sqrt(v_const)

verification_z_norm.activate = activate
verification_z_norm.derivative_activate = derivative_activate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_times', type=int, default=100)
    parser.add_argument('--n', type=int, default=180)
    parser.add_argument('--k', type=int, default=120)
    parser.add_argument('--p', type=int, nargs='+', default=[-1, 0, 1])
    args = parser.parse_args()
    optimization.n = args.n
    optimization.k = args.k
    p = normalize(np.array(args.p))
    a = np.zeros(len(p))
    for i in range(len(p)):
        a[i] = np.power(optimization.n / np.sqrt(optimization.k), i) * p[i] / np.sqrt(optimization.n)

    empirical_result = verification_z_norm.evaluate_coefficient_epsilon_2(args.sample_times)
    theoretical_result = verification_z_norm.get_coeff_epsilon_2_theoretical(a)
    print(empirical_result, theoretical_result)