import os
import argparse
import json
import numpy as np
from matplotlib import pyplot as plt
import optimization
import m_not_small
import verification_z_norm

# see https://en.wikipedia.org/wiki/Hermite_polynomials#Definition
# start with quadratic form
Hermite = [[-1, 0, 1],
    [0, -3, 0, 1],
    [3, 0, -6, 0, 1],
    [0, 15, 0, -10, 0, 1]]

LargestTerm = [[0, 0, 1],
    [0, 0, 0, 1],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1]]

FIXED_M_FILE = 'build/m_fixed.json'

def normalize(v):
    m = len(v) - 1
    N = m_not_small.construct_N(m, theoretical=True)
    v_const = v @ N @ v
    return v / np.sqrt(v_const)
    
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
        a[i] = np.power(optimization.n / np.sqrt(optimization.k), i) * p[i] / np.sqrt(optimization.n)

def get_empirical_result(polys, sample_times):
    result_list = []
    for p in polys:
        p_normalize = normalize(p)
        set_poly_coefficient(p_normalize)
        empirical_result = verification_z_norm.evaluate_coefficient_epsilon_2(sample_times)
        result_list.append(empirical_result)
    return result_list

def save_result(dic):
    with open(FIXED_M_FILE, 'w') as fp:
        json.dump(dic, fp)

def load_result():
    with open(FIXED_M_FILE, 'r') as fp:
        dic = json.load(fp)
    return dic

def add_theoretical(dic):
    theoretical = []
    common_factor = (1 - optimization.k / optimization.n)
    for i in range(len(Hermite)):
        theoretical.append(-1 * common_factor * (i + 1))
    dic['theoretical'] = theoretical

def plot_fixed_m(dic, save_fig=True, show=False):
    H = dic['Hermite']
    x_axis = [i + 2 for i in range(len(H))]
    plt.xlabel('degree of polynomials')
    plt.ylabel(r'$C[\xi]$').set_rotation(0)
    for k, v in dic.items():
        plt.plot(x_axis, v, label=k)
    plt.legend()
    if save_fig:
        plt.savefig('build/fixed_m.eps')
    if show:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_times', type=int, default=100)
    parser.add_argument('--n', type=int, default=180)
    parser.add_argument('--k', type=int, default=120)
    parser.add_argument('--mode', default='Hermite', choices=['Hermite', 'Theoretical', 'Largest', 'All'])
    parser.add_argument('--task', default='computation', choices=['plot', 'computation'])
    parser.add_argument('--show', default=False, type=bool, nargs='?', const=True)
    args = parser.parse_args()
    optimization.n = args.n
    optimization.k = args.k
    if os.path.exists(FIXED_M_FILE):
        dic = load_result()
    else:
        dic = {}
    if args.task == 'computation':
        if args.mode == 'Hermite' or args.mode == 'All':
            empirical_result_list = get_empirical_result(Hermite, args.sample_times)
            dic['Hermite'] = empirical_result_list
        if args.mode == 'Theoretical' or args.mode == 'All':
            add_theoretical(dic)
        if args.mode == 'Largest' or args.mode == 'All':
            empirical_result_list = get_empirical_result(LargestTerm, args.sample_times)
            dic['Largest'] = empirical_result_list
        save_result(dic)
    elif args.task == 'plot':
        plot_fixed_m(dic, show=args.show)