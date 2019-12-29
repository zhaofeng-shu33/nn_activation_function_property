# Usages: python m_not_small.py --n 240 --k 160 # -0.97
# the result should be near -1
import argparse

import numpy as np
import scipy

k = 40
n = 60

def get_orthogonal_coordinate(n_, k_):
    z = scipy.randn(n_, n_) # n by n random matrix
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph)
    return q[:2,:k_]
    
def compute_A11(n_, k_, t): # E[A11^t]
    result = 1
    for i in range(0, t):
        result *= (k_ + 2 * i) / (n_ + 2 * i)
    return result

def compute_A12_simulation(n_, k_, t, num_of_iteration = 1000): # E[A12^t] using MC simulation
    result = 0
    for i in range(num_of_iteration):
        v2 = get_orthogonal_coordinate(n_, k_)
        value_base = v2[0,:] @ v2[1,:]
        result += value_base ** t
    return (result / num_of_iteration)

def compute_A12(n_, k_, t):
    if n_ != k_ + 1 and k_ != 1:
        raise NotImplementedError("not supported (n, k) pairs: (%d, %d)" %(n_, k_))
    if t % 2 == 1:
        return 0
    t_ = int(t / 2)
    result = 1 / (n_ - 2)
    for i in range(t_):
        numerator = 2 * i + 1
        denominator = n_ - 2 + 2 * i
        result *= (numerator / denominator)
    for i in range(t_):
        numerator = 2 * i + 1
        denominator = n_ - 2 + 2 * (i + t_)
        result *= (numerator / denominator)
    result /= (n_ - 2 + 4 * t_)
    return result

def double_factorial(n):
    if (n == 0 or n == 1 or n == -1): 
        return 1; 
    return n * double_factorial(n - 2)

def compute_N(i, j, new_api=True):
    if (i + j) % 2 == 1:
        return 0
    t = int((i + j) / 2)
    log_result = 0
    if 2 * t > n - k and (n - k) % 2 == 0 and new_api:
        for r in range(0, int((n-k) / 2)):
            numerator = k + 2 * r
            denominator = k + 2 * t + 2 * r
            log_result += np.log(numerator / denominator)
        for s in range(0, t):
            log_result += np.log((2 * s + 1) / n)
    else:
        for s in range(0, t):
            numerator = (2 * s + k) * (2 * s + 1)
            denominator = (2 * s + n) * n
            log_result += np.log(numerator / denominator)
    result = np.exp(log_result) * n
    return result

def compute_N_theoretical(i, j):
    if (i + j) % 2 == 1:
        return 0
    return double_factorial(i + j - 1)

def compute_M_without_r_theoretical(i, j):
    if (i + j) % 2 == 1:
        return 0
    if i + j == 0:
        return 1
    elif i == 1 or j == 1:
        return 0
    else:
        return (-1) * (i - 1) * (j - 1) * double_factorial(i + j - 3)

def compute_M_without_r(i, j, new_api=False):
    if (i + j) % 2 == 1:
        return 0
    t = int((i + j) / 2)
    if t == 0:
        return n
    elif i == 1 or j == 1:
        return 0
    else:
        log_result = 0
        if 2 * t > n - k + 2 and (n - k) % 2 == 0 and new_api:
            for r in range(0, int((n-k) / 2) + 1):
                numerator = k + 2 * r
                denominator = k + 2 * t + 2 * r
                log_result += np.log(numerator / denominator)
            for s in range(0, t):
                log_result += np.log(abs(2 * s - 1) / n)
        else:
            for s in range(0, t):
                numerator = (2 * s + k) * abs(2 * s - 1)
                denominator = (2 * s + n + 2) * n
                log_result += np.log(numerator / denominator)
        result = np.exp(log_result)
        result *= n * (i - 1) * (j - 1)
        if t >= 1:
            result = - result
    return result

def construct_N(m, theoretical=False):
    a = np.zeros([m + 1, m + 1])
    for i in range(m + 1):
        for j in range(m + 1):
            if j < i:
                a[i, j] = a[j, i]
            elif theoretical:
                a[i, j] = compute_N_theoretical(i, j)
            else:
                a[i, j] = compute_N(i, j)
    return a

def construct_M_without_r(m, theoretical=False):
    a = np.zeros([m + 1, m + 1])
    for i in range(m + 1):
        for j in range(m + 1):
            if j < i:
                a[i, j] = a[j, i]
            elif theoretical:
                a[i, j] = compute_M_without_r_theoretical(i, j)
            else:
                a[i, j] = compute_M_without_r(i, j)
    return a

def get_minimum(m, theoretical=False, filter_array=[], get_vector=False):
    N = construct_N(m, theoretical)
    M = construct_M_without_r(m, theoretical)
    if len(filter_array) > 0:
        M_f = M[:,filter_array][filter_array,:]
        N_f = N[:,filter_array][filter_array,:]
        return compute_result(M_f, N_f)
    return compute_result(M, N, get_vector)

def compute_result(M, N, get_vector=False):
    U = np.linalg.cholesky(N).T
    U_inv = np.linalg.inv(U)
    M_trans = U_inv.T @ M @ U_inv
    eig_val, eig_vector = np.linalg.eig(M_trans)
    if not get_vector:
        return np.min(eig_val)
    min_val = 1000
    for i in range(M.shape[0]):
        if eig_val[i] < min_val:
            min_val = eig_val[i]
            min_vector = eig_vector[:,i]
    return (min_val, min_vector)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--k', type=int, default=40)
    parser.add_argument('--m', type=int, default=2)
    parser.add_argument('--theoretical', default=False, type=bool,
        nargs='?', const=True)
    parser.add_argument('--filter', nargs='+', type=int, default=[])
    args = parser.parse_args()
    n = args.n
    k = args.k    
    print(get_minimum(args.m, args.theoretical, args.filter)) # -0.92
    
