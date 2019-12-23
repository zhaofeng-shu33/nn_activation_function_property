# Usages: python m_not_small.py --n 240 --k 160 # -0.97
# the result should be near -1
import numpy as np
import argparse
k = 40
n = 60
def compute_N(i, j):
    if (i + j) % 2 == 1:
        return 0
    t = int((i + j) / 2)
    log_result = 0
    for s in range(0, t):
        numerator = (2 * s + k) * (2 * s + 1)
        denominator = (2 * s + n) * n
        log_result += np.log(numerator / denominator)
    result = np.exp(log_result) * n
    return result

def compute_M_without_r(i, j):
    if (i + j) % 2 == 1:
        return 0
    t = int((i + j) / 2)
    r = k / n
    if t == 0:
        return n
    elif i == 1 or j == 1:
        return 0
    else:
        result = 1
        for s in range(0, t):
            numerator = (2 * s + k) * (2 * s - 1)
            denominator = (2 * s + n + 2) * n
            result *= (numerator / denominator)
        result *= n * (i - 1) * (j - 1)
    return result

def construct_N(m):
    a = np.zeros([m + 1, m + 1])
    for i in range(m + 1):
        for j in range(m + 1):
            if j < i:
                a[i, j] = a[j, i]
            else:
                a[i, j] = compute_N(i, j)
    return a

def construct_M_without_r(m):
    a = np.zeros([m + 1, m + 1])
    for i in range(m + 1):
        for j in range(m + 1):
            if j < i:
                a[i, j] = a[j, i]
            else:
                a[i, j] = compute_M_without_r(i, j)
    return a

def get_minimum(m):
    N = construct_N(m)
    M = construct_M_without_r(m)
    return compute_result(M, N)

def compute_result(M, N):
    U = np.linalg.cholesky(N).T
    U_inv = np.linalg.inv(U)
    M_trans = U_inv.T @ M @ U_inv
    eig_val, _ = np.linalg.eig(M_trans)
    return np.min(eig_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=60)
    parser.add_argument('--k', type=int, default=40)
    args = parser.parse_args()
    n = args.n
    k = args.k    
    print(get_minimum(2)) # -0.92
    
