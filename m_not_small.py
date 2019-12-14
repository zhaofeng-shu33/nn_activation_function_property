import numpy as np
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
    U = np.linalg.cholesky(N).T
    U_inv = np.linalg.inv(U)
    M_trans = U_inv.T @ M @ U_inv
    eig_val, _ = np.linalg.eig(M_trans)
    return np.min(eig_val)

if __name__ == '__main__':
    for m in range(2, 10):
        print(get_minimum(m))
