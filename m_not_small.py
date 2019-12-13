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

def compute_M(i, j):
    if (i + j) % 2 == 1:
        return 0
    t = int((i + j) / 2)
    r = k / n
    if t == 0:
        return n * (1 - r)
    elif t == 1:
        return 0
    else:
        result = 1
        for s in range(0, t):
            numerator = (2 * s + k) * (2 * s - 1)
            denominator = (2 * s + n + 2) * n
            result *= (numerator / denominator)
        result *= n * (1 - r) * (i - 1) * (j - 1)
    return result

if __name__ == '__main__':
    print(compute_M(13, 14))
    print(compute_M(4, 16))
