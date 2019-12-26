import argparse
import numpy as np

def coefficient(n, k):
    nplus2 = n + 2
    nplus4 = n + 4
    a_numerator = k * n + 3 * n + k
    a_denominator = nplus2 ** 2
    a = a_numerator / a_denominator
    c_numerator = k * n + 3 * k + n + 2
    c_denominator = -1 * nplus2 * nplus4
    c = c_numerator / c_denominator
    b_numerator = 2 * (n + k + 6)
    b_denominator = -1 * n * nplus4
    b = b_numerator / b_denominator
    return np.min(np.roots([c, b, a]))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=10, type=int)
    parser.add_argument('--k', default=8, type=int)
    args = parser.parse_args()
    print(coefficient(args.n, args.k))