import numpy as np
from itertools import count
np.set_printoptions(precision=4)


def solve(A, eps):
    n = A.shape[0]
    x0 = np.array([0, 0, 1])

    x = np.dot(A, x0)
    # x /= np.max(np.abs(x))
    L0 = (x * x0).sum() / (x0 * x0).sum()
    print(f"x0 = {x0 / np.linalg.norm(x0)}")
    print(f"x1 = {x} \t\t L1 = {L0}")
    x0 = x
    for i in count(2):
        # xnorm = x
        xnorm = x0 / np.linalg.norm(x0)
        x = np.dot(A, xnorm)
        # x /= np.max(np.abs(x))
        L = (x * xnorm).sum() / (xnorm * xnorm).sum()
        print(f"x{i} = {x} \t\t L{i} = {L}")
        if np.abs(L - L0) <= eps:
            return L
        x0 = x
        L0 = L


if __name__ == '__main__':
    eps = 10**-3
    A = np.array([
        [5, 1, 2],
        [1, 4, 1],
        [2, 1, 3]
    ])
    print(f"{A} = A")
    Lmax = solve(A, eps)
    print(f"Lmax = {Lmax}\n")
    B = Lmax*np.identity(A.shape[0]) - A
    print(f"{B} = B")
    Lmin = Lmax - solve(B, eps)
    print(f"Lmin(A) = Lmax(A) - Lmax(B) = {Lmin}")
