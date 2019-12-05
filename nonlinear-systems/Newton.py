import numpy as np
from itertools import count
np.set_printoptions(precision=4)


def apply(F, x):
    n = F.shape[0]
    if len(F.shape) == 2:
        res = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(n):
                res[i, j] = F[i, j](*x)
    else:
        res = np.zeros(shape=n)
        for i in range(n):
            res[i] = F[i](*x)
    return res


def solve():
    eps = 0.00001
    F = np.array([
        lambda x, y: x - 0.5 * np.sin((x - y) / 2),
        lambda x, y: y - 0.5 * np.cos((x + y) / 2)
    ], dtype=object)

    dF = np.array([
        [lambda x, y: 1 - 0.25 * np.cos((x - y) / 2), lambda x, y:     0.25 * np.cos((x - y) / 2)],
        [lambda x, y:     0.25 * np.sin((x + y) / 2), lambda x, y: 1 - 0.25 * np.sin((x + y) / 2)]
    ], dtype=object)

    x0 = (0, 0)
    print(f"x0 = {x0}\n")
    for i in count():
        A = apply(dF, x0)
        F0 = apply(F, x0)
        z = np.linalg.solve(A, F0)
        x = np.array(x0) - np.array(z)
        print(f"{A} = A{i} \t\t F(x{i}) = {F0} \t\t z{i} = {z} \t\t x{i + 1} = {x}\n")
        if np.abs(z).max() <= eps:
            return x
        x0 = x


if __name__ == '__main__':
    res = solve()
