import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


######################
f = lambda x: 4.**x
nodes_cnt = 4
a, b = -1, 2
######################


def chebyshev_roots(n, a, b):
    f = lambda k: (a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) / (2 * n) * np.pi)
    return np.fromfunction(f, shape=(n, ))


def rnd(nums):
    return np.round(nums, 3)


def build(real, polynom, a, b):
    xs = np.linspace(a, b, (b - a) * 100)
    ys_r = np.vectorize(real)(xs)
    ys_p = np.polyval(polynom, xs)
    fig, ax = plt.subplots()
    ax.plot(xs, ys_r, label="real")
    ax.plot(xs, ys_p, label="interpolation")
    ax.set_title(f"count of nodes is {nodes_cnt}")
    ax.legend()
    plt.show()


def pretty(polynom):
    return " + ".join(
        map(
            lambda x: f"{rnd(x[1])}x{polynom.size - x[0] - 1}",
            enumerate(polynom[:-1]))
    ) + f" + {rnd(polynom[-1])}"


def derivative(n, func):
    xs = np.linspace(a, b, (b - a) * 10000)
    ys = np.vectorize(func)(xs)
    res = np.diff(ys) / np.diff(xs)
    for k in range(1, n):
        res = np.diff(res) / np.diff(xs)[:-k]
    return res


def error(n, func, a, b):
    return derivative(n, func).max() / np.math.factorial(n) * ((b - a)**n) / (2**(2*n - 1))


nodes = chebyshev_roots(nodes_cnt, a, b)
f_nodes = np.vectorize(f)(nodes)
data = pd.DataFrame(data={'k': np.arange(nodes_cnt),
                          'x': nodes,
                          'f(x)': f_nodes}).set_index('k')

coef = np.zeros(nodes_cnt)
polynoms = np.zeros(nodes_cnt, dtype=object)
for k in range(nodes_cnt):
    coef[k] = data['f(x)'][k] / np.prod([data['x'][k] - data['x'][j] for j in range(nodes_cnt) if j != k])
    for j in range(nodes_cnt):
        if j != k:
            if not np.any(polynoms[k]):
                polynoms[k] = np.array([1, -data['x'][j]])
            else:
                polynoms[k] = np.convolve([1, -data['x'][j]], polynoms[k])
polynoms = np.stack(polynoms, axis=0)

L = np.zeros(nodes_cnt)
for k in range(nodes_cnt):
    L[k] = np.sum(polynoms[:, k] * coef)

print(rnd(data))
print()
print(f"L{nodes_cnt - 1} = {pretty(L)}")
# print()
# print(f"max error is {rnd(error(nodes_cnt, f, a, b))}")

build(f, L, a, b)


