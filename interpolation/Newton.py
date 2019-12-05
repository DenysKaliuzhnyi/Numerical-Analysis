import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


######################
f = lambda x: 3.**x
nodes_cnt = 3
a, b = -1, 1
######################


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


def chebyshev_roots(n, a, b):
    f = lambda k: (a + b) / 2 + (b - a) / 2 * np.cos((2 * k + 1) / (2 * n) * np.pi)
    return np.fromfunction(f, shape=(n, ))


def rnd(nums):
    return np.round(nums, 3)


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
data = pd.DataFrame(data={'x': nodes,
                          'f1': f_nodes})
for i in range(2, nodes_cnt + 1):
    data[f"f{i}"] = 0
    for j in range(nodes_cnt - i + 1):
        data.loc[j, f"f{i}"] = (data[f"f{i - 1}"][j] - data[f"f{i - 1}"][j + 1]) / (data['x'][j] - data['x'][j + 1 + i - 2])

polynoms = np.zeros(nodes_cnt, dtype=object)
polynoms[0] = np.ones(1)
for i in range(1, nodes_cnt):
    polynoms[i] = np.array([1, -data['x'][0]])
for i in range(2, nodes_cnt):
    for j in range(2, i + 1):
        polynoms[i] = np.convolve(polynoms[i], np.array([1, -data['x'][j - 1]]))

L = np.zeros(nodes_cnt)
for i in range(nodes_cnt):
    for j in range(i, nodes_cnt):
        L[-i-1] += data[f'f{j+1}'][0] * polynoms[j][-i-1]

print(data)
print()
print(f"L{nodes_cnt - 1} = {pretty(L)}")
# print()
# print(f"max error is {rnd(error(nodes_cnt, f, a, b))}")

build(f, L, a, b)
