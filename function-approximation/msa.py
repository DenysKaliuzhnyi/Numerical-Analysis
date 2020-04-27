"""Побудуємо НСК табульованої функції F на проміжку [a, b] за допомогою многочленів Лежандра"""
from functools import lru_cache
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


@lru_cache(3)
def legendre_basis(n):
    """створюємо n-тий базисний вектор поліному Лежандра"""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x
    return ((2*n - 1)*x*legendre_basis(n-1) - (n - 1)*legendre_basis(n-2))/n


"""задаємо кількість точок для табуляції, кількість базисних функцій, відрізок апроксимації та саму функцію"""
n_points = 100
n_basis = 4
a, b = -1, 1
f = lambda x: 1/(1 + np.exp(-5*x))


"""задаємо табульовану таблицю, пам'ять для матриці Грамма та fk"""
x = np.linspace(a, b, n_points)
y = f(x)
G = np.zeros((n_basis, n_basis))
fk = np.zeros(n_basis)


for i in range(n_basis):
    fk[i] = (legendre_basis(i) * y).sum()

for i in range(n_basis):
    for j in range(i, n_basis):
        dot = (legendre_basis(i) * legendre_basis(j)).sum()
        G[i, j] = dot
        G[j, i] = dot

coef = np.linalg.solve(G, fk)

"""подивимось на знайдені коефіцієнти при базисних функція"""
print(*zip(coef.round(2), [f"L{i}" for i in range(n_basis)]))

"""порахуємо за допомогою апроксимації за базисом Лежандра оцінку функції f"""
PHI = np.array([legendre_basis(i)*coef[i] for i in range(n_basis)]).sum(axis=0)

"""знайдемо відхилення"""
bias = ((y - PHI)**2).sum()

"""побудуємо график табульованої та апроксимуючої функції"""
fig, ax = plt.subplots()
ax.axis('equal')
ax.plot(x, y, label='real function')
ax.plot(x, PHI, lS='-.', label=f'Legendre approximation with {n_basis} basis vectors')
ax.set_title("$f(x) = \\dfrac{1}{1+e^{-5x}}$" + f", bias = {bias.round(4)}")
ax.legend()

plt.show()
