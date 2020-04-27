"""Побудуємо НРН табульованої функції F на проміжку [a, b] 0, 1 та 2-го степенів"""
from functools import lru_cache
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


"""задаємо кількість точок для табуляції, кількість базисних функцій, відрізок апроксимації та саму функцію"""
n_points = 1000
a, b = 0, 1
f = lambda x: np.sqrt(x)

x = np.linspace(a, b, n_points)
y = f(x)


fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(6, 8))


# Q0 = (m + M)/2
# delta(Q0) = max|sqrt(x) - 0.5| = 0.5
Q0 = np.repeat(0.5*(np.max(y) + np.min(y)), n_points)
ax1.plot(x, y, label='$f(x) = \sqrt{x}$')
ax1.plot(x, Q0, label='$Q_0 = \dfrac{1}{2}$,  $\Delta(Q_0)=\dfrac{1}{2}$')
ax1.plot([a, b], [np.max(y), np.max(y)], '--', color='0', alpha=0.25)
ax1.plot([a, b], [np.min(y), np.min(y)], '--', color='0', alpha=0.25)
ax1.legend(loc='upper left')


# Q1 = x = 0.125
# функція опукла вниз, то точки a, b - точки чебешевського альтернансу

# f(a)  - c0 - c1*a  =  sign*delta(f)
# f(xi) - c0 - c1*xi = -sign*delta(f)
# f(b)  - c0 - c1*b  =  sign*delta(f)
# a = 0, b = 1

# phi(x) = f(x) - Q1(x)
# phi'(xi) = 0  =>  phi(a) = -b, phi(xi) = sqrt(xi) - a*xi - b, phi(b) = 1 - a - b
# phi(a) = -phi(xi) = phi(b)
# -b = -sqrt(xi) + a*xi + b
# -b = 1 - a - b
# then a = 1, b = 1/8, delta(Q1) = 1/8
Q1 = x + 0.125
ax2.plot(x, y, label='$f(x) = \sqrt{x}$')
ax2.plot(x, Q1, label='$Q_1 = x + \dfrac{1}{8}$,  $\Delta(Q_1)=\dfrac{1}{8}$')
ax2.plot([a, b], [f(a) + 0.125*2, f(b)+ 0.125*2], '--', color='0', alpha=0.25)
ax2.plot([a, b], [f(a), f(b)], '--', color='0', alpha=0.25)
ax2.legend(loc='upper left')


# Qn(x) = Pn+1(x) - An+1(x) * Tn+1(x)
# Tn+1(x) - нормаований багаточлен чебишова  на проміжку [a; b]
# xk = (b + a)/2 + (b - a)/2 * tk; tk = cos(((2k+1)*pi)/(2*(n+1))) - тча
f = lambda x: x**3 + 3*x**2 - 2*x + 5
x = np.linspace(a, b, n_points)
y = f(x)
n = 2

T3 = lambda x: (4*x**3 - 3*x)/2**n
tk = np.array([np.cos(((2*k+1)*np.pi)/(2*(n+1))) for k in range(n + 1)])
xk = (b + a)/2 + (b - a)/2 * tk
# Q2 = 5 - 5/4*x + 3*x^2
Q2 = y - T3(x)
delta = 1/2**n

ax3.plot(x, y, label='$f(x) = x^3 + 3x^2 - 2x + 5$')
ax3.plot(x, Q2, label='$Q_2 = 3x^2 - \dfrac{5}{4}x + 5$,  $\Delta(Q_2)=\dfrac{1}{4}$')
ax3.plot(x, Q2 + delta, '--', color='0', alpha=0.25)
ax3.plot(x, Q2 - delta, '--', color='0', alpha=0.25)
ax3.legend()

fig.tight_layout()
plt.show()
