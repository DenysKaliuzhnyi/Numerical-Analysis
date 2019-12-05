import math
"""https://math.semestr.ru/optim/newton.php"""


def findRootDichotomy(func, a, b, eps, prnt=False):
    m = (a + b) / 2
    i = 0
    while True:
        m = (a + b) / 2
        x = func(m)
        if prnt:
            print(f"[a, b] = [{a}, {b}]".ljust(37) + f"x{i} = {m}".ljust(20) + f"f(x{i}) = {x}".ljust(33) + f"(b - a)/2 = {(b - a) / 2}")
        if x * func(a) > 0:
            a = m
        else:
            b = m
        if (b - a) / 2 <= eps:
            break
        i += 1
    m = (a + b) / 2
    i += 1
    if prnt:
        print(f"[a, b] = [{a}, {b}]".ljust(37) + f"x{i} = {m}".ljust(20) + f"f(x{i}) = {x}".ljust(33) + f"(b - a)/2 = {(b - a) / 2}")
    return m, i


def findRootNewton(func, df, x0, eps, prnt=False):
    if prnt:
        print(f"x0 = {x0}")
    i = 1
    while True:
        xn = x0 - func(x0)/df(x0)
        if prnt:
            print(f"x{i} = {xn}".ljust(28) + f"f(x{i}) = {func(xn)}".ljust(33) + f"|x{i} - x{i-1}| = {abs(xn - x0)}")
        if abs(xn - x0) <= eps:
            break
        i += 1
        x0 = xn
    return xn, i


if __name__ == '__main__':
    func = lambda x: x**2 * math.log10(x) - 1
    df = lambda x: (x + 2*x*math.log(x))/math.log(10)
    ddf = lambda x: (2*math.log(x) + 3)/math.log(10)
    eps = 10**-3
    a, b = 1, 2
    print("""
x^2 * lg(x) - 1 = 0
eps = 0.001
Проаналізуємо функцію:
lg(x) = 1 / x^2
x = 10^(1 / x^2)  (див. графік.png)""")
    print("f(1) =", func(1))
    print("f(2) =", func(2))
    print("""Функція неперервна та монотонно зростає на відрізку [1, 2],
f(1)*f(2) < 0 тоді за теоремою про корінь неперервної функції
існує єдиний x0 з відрізка [1, 2], що f(x0) = 0 \n 
Починаємо ітераційний процес методом дихотомії:""")

    root, cnt = findRootDichotomy(func, a, b, eps, prnt=True)
    print(f"Отже, з точністю eps = {eps} маємо x = {root} та апостеріорну оцінку n = {cnt}\n"
          f"Апріорна оцінка являє собою  n = [log2((b - a)/eps)] + 1 = "
          f"[log2(({b} - {a})/{eps})] + 1 = {int(math.log2((b-a)/eps)) + 1}")

    a, b = 1.5, 2
    print(f"""\n\nДля методу Ньютона додатково дослідимо функцію f.
f'(x) = (x + 2*x*ln(x))/ln(10)
f''(x) = (2*ln(x) + 3)/ln(10)
Якщо для даного методу взяти такий самий інтервал [1, 2],
то виявиться, що f(1)*f''(1) = {func(1)*ddf(1)} < 0 
Тому x0 = 2, a звідси q = {ddf(2)/(2*df(1))} > 1, що погано,
тому звузимо проміжок до [1.5, 2], бо дійнсно f(1.5) = {func(1.5)} < 0,
а також відомо f(2) > 0, тобто коріть лежить у [1.5, 2].
У такому випадку f(1.5)*f''(1.5) = {func(1.5)*ddf(1.5)} < 0,
тому знову x0 = 2, але тепер q = {ddf(2)/(4*df(1.5))} < 1,
(q = M2/(4*m1), m1 = f'(1.5) = {df(1.5)}, M2 = f''(2) = {ddf(2)})
тобто виконується достатня умова збіжності для методу Ньютона.\n
Тоді починаємо ітераційний процес:""")
    q = ddf(2)/(4*df(1.5))
    root, cnt = findRootNewton(func, df, x0=2, eps=eps, prnt=True)
    print(f"Отже, з точністю eps = {eps} маємо x = {root} та апостеріорну оцінку n = {cnt}\n"
          f"Апріорна оцінка являє собою  n = [log2(1 + ln(|x*-x0|/eps) / ln(1/q))] + 1 = "
          f"[log2(1 + ln(|1.8966510020402856-{2}|/{eps}) / ln(1/{q}))] + 1 "
          f"= {int(math.log2(1 + math.log(abs(2 - 1.8966510020402856)/eps) / math.log(1/q))) + 1}")
    input()





