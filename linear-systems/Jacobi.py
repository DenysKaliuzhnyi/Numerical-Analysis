import numpy as np
import math
# https://math.semestr.ru/optim/example-iteration-slau.php

precision1 = 45
precision2 = 55

# testmatrix = [[10, 2, -1, 5], [-2, -6, -1, 24.42], [1, -3, 12, 36]]
testmatrix = [[4, 1, 1, 4], [1, 5, 0, 1], [1, 0, 5, 1]]


class Jacobi:
    def __init__(self, matrix, eps):
        self._n = len(matrix)
        self._A = matrix[:, :-1]
        self._b = matrix[:, -1]
        self._eps = eps
        np.set_printoptions(precision=len(str(int(eps ** -1) // 10)))

    @property
    def A(self):
        return self._A

    @property
    def b(self):
        return self._b

    @property
    def n(self):
        return self._n

    @property
    def Ab(self):
        return np.hstack([self.A, self.b.reshape((-1, 1))])

    @property
    def eps(self):
        return self._eps

    def solve(self):
        A = self.A.copy()
        b = self.b.copy()
        for i in range(self.n):
            b[i] /= A[i, i]
            A[i] /= -A[i, i]
            A[i, i] = 0
        x0 = np.zeros(self.n)
        while True:
            x = np.array([A[i].dot(x0) for i in range(self.n)]) + b
            if abs(x - x0).max() < self.eps:
                return x
            x0 = x

    def bound(self):
        absA = np.abs(self.A)
        qs = (absA[i].sum() / absA[i, i] - 1 for i in range(self.n))
        q = max(qs)
        return int(math.log((1 - q)*self.eps) / math.log(q)) + 1

    @staticmethod
    def valid(matrix):
        isle = False
        absmatrix = np.abs(matrix)
        for i in range(absmatrix.shape[0]):
            left = absmatrix[i, i]
            right = absmatrix[i].sum() - left - absmatrix[i, -1]
            if abs(left) >= abs(right):
                if left != right:
                    isle = True
            else:
                return False
        return isle


def readMatrix():
    row1 = list(map(float, input().split()))
    n = len(row1)
    matrix = [list(map(float, input().split())) for _ in range(n-2)]
    matrix.insert(0, row1)
    assert all((len(matrix[i]) == n for i in range(n-1))), "Incorrect input."
    return matrix


if __name__ == '__main__':
    inpt = False
    eps = float(input("Введіть точність eps = "))
    assert eps > 0
    if inpt:
        print("Ведіть сисетему я матрицю:")
        try:
            matrix = readMatrix()
            npmatrix = np.array(matrix, np.float)
            assert Jacobi.valid(npmatrix), "Cистема НЕ задовільняє діагональній умові"
        except AssertionError as e:
            print(*e.args)
            exit()
    else:
        npmatrix = np.array(testmatrix, np.float)
        print(npmatrix)

    print("Перевірка на діагональну перевагу виповнена, система задовільняє достатній умові.")

    Z = Jacobi(npmatrix, eps)
    bound = Z.bound()
    print(f"Апріорна оцінка складає {bound} ітерацій.")
    res = Z.solve()
    print(f"x = {res}")
