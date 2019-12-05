import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(precision=3)
# http://www.scriru.com/14/94179136356.php

# testmatrix = [[1, 2, 3, 1], [2, 5, 5, 2], [3, 5, 6, 3]]
testmatrix = [[1, 0, -1, 2], [0, 4, 4, 12], [-1, 4, 14, 19]]
# 1 3 1
# 3 1 2


class SqrtRoots:
    def __init__(self, matrix):
        self._n = len(matrix)
        self._A = matrix[:, :-1]
        self._b = matrix[:, -1]

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

    def solve(self):
        D = np.zeros((self.n, self.n), np.int8)
        S = np.zeros((self.n, self.n), np.float)
        for i in range(self.n):
            p = self.A[i, i] - sum(S[k, i]**2 * D[k, k] for k in range(i))
            D[i, i] = p / abs(p)
            S[i, i] = np.sqrt(abs(p))
            for j in range(i + 1, self.n):
                S[i, j] = (self.A[i, j] - sum(S[k, i]*S[k, j]*D[k, k] for k in range(i)))/(D[i, i]*S[i, i])
        STD = S.transpose()
        for i in range(self.n):
            STD[i, i] *= D[i, i]
        Y = np.zeros(self.n, np.float)
        for i in range(self.n):
            Y[i] = (self.b[i] - sum(STD[i, k]*Y[k] for k in range(i))) / STD[i, i]
        X = np.zeros(self.n, np.float)
        for i in reversed(range(self.n)):
            X[i] = (Y[i] - sum(STD[i, k]*X[k] for k in range(i + 1, self.n))) / STD[i, i]
        return X

    @staticmethod
    def valid(matrix):
        return np.array_equal(matrix, matrix.transpose())


def readMatrix():
    row1 = list(map(float, input().split()))
    n = len(row1)
    matrix = [list(map(float, input().split())) for _ in range(n-2)]
    matrix.insert(0, row1)
    assert all((len(matrix[i]) == n for i in range(n-1))), "Incorrect input."
    return matrix


if __name__ == '__main__':
    inpt = True
    npmatrix = None
    if inpt:
        print("Ведіть сисетему я симетричну матрицю:")
        try:
            matrix = readMatrix()
            npmatrix = np.array(matrix, np.float)
            assert SqrtRoots.valid(npmatrix[:, :-1]), "Матриця не є симетричною."
        except AssertionError as e:
            print(*e.args)
            exit()
    else:
        npmatrix = np.array(testmatrix, np.float)
        print(f"A = \n {npmatrix}\n")

    print("Перевірка на симетричність виповнена, система задовільняє достатній умові.")
    Z = SqrtRoots(npmatrix)
    res = Z.solve()
    print(f"x = {res}")


# print(np.linalg.solve(Z.A, Z.b))

# np.random.seed(34)
# npmatrix = np.random.randint(0, 100, (10, 11)).astype(np.complex)
# print(f"A = \n {npmatrix.astype(np.float)}\n")
# for i in range(npmatrix.shape[0]):
#     for j in range(i + 1, npmatrix.shape[0]):
#         npmatrix[i, j] = npmatrix[j, i]