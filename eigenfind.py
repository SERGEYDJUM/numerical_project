from numpy.typing import NDArray
from numpy.random import randn
from numpy.linalg import norm
from math import isclose
from cmath import sqrt as csqrt
import numpy as np

from utils import hessenberg_transform, validate_matrix, gauss_jordan, qr_decomposition

Matrix = NDArray[np.float_]
Vector = NDArray[np.float_]


def max_eigen_pair(
    A: Matrix, max_iter: int = 512, eps: float = 1e-12, deterministic: bool = False
) -> (float, Vector):
    """Находит наибольшее по модулю собственное значение и соответствующий собственный вектор \
        с помощью метода степенных итераций.
    
    Примечание: не для всех матриц данный метод сойдётся к собственному вектору. \
        Так, для комплексных собственных чисел вектор будет вращаться, а для \
            кратных - может сойтись к корневому вектору.
        
    Args:
        A (NDArray): Квадратная матрица.

        max_iter (int): Максимальное количество итераций алгоритма.

        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.
        
        deterministic (bool): Включает заполнение начальных значений единицами, вместо случайных чисел.

    Returns:
        (float, NDArray): Собственное значение и вектор.
    """

    validate_matrix(A)
    n = A.shape[0]
    X = np.ones(n) if deterministic else randn(n)
    X = X / norm(X)
    lam, lam_new = float("-inf"), float("inf")
    for _ in range(max_iter):
        X_new = A @ X
        lam, lam_new = lam_new, (X @ X_new) / (X @ X)
        X = X_new / norm(X_new)

        if abs(lam - lam_new) < eps:
            break

    return (lam_new, X)


def closest_eigen_pair(
    A: Matrix,
    approx_eigen: float = 0.0,
    max_iter: int = 512,
    eps: float = 1e-9,
    deterministic: bool = False,
) -> (float, Vector):
    """Находит самое близкое к approx_eigen собственное значение и \
        соответствующий собственный вектор c помощью метода итераций Рэлея.

    Args:
        A (NDArray): Квадратная матрица.

        max_iter (int): Максимальное количество итераций алгоритма.

        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.
        
        deterministic (bool): Включает заполнение начальных значений единицами, вместо случайных чисел.

    Returns:
        (float, NDArray): Собственное значение и вектор.
    """

    validate_matrix(A)

    n = A.shape[0]
    X = np.ones(n) if deterministic else randn(n)
    X /= norm(X)
    Eye = np.eye(n)

    lam, old_lam = approx_eigen, 0
    for _ in range(max_iter):
        try:
            X = gauss_jordan(A - lam * Eye, X)
        except ValueError:
            return lam, X

        X /= norm(X)

        old_lam, lam = lam, X @ A @ X
        if abs(old_lam - lam) < eps:
            break

    return lam, X


def min_eigen_pair(
    A: Matrix, max_iter: int = 512, eps: float = 1e-12
) -> (float, Vector):
    # TODO: Make so that it actually finds smallest
    """Находит минимальное по модулю собственное значение и \
        соответствующий собственный вектор c помощью метода итераций Рэлея.

    Args:
        A (NDArray): Квадратная матрица.

        max_iter (int): Максимальное количество итераций алгоритма.

        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.

    Returns:
        (float, NDArray): Собственное значение и вектор.
    """

    def guess():
        return closest_eigen_pair(A, 0, max_iter, eps, deterministic=False)

    guesses = [guess(), guess(), guess()]

    return sorted(guesses, key=lambda x: abs(x[0]))[0]


def eigen_pairs_symmetric(
    A: Matrix, max_iter: int = 512, eps: float = 1e-12
) -> (Vector, Matrix):
    """Находит все собственные значения матрицы и соответствующие собственные векторы с \
        помощью метода поворотов. С несимметричными матрицами корректность не гарантирована.

    Примечание: из-за повышенных требований к точности, \
        результаты для матриц высокого ранга будут неверными.

    Args:
        A (NDArray): Квадратная матрица.

        max_iter (int): Максимальное количество итераций алгоритма.

        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.

    Returns:
        (NDArray, NDArray): Массив собственных значений и матрица с собственными векторами по рядам.
    """

    validate_matrix(A)
    n = A.shape[0]
    H_res = np.identity(n)
    for _ in range(max_iter):
        small = True
        aa = abs(A[0, -1])
        i_a, j_a = 0, -1

        for i in range(n):
            for j in range(n):
                if i != j:
                    el_abs = abs(A[i, j])

                    if el_abs > eps:
                        small = False

                    if i < j and el_abs > aa:
                        aa = el_abs
                        i_a, j_a = i, j

        if small:
            break

        phi = np.arctan(2 * A[i_a, j_a] / (A[i_a, i_a] - A[j_a, j_a])) / 2

        H = np.identity(n)
        H[i_a, i_a] = np.cos(phi)
        H[j_a, j_a] = H[i_a, i_a]
        H[i_a, j_a] = -np.sin(phi)
        H[j_a, i_a] = -H[i_a, j_a]
        H_res = H_res @ H

        A = np.transpose(H) @ (A @ H)

    return np.diag(A), np.transpose(H_res)


def eigen_values(
    A: Matrix, max_iter: int = 1024
) -> Vector:
    """Вычисляет комплексные собственные значения квадратной матрицы итеративно \
        с помощью QR-разложения.

        Примечание: низкая точность, медленная сходимость.
    Args:
        A (NDArray): Матрица.
        
        max_iter (int, optional): Ограничение по итерациям алгоритма.
        
    Returns:
        (NDArray): Массив собственных значений.
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")


    n = A.shape[0]
    A = hessenberg_transform(A)
    # Q_k = np.eye(n)
    for _ in range(max_iter):
        # shift = np.eye(n) * (A[-1, -1] * 0.99)  # Limit shift
        # Q, R = qr_decomposition(A - shift)
        Q, R = qr_decomposition(A)
        # A = R @ Q + shift
        A = R @ Q
        # Q_k = Q_k @ Q

        # if isclose(norm(np.tril(A, k=-1)), 0):
        #     break

    Lams = np.zeros(n, dtype=complex)

    skip = False
    for i in range(n - 1):
        if skip:
            skip = False
            continue

        if np.isclose(A[i + 1, i], 0):
            Lams[i] = A[i, i]
        else:
            extralams = rank_two_eigen_pairs(A[i : i + 2, i : i + 2])[0]
            Lams[[i, i + 1]] = extralams[[0, 1]]
            skip = True

    if not skip:
        Lams[-1] = A[-1, -1]
    return Lams


def rank_two_eigen_pairs(A: Matrix) -> (Vector, Matrix):
    """Находит все собственные значения матрицы 2x2 и соответствующие собственные векторы напрямую.

    Примечание: поддерживаются только невырожденные матрицы с действительными собственными значениями.

    Args:
        A (NDArray): Квадратная матрица.

    Returns:
        (NDArray, NDArray): Массив собственных значений и матрица с собственными векторами по рядам.
    """
    if A.shape[0] != 2 or A.shape[1] != 2:
        raise ValueError("Matrix with incorrect dims passed")

    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]

    det = a * d - b * c
    if isclose(det, 0.0):
        raise ValueError("Non-singular matricies are not supported")

    dis = (a + d) ** 2 - 4 * det

    vecs = np.zeros((2, 2), dtype=complex)
    lams = np.zeros(2, dtype=complex)

    for i, sign in enumerate((-1, 1)):
        lams[i] = (a + d + sign * csqrt(complex(dis))) / 2

    if np.isclose(d, 0) or np.isclose(b, 0):
        vecs[0] = np.array([lams[0], 0])
        vecs[1] = np.array([0, lams[1]])
    else:
        vecs[0] = np.array([b, lams[0] - a])
        vecs[1] = np.array([lams[1] - d, c])

        for i in range(2):
            vin = norm(vecs[i])
            if not isclose(vin, 0):
                vecs[i] /= vin

    return lams, vecs
