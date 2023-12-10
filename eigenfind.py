from numpy.typing import NDArray
from numpy.linalg import norm
import numpy as np

from utils import validate_matrix, lu_decomposition, lu_solve, partial_pivot_matrix

Matrix = NDArray[np.float_]
Vector = NDArray[np.float_]


def max_eigen_pair(
    A: Matrix, max_iter: int = 512, eps: float = 1e-9
) -> (float, Vector):
    """Находит наибольшее по модулю собственное значение и соответствующий собственный вектор.

    Args:
        A (NDArray): Квадратная матрица.

        max_iter (int): Максимальное количество итераций алгоритма.

        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.

    Returns:
        (float, NDArray): Собственное значение и вектор.
    """

    validate_matrix(A)
    n = A.shape[0]
    X = np.ones(n)
    X_old, X = X / norm(X), X
    lam_old, lam = 0.0, 0.0

    for _ in range(max_iter):
        X_old, X = X, A @ X

        max_idx = np.argmax(np.abs(X_old))
        lam_old, lam = lam, X[max_idx] / X_old[max_idx]

        X /= norm(X)
        if abs(lam_old - lam) < eps:
            break

    return (lam, X)


def closest_eigen_pair(
    A: Matrix, approx_eigen: float = 0.0, max_iter: int = 512, eps: float = 1e-9
) -> (float, Vector):
    """Находит самое близкое к модулю approx_eigen собственное значение и соответствующий собственный вектор.

    Args:
        A (NDArray): Квадратная матрица.

        max_iter (int): Максимальное количество итераций алгоритма.

        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.

    Returns:
        (float, NDArray): Собственное значение и вектор.
    """

    validate_matrix(A)
    A = abs(approx_eigen) * np.identity(A.shape[0]) - A  # Check for zeros on diag

    X = np.ones(A.shape[0])
    X_old, X = X, X / norm(X)

    P = partial_pivot_matrix(A)
    LU = lu_decomposition(A, P)

    lam_old, lam = 1, 2
    for _ in range(max_iter):
        X, X_old = lu_solve(LU, P, X), X
        X /= norm(X)

        max_idx = np.argmax(np.abs(X))
        lam_old, lam = lam, X[max_idx] / X_old[max_idx]

        if abs(lam_old - lam) < eps:
            break

    return (lam, X)  # TODO: Fix sign


def min_eigen_pair(
    A: Matrix, max_iter: int = 512, eps: float = 1e-9
) -> (float, Vector):
    """Находит наименьшее по модулю собственное значение и соответствующий собственный вектор.

    Args:
        A (NDArray): Квадратная матрица.

        max_iter (int): Максимальное количество итераций алгоритма.

        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.

    Returns:
        (float, NDArray): Собственное значение и вектор.
    """

    return closest_eigen_pair(A, max_iter=max_iter, eps=eps)


def eigen_pairs(A: Matrix, max_iter: int = 512, eps: float = 1e-9) -> (Vector, Matrix):
    """Находит все собственные значения и соответствующие собственные векторы.

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
