from numpy.typing import NDArray
import numpy as np
from numpy.linalg import norm
from math import isclose

Matrix = NDArray[np.float_]
Vector = NDArray[np.float_]


def determinant(A: Matrix) -> float:
    """Вычисляет определитель наибольшего минора матрицы (самой матрицы).

    Примечание:

    Args:
        A (NDArray): Матрица

    Returns:
        (float): Определитель матрицы.
    """

    A = np.copy(A)
    n = A.shape[0]
    det_A = 1.0
    swap_mul = 1
    for i in range(n):
        max_lead = i + np.argmax(np.abs(A[i:, i]))
        if max_lead != i:
            A[[max_lead, i]] = A[[i, max_lead]]
            swap_mul *= -1

        if isclose(A[i, i], 0):
            return 0.0

        det_A *= A[i, i]
        A[i] /= A[i, i]
        for j in range(i + 1, n):
            A[j] -= A[i] * A[j, i]

    return det_A * swap_mul


def matrix_is_singular(A: Matrix) -> bool:
    """Определяет, является ли матрица невырожденой.

    Args:
        A (NDArray): Матрица.

    Returns:
        (bool): False, если матрица вырождена.
    """

    return not isclose(abs(determinant(A)), 0, abs_tol=1e-3)


def validate_matrix(A: Matrix):
    """Выводит ошибку, если матрица вырождена или неквадратна.

    Args:
        A (NDArray): Матрица.
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    if not matrix_is_singular(A):
        raise ValueError("Matrix must have a determinant")


def partial_pivot_matrix(A: Matrix) -> Matrix:
    """Строит матрицу перестановок строк P, при которой B = P @ A имеет \
    наибольшую по модулю диагональ после приведения к ступенчатому виду.

    Args:
        A (NDArray): Исходная матрица.

    Returns:
        (NDArray): Матрица P.
    """

    n = A.shape[0]
    P = np.identity(n, dtype=np.float_)
    for i in range(n):
        max_idx = i + np.argmax(np.abs(A[i:n, i]))
        P[[i, max_idx]] = P[[max_idx, i]]
    return P


def gauss_jordan(A: Matrix, Y: Vector) -> Vector:
    """Решает систему уравнений A @ X = Y.

    Args:
        A (NDArray): Матрица.
        Y (NDArray): Столбец свободных значений.

    Returns:
        (NDArray): Решение системы уравнений.
    """

    n = A.shape[0]
    AY = np.column_stack((A, Y))

    for i in range(n):
        max_lead = i + np.argmax(np.abs(AY[i:, i]))
        if max_lead != i:
            AY[[max_lead, i]] = AY[[i, max_lead]]

        if isclose(AY[i, i], 0.0):
            raise ValueError("Solution doesn't exist: A is not singular")

        AY[i] /= AY[i, i]

        for j in range(i + 1, n):
            AY[j] -= AY[i] * AY[j, i]

    for i in range(n - 1)[::-1]:
        for j in range(i + 1, n):
            AY[i, -1] -= AY[j, -1] * AY[i, j]

    return AY[:, -1]


def householder_reflection(U: Vector) -> Matrix:
    """Строит значимую часть матрицы оператора \
        отражения от плоскости с нормалью U.

    Args:
        U (NDArray): Вектор, перпендикулярный плоскости.

    Returns:
        (NDArray): Нижняя правая часть матрицы отражения.
    """
    
    n = U.shape[0]
    P = np.eye(n)
    
    if np.isclose(norm(U), 0.0):
        return P
    
    v = U / (U[0] + np.copysign(norm(U), U[0]))
    v[0] = 1
    v = v[:, np.newaxis]
    P -= (v @ v.T) / (v.T @ v) * 2
    return P


def hessenberg_transform(A: Matrix) -> Matrix:
    """Преобразует A к матрице Хессенберга с помощью отражений Хаусхолдера.

    Примечание: поиск собственных значений такой матрицы легче, чем общей.

    Args:
        A (NDArray): Квадратная матрица.

    Returns:
        (NDArray): Квадратная матрица Хессенберга.
    """    
    
    n = A.shape[0]
    A = A.copy()
    
    for i in range(n-2):
        P = np.eye(n)
        P[i+1:, i+1:] = householder_reflection(A[i+1:, i])
        A = P @ (A @ np.conj(P.T))
        
    return A


def qr_decomposition(R: Matrix) -> (Matrix, Matrix):
    """Производит QR-разложение матрицы A с помощью преобразования Хаусхолдера.

    Args:
        A (Matrix): Матрица

    Returns:
        (NDArray, NDArray): Q, R матрицы.
    """

    n = R.shape[0]
    R = R.copy()
    Q = np.eye(n)

    for i in range(n):
        H = np.eye(n)
        H[i:, i:] = householder_reflection(R[i:, i])
        Q = Q @ H
        R = H @ R

    return Q, np.triu(R)
