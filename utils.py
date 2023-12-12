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

    return not isclose(abs(determinant(A)), 0, rel_tol=1e-4)


def validate_matrix(A: Matrix):
    """Выводит ошибку, если матрица вырождена или неквадратна.

    Args:
        A (NDArray): Матрица.
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")
    if not matrix_is_singular(A):
        raise ValueError("Matrix must have a determinant")


def validate_square(A: Matrix):
    """Выводит ошибку, если матрица неквадратна.

    Args:
        A (NDArray): Матрица.
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square")


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


def qr_decomposition(A: Matrix) -> (Matrix, Matrix):
    """Производит QR-разложение матрицы A с помощью преобразования Хаусхолдера.

    Args:
        A (Matrix): Матрица

    Returns:
        (NDArray, NDArray): Q, R матрицы.
    """
    
    n = A.shape[0]
    R = A.copy()
    Q = np.eye(n)
    
    for i in range(n):
        u = R[i:, i, np.newaxis]
        v = u / (u[0] + norm(u) * np.sign(u[0]))
        v[0] = 1
        
        diff = (v @ v.T) * 2 / (v.T @ v) 
        H = np.eye(n)
        H[i:, i:] -= diff
        
        R = H @ R
        Q = H @ Q
        
    return Q[:n].T, np.triu(R[:n])

#######################################################

if __name__ == "__main__":
    def close(x, y) -> bool:
        return norm(x - y) < 1e-7
    
    matricies = filter(
        matrix_is_singular, [np.random.rand(r, r) for r in range(3, 128, 3)]
    )

    for i, A in enumerate(matricies):
        Y = np.ones(A.shape[0])
        
        assert isclose(np.linalg.det(A), determinant(A)), "Failed det check"
        
        assert close(
            np.linalg.solve(A, Y), gauss_jordan(A, Y)
        ), "Failed SLE solution check"
        
        Q, R = qr_decomposition(A)
        assert close(Q @ R, A), "Failed QR check"
        break