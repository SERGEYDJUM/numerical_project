from numpy.typing import NDArray
import numpy as np

Matrix = NDArray[np.float_]
Vector = NDArray[np.float_]

GLOBAL_EPS = 1e-9

def determinant(A: Matrix, eps: float = 1e-9) -> float:
    """Вычисляет определитель наибольшего минора матрицы (самой матрицы).

    Примечание: 

    Args:
        A (NDArray): Матрица

    Returns:
        float: Определитель матрицы.
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
        
        if abs(A[i, i]) < GLOBAL_EPS:
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
        bool: False, если матрица вырождена.
    """
    
    return abs(determinant(A)) > GLOBAL_EPS


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
        NDArray: Матрица P.
    """
    
    n = A.shape[0]
    P = np.identity(n, dtype=np.float_)
    for i in range(n):
        max_idx = i + np.argmax(np.abs(A[i:n, i]))
        P[[i, max_idx]] = P[[max_idx, i]]
    return P


def lu_decomposition(A: Matrix, P: Matrix) -> Matrix:
    """Раскладывает матрицу A на две матрицы L и U так, что \
    P @ A = L @ U.

    Args:
        A (NDArray): Исходная матрица.
        P (NDArray): Матрица перестановок строк.

    Returns:
        NDArray: LU матрица, верхняя треугольная часть которой = U, \
        а ниже неё L - I, где I - единичная матрица.
    """
    
    n = A.shape[0]
    PA = P @ A
    LU = np.identity(n)
    for i in range(n):
        for j in range(i):
            LU[j, i] = PA[j, i] - LU[:j, i] @ LU[j, :j]
        for j in range(i, n):
            LU[j, i] = (PA[j, i] - LU[:i, i] @ LU[j, :i]) / LU[i, i]
    return LU


def lu_solve(LU: Matrix, P: Matrix, Y: Vector) -> Vector: # TODO: Fix gigantic error
    n = LU.shape[0]
    y = np.zeros_like(Y)
    for i in range(n):
        y[i] = Y[i] - LU[i, :i] @ y[:i]

    X = np.zeros_like(Y)
    for i in reversed(range(n)):
        X[i] = (y[i] - (LU[i, i + 1 : n] @ X[i + 1 : n])) / LU[i, i]

    return X


if __name__ == "__main__":
    def close(x, y, eps = 1e-5) -> bool:
        return np.linalg.norm(x-y) < eps
    
    matricies = [np.random.rand(r, r) for r in range(2, 128, 4)]
    
    matricies = filter(matrix_is_singular, [np.random.rand(r, r) for r in range(2, 128, 4)])
    
    for i, A in enumerate(matricies):
        P = partial_pivot_matrix(A)
        LU = lu_decomposition(A, P)
        L = np.tril(LU, k=-1) + np.eye(A.shape[0])
        U = np.triu(LU)
        assert np.linalg.det(A) / determinant(A) - 1 < 1e-9, "Failed det check"
        assert close(P@A, L@U), "Failed LU check"
    