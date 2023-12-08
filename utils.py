from numpy.typing import NDArray
# from numpy.linalg import norm
import numpy as np

Matrix = NDArray[np.float_]
Vector = NDArray[np.float_]


def validate_matrix(A: Matrix):
    assert A.shape[0] == A.shape[1], "Matrix must be square"


def partial_pivot_matrix(A: Matrix) -> Matrix:
    n = A.shape[0]
    P = np.identity(n, dtype=np.float_)
    for i in range(n):
        max_idx = i + np.argmax(np.abs(A[i:n, i]))
        if max_idx != i:
            P[[i, max_idx]] = P[[max_idx, i]]
    return P


def lu_decomposition(A: Matrix, P: Matrix) -> Matrix:
    n = A.shape[0]
    PA = P @ A
    LU = np.identity(n)
    for i in range(n):
        for j in range(i):
            LU[j, i] = PA[j, i] - LU[:j, i] @ LU[j, :j]
        for j in range(i, n):
            LU[j, i] = (PA[j, i] - LU[:i, i] @ LU[j, :i]) / LU[i, i]
    return LU


def lu_solve(LU: Matrix, P: Matrix, Y: Vector) -> Vector: # Gigantic error
    n = LU.shape[0]
    y = np.zeros_like(Y)
    for i in range(n):
        y[i] = Y[i] - LU[i, :i] @ y[:i]

    X = np.zeros_like(Y)
    for i in reversed(range(n)):
        X[i] = (y[i] - (LU[i, i + 1 : n] @ X[i + 1 : n])) / LU[i, i]

    return X


if __name__ == "__main__":
    import scipy.linalg as spl
    for A in [
        [[1, 1, 1], [4, 3, -1], [3, 5, 3]],
        [[2, 0, 0], [0, 3, 4], [0, 4, 9]],
        [[1, 2, 3], [3, 2, 1], [2, 1, 3]]
    ]:
        A = np.array(A, dtype=float)
        
        lu, p = spl.lu_factor(A)
        P = partial_pivot_matrix(A)
        LU = lu_decomposition(A, P)
        assert np.linalg.norm(LU - lu) < 1e-9
        break
