from numpy.typing import NDArray
from numpy.linalg import norm
import numpy as np

from utils import validate_matrix, lu_decomposition, lu_solve, partial_pivot_matrix

Matrix = NDArray[np.float_]
Vector = NDArray[np.float_]

def max_eigen_pair(A: Matrix, max_iter: int = 512, eps: float = 1e-9) -> (float, Vector):
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
        lam_old, lam = lam, abs(X[max_idx] / X_old[max_idx])

        X /= norm(X)
        if abs(lam_old - lam) < eps:
            break
    
    return (lam, X) # TODO: Polarity


def closest_eigen_pair(A: Matrix, approx_eigen: float = 0.0, max_iter: int = 512, eps: float = 1e-9) -> (float, Vector):
    """Находит самое близкое к модулю approx_eigen собственное значение и соответствующий собственный вектор.

    Args:
        A (NDArray): Квадратная матрица. 
        
        max_iter (int): Максимальное количество итераций алгоритма.
        
        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.
        
    Returns:
        (float, NDArray): Собственное значение и вектор.
    """
    
    validate_matrix(A)
    A = abs(approx_eigen)*np.identity(A.shape[0]) - A # Check for zeros on diag
    
    X = np.ones(A.shape[0])
    X_old, X = X, X / norm(X)
    
    P = partial_pivot_matrix(A)
    LU = lu_decomposition(A, P)
    
    lam_old, lam = 1, 2
    for _ in range(max_iter):
        X, X_old = lu_solve(LU, P, X), X
        X /= norm(X)
        
        max_idx = np.argmax(np.abs(X))
        lam_old, lam = lam, abs(X_old[max_idx] / X[max_idx])
        
        if abs(lam_old - lam) < eps:
            break
    
    return (lam, -X) # TODO: Polarity


def min_eigen_pair(A: Matrix, max_iter: int = 512, eps: float = 1e-9) -> (float, Vector):
    """Находит наименьшее по модулю собственное значение и соответствующий собственный вектор.

    Args:
        A (NDArray): Квадратная матрица. 
        
        max_iter (int): Максимальное количество итераций алгоритма.
        
        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.
        
    Returns:
        (float, NDArray): Собственное значение и вектор.
    """
    
    return closest_eigen_pair(A, max_iter=max_iter, eps=eps)


