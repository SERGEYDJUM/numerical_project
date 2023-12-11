from numpy.typing import NDArray
from numpy.linalg import norm
from numpy.random import randn
import numpy as np

from utils import validate_matrix, gauss_jordan, GLOBAL_EPS, matrix_is_singular

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
        X = gauss_jordan(lam * Eye - A, X)
        X /= norm(X)

        old_lam, lam = lam, X @ A @ X
        if abs(old_lam - lam) < eps:
            break

    return lam, X


def min_eigen_pair(A: Matrix, max_iter: int = 512, eps: float = 1e-12, deterministic: bool = False) -> (float, Vector):
    """Находит минимальное по модулю собственное значение и \
        соответствующий собственный вектор c помощью метода итераций Рэлея.

    Args:
        A (NDArray): Квадратная матрица.

        max_iter (int): Максимальное количество итераций алгоритма.

        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.
        
        deterministic (bool): Включает заполнение начальных значений единицами, вместо случайных чисел.

    Returns:
        (float, NDArray): Собственное значение и вектор.
    """
    
    attempts = [closest_eigen_pair(A, 10, max_iter, eps, deterministic),
        closest_eigen_pair(A, 0, max_iter, eps, deterministic),
        closest_eigen_pair(A, -10, max_iter, eps, deterministic)]

    return sorted(attempts, key=lambda x: abs(x[0]))[0]


def eigen_pairs_symmetric(
    A: Matrix, max_iter: int = 512, eps: float = 1e-12
) -> (Vector, Matrix):
    """Находит все собственные значения симметричной матрицы и соответствующие собственные векторы с \
        помощью метода поворотов.

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


if __name__ == "__main__":
    def close(x, y, eps=1e-5) -> bool:
        return np.linalg.norm(x - y) < eps

    def pair_fits_any(corr_vecs: Matrix, corr_lams: Vector, incorrect: Vector, lam: float) -> bool:
        for corvec, corlam in zip(corr_vecs, corr_lams):
            if close(corvec, incorrect, 1e-4) or close(corvec, incorrect * -1, 1e-4):
                if abs(corlam - lam) < 1e-4:
                    return True
        return False
    
    def get_rand_symm_matrix(r: int) -> Matrix:
        A = np.random.rand(r, r)
        A = np.triu(A) + np.triu(A, k=1).T
        assert close(A - A.T, np.zeros_like(A))
        return A

    matricies = filter(
        matrix_is_singular, [np.random.rand(r, r) for r in range(2, 18, 1)]
    )

    for i, A in enumerate(matricies):
        np_lams, np_vecs = np.linalg.eig(A)            
        np_vecs = np.transpose(np_vecs)
        
        if not close(np.conj(np_lams), np_lams):
            # Skipping complex eigenvalues
            continue
        
        if any(not close(vec, np.conj(vec)) for vec in np_vecs):
            # Skipping complex eigenvectors
            continue
            
        true_res = list(sorted(zip(np_lams, np_vecs), key=lambda x: abs(x[0])))
        np_small = true_res[0]
        np_big = true_res[-1]

        lam_small, vec_small = min_eigen_pair(A, deterministic=True)
        if abs(lam_small - np_small[0]) > GLOBAL_EPS:
            if pair_fits_any(np_vecs, np_lams, vec_small, lam_small):
                print(f"Lambdas by NumPy and wrong smallest: {np_lams, lam_small}")
                assert False, "Is not a minimal eigenvalue"
            else:
                assert False, "Calculated eigenvector won't fit any NP vectors"
            

    matricies = [get_rand_symm_matrix(r) for r in range(2, 18)]

    for i, A in enumerate(matricies):
        np_lams, np_vecs = np.linalg.eig(A)
        np_vecs = np.transpose(np_vecs)
        lams, vecs = eigen_pairs_symmetric(A)

        true_res = sorted(zip(np_lams, np_vecs), key=lambda x: x[0])
        res = sorted(zip(lams, vecs), key=lambda x: x[0])

        for re, tre in zip(res, true_res):
            assert abs(re[0] - tre[0]) < 1e-6
            assert close(re[1], tre[1]) or close(re[1], tre[1] * -1)
