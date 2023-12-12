from numpy.typing import NDArray
from numpy.random import randn
from numpy.linalg import norm
from math import isclose
import numpy as np

from utils import validate_matrix, gauss_jordan, matrix_is_singular, qr_decomposition, validate_square

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
            X = gauss_jordan(lam * Eye - A, X)
        except ValueError:
            return lam, X

        X /= norm(X)

        old_lam, lam = lam, X @ A @ X
        if abs(old_lam - lam) < eps:
            break

    return lam, X


def min_eigen_pair(
    A: Matrix, max_iter: int = 512, eps: float = 1e-12, deterministic: bool = False
) -> (float, Vector):
    # TODO: Make so that it actually finds smallest
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

    return closest_eigen_pair(A, 0, max_iter, eps, deterministic)


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


def eigen_values(A: Matrix, max_iter: int = 1024, eps: float = 1e-9) -> Vector:
    """Вычисляет собственные значения квадратной матрицы итеративно \
        с помощью QR-разложения.

    Args:
        A (NDArray): Матрица.
        
        max_iter (int, optional): Ограничение по итерациям алгоритма.
        
        eps (float): Ограничение по точности, по достижении которй итерации прекращаются.
        
    Returns:
        (NDArray): Массив собственных значений.
    """    
    
    validate_square(A)
    n = A.shape[0]
    A_k = A
    Q_k = np.eye(n)
    for _ in range(max_iter):
        shift = np.eye(n) * (A_k[-1, -1] * 0.99) # Limit shift
        Q, R = qr_decomposition(A_k - shift)
        A_k = R @ Q + shift
        Q_k = Q_k @ Q
        
        if isclose(norm(np.tril(A_k, k=-1)), 0, rel_tol=eps):
            break
    
    return np.diag(A_k)


def rank_two_eigen_pairs(A: Matrix) -> (Vector, Matrix):
    """Находит все собственные значения матрицы 2x2 и соответствующие собственные векторы напрямую.
    
    Примечание: поддерживаются только невырожденные матрицы с действительными собственными значениями.

    Args:
        A (NDArray): Квадратная матрица.

    Returns:
        (NDArray, NDArray): Массив собственных значений и матрица с собственными векторами по рядам.
    """
    if A.shape[0] != 2:
        raise ValueError("Matrix with incorrect dims was passed")
    
    a = A[0, 0]
    b = A[0, 1]
    c = A[1, 0]
    d = A[1, 1]
    
    det = a*d - b*c
    if isclose(det, 0.0):
        raise ValueError("Non-singular matricies are not supported")
    
    dis = (a + d)**2 - 4*det
    
    if dis < 0.0:
        raise ValueError("Complex eigenvalues are not supported")
    
    vecs = np.zeros((2, 2))
    lams = np.zeros(2)
    for i, sign in enumerate((-1, 1)):
        lams[i] = (a + d + sign * np.sqrt(dis))/2
        vecs[i] = np.array([1.0, (a + c - lams[i])/(lams[i] - d - b)])
        vecs[i] /= norm(vecs[i])
            
    return lams, vecs 

####################################################################

if __name__ == "__main__":

    def close(x, y, eps=1e-4) -> bool:
        return norm(x - y) < eps

    def pair_fits_any(
        corr_vecs: Matrix, corr_lams: Vector, incorrect: Vector, lam: float
    ) -> bool:
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


    # Проверка QR
    matricies = filter(matrix_is_singular, [np.random.rand(r, r) for r in range(2, 32, 3)])
    
    for i, matrix in enumerate(matricies):
        lams = sorted(eigen_values(matrix))
        tlams = sorted(np.linalg.eig(matrix)[0])
        
        for lam, tlam in zip(lams, tlams):
            if isinstance(tlam, complex):
                continue
            assert isclose(norm(lam), norm(tlam)), f"{lam} != {tlam}"
    
    
    # Проверка 2x2
    matricies = filter(matrix_is_singular, [np.random.rand(2, 2) for _ in range(100)])
    
    for i, A in enumerate(matricies):
        
        np_lams, np_vecs = np.linalg.eig(A)
        lams, _ = rank_two_eigen_pairs(A)
        
        true_res = sorted(zip(np_lams, np_vecs), key=lambda x: x[0])
        res = sorted(zip(lams, lams), key=lambda x: x[0])
        
        for r, tr in zip(res, true_res):
            if abs(r[0] - tr[0]) > 1e-5:
                print(A)
                print(np_lams, lams)
                assert False


    # Проверка метода Рэлея и степенных итераций
    matricies = filter(
        matrix_is_singular, [np.random.rand(r, r) for r in range(2, 28, 1)]
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

        lam_small, vec_small = min_eigen_pair(A)
        lam_big, vec_big = max_eigen_pair(A)
        
        if abs(lam_big - np_big[0]) > 1e-9:
            if pair_fits_any(np_vecs, np_lams, vec_big, lam_big):
                print(f"Lambdas by NumPy and wrong biggest: {np_lams, lam_big}")
                assert False, "Is not a biggest eigenvalue"
            else:
                print(vec_small, np_vecs)
                assert False, "Calculated eigenvector won't fit any NP vectors"
        
        if abs(lam_small - np_small[0]) > 1e-9:
            if pair_fits_any(np_vecs, np_lams, vec_small, lam_small):
                print(f"Lambdas by NumPy and wrong smallest: {np_lams, lam_small}")
                assert False, "Is not a minimal eigenvalue"
            else:
                print(vec_small, np_vecs)
                assert False, "Calculated eigenvector won't fit any NP vectors"


    # Проверка метода вращений
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
