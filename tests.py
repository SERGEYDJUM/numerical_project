from numpy.typing import NDArray
from numpy.linalg import norm
from math import isclose
import numpy as np

from utils import matrix_is_singular, gauss_jordan, determinant, qr_decomposition
from eigenfind import eigen_values, eigen_pairs_symmetric, max_eigen_pair, min_eigen_pair, rank_two_eigen_pairs

Matrix = NDArray[np.float_]
Vector = NDArray[np.float_]    


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
        
        
# Проверка 2x2
matricies = [np.random.rand(2, 2) for _ in range(500)]

for i, A in enumerate(matricies):
    
    np_lams, np_vecs = np.linalg.eig(A)
    np_vecs = np_vecs.T
    lams, vecs = rank_two_eigen_pairs(A)
    
    true_res = sorted(zip(np_lams, np_vecs), key=lambda x: (x[0].real, x[0].imag))
    res = sorted(zip(lams, vecs), key=lambda x: (x[0].real, x[0].imag))
    
    for r, tr in zip(res, true_res):
        if isclose(r[0].real, tr[0].real) and isclose(r[0].imag, tr[0].imag):
            test1 = norm(r[1] - tr[1])
            test2 = norm(r[1] + tr[1])
            if not (np.isclose(test1, 0) or np.isclose(test2, 0)):
                assert False, f"Eigenvectors do not match: {r[1]} != {tr[1]}"
        else:
            assert False, "2x2 Matrix check failed"
        

# Проверка QR
matricies = filter(matrix_is_singular, [np.random.rand(r, r) for r in range(2, 32, 3)])
# matricies = [get_rand_symm_matrix(r) for r in range(2, 18)]

for i, matrix in enumerate(matricies):
    tlams = sorted(np.linalg.eig(matrix)[0], key=lambda x: (x.real, x.imag))
    lams = sorted(eigen_values(matrix), key=lambda x: (x.real, x.imag))
    
    for lam, tlam in zip(lams, tlams):
        if norm(lam - tlam) > 1e-3:
            print(f"{i=}", end="\n\n")
            print(f"{tlams=}", end="\n\n")
            print(f"{lams=}", end="\n\n")
        assert norm(lam - tlam) < 1e-5, f"{lam} != {tlam}"
        
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