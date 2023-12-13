from numpy.typing import NDArray
from numpy.linalg import norm
from math import isclose
import numpy as np

from utils import matrix_is_singular, gauss_jordan, determinant, qr_decomposition
from eigenfind import (
    eigen_values,
    eigen_pairs_symmetric,
    eigen_values_real,
    max_eigen_pair,
    min_eigen_pair,
    rank_two_eigen_pairs,
)

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


def get_symm_singular(r: int) -> Matrix:
    while True:
        A = get_rand_symm_matrix(r)
        if matrix_is_singular(A):
            return A


def T_relay() -> str:
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
                return f"Is not a biggest eigenvalue: {np_lams} {lam_big}"
            else:
                print(vec_small, np_vecs)
                return f"Calculated eigenvector won't fit any NP vectors {vec_big} {np_vecs}"

        if abs(lam_small - np_small[0]) > 1e-9:
            if pair_fits_any(np_vecs, np_lams, vec_small, lam_small):
                return f"Is not a smallest eigenvalue: {np_lams} {lam_small}"
            else:
                return f"Calculated eigenvector won't fit any NP vectors {vec_small} {np_vecs}"


def T_rotation() -> str:
    # Проверка метода вращений
    matricies = [get_symm_singular(r) for r in range(2, 18)]

    for i, A in enumerate(matricies):
        np_lams, np_vecs = np.linalg.eig(A)
        np_vecs = np.transpose(np_vecs)
        lams, vecs = eigen_pairs_symmetric(A)

        true_res = sorted(zip(np_lams, np_vecs), key=lambda x: x[0])
        res = sorted(zip(lams, vecs), key=lambda x: x[0])

        for re, tre in zip(res, true_res):
            if not abs(re[0] - tre[0]) < 1e-6:
                return "Eigenvalues do not match"

            if not (close(re[1], tre[1]) or close(re[1], tre[1] * -1)):
                return "Eigenvectors do not match"


def T_two_by_two_complex() -> str:
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
                    return f"Eigenvectors do not match: {r[1]} != {tr[1]}"
            else:
                return "2x2 Matrix check failed"


def T_qr_iter() -> str:
    matricies = filter(
        matrix_is_singular, [np.random.rand(r, r) for r in range(2, 32, 3)]
    )
    # matricies = [get_rand_symm_matrix(r) for r in range(2, 18)]

    for i, matrix in enumerate(matricies):
        true_lams = sorted(np.linalg.eig(matrix)[0], key=lambda x: (x.real, x.imag))
        lams = sorted(eigen_values(matrix), key=lambda x: (x.real, x.imag))

        for lam, true_lam in zip(lams, true_lams):
            if norm(lam - true_lam) > 1e-2:
                # print(f"{i=}", end="\n\n")
                # print(f"{true_lams=}", end="\n\n")
                # print(f"{lams=}", end="\n\n")
                # print(matrix)
                return f"Failed qr_iter_complex check: {lam} != {true_lam} on dim {matrix.shape[0]}"
            
            
def T_qr_iter_realonly() -> str:
    matricies = [get_symm_singular(r) for r in range(2, 32, 3)]
    # matricies = [get_rand_symm_matrix(r) for r in range(2, 18)]

    for i, matrix in enumerate(matricies):
        true_lams = sorted(np.linalg.eig(matrix)[0], key=lambda x: (x.real, x.imag))
        lams = sorted(eigen_values_real(matrix), key=lambda x: (x.real, x.imag))

        for lam, true_lam in zip(lams, true_lams):
            if norm(lam - true_lam) > 1e-2:
                # print(f"{i=}", end="\n\n")
                # print(f"{true_lams=}", end="\n\n")
                # print(f"{lams=}", end="\n\n")
                # print(matrix)
                return f"Failed qr_iter_real check: {lam} != {true_lam} on dim {matrix.shape[0]}"


def T_det() -> str:
    matricies = [np.random.rand(r, r) for r in range(3, 128, 3)]

    for i, A in enumerate(matricies):
        if not isclose(np.linalg.det(A), determinant(A)):
            return "Failed det check"


def T_qr_gauss() -> str:
    matricies = filter(
        matrix_is_singular, [np.random.rand(r, r) for r in range(3, 128, 3)]
    )

    for i, A in enumerate(matricies):
        Y = np.ones(A.shape[0])
        
        Q, R = qr_decomposition(A)
        if not close(Q @ R, A, eps=1e-9):
            return "Failed QR check"
        
        if not close(np.linalg.solve(A, Y), gauss_jordan(A, Y)):
            return "Failed SLE solution check"


def playground() -> str:
    def comp_key(x):
        return (x.real, x.imag)
    
    A = np.array([
        [0.08770608, 0.66733973, 0.68138823, 0.95954237, 0.15455779],
        [0.09314774, 0.75665306, 0.70949404, 0.4281082,  0.92111275],
        [0.42515646, 0.28478231, 0.47489026, 0.77481373, 0.20974817],
        [0.94189888, 0.00381377, 0.8366714,  0.04782337, 0.59311576],
        [0.04456993, 0.4124859,  0.01261069, 0.58334299, 0.33979129],
    ])
    
    lams = sorted(eigen_values(A), key=comp_key)
    true_lams = sorted(np.linalg.eig(A)[0], key=comp_key)
    
    matched = []
    for tlam in true_lams:
        for lam in lams:
            if np.isclose(lam, tlam):
                matched.append(lam)
                break
    
    print(f"Determinant: {determinant(A)}")
    print(f"True eigen values: {true_lams}")
    print(f"QRI eigenvalues: {lams}")
    print(f"Mismatched: {[x for x in lams if x not in matched]}")
    print("\n\n")
    
    
    

if __name__ == "__main__":
    tests = [T_det, T_qr_gauss, T_two_by_two_complex, T_relay, T_rotation, T_qr_iter_realonly, T_qr_iter]
    # tests = [playground, T_det, T_qr_gauss]
    
    
    for test in tests:
        if msg := test():
            print(msg)
