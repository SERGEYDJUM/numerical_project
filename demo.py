from eigenfind import min_eigen_pair, max_eigen_pair, eigen_pairs
import numpy as np
from numpy.linalg import norm
# import scipy.linalg as spl


def print_round(val, vec, order = 2):
    print(f"\t{val:.{order}f}: [ ", end='')
    for coord in vec:
        print(f"{coord:.{order}f} ", end='')
    print("]")



A = np.array([[-1, 0], [0, -1]], dtype=float)
# A = np.array([[-2, 0, 0], [0, 3, 4], [0, 4, 9]], dtype=float)

npvals, npvecs = np.linalg.eig(A)
npvecs = np.swapaxes(npvecs, 0, 1)
true_val_vec = list(sorted(zip(npvals, npvecs), key=lambda x: x[0]))

def closest(lam):
    clo_val_idx = 0
    for i, t in enumerate(true_val_vec):
        val, _ = t
        if abs(lam - val) <= abs(lam - true_val_vec[clo_val_idx][0]):
            clo_val_idx = i
    return true_val_vec[clo_val_idx]

print("Eigenvalues and eigenvectors calculated by numpy.linalg.eig:")
for val, vec in true_val_vec:
    print_round(val, vec)
print()

print("[Метод простых итераций] Наибольшее по модулю собственное значение и соотвествующий вектор: ")
val, vec = max_eigen_pair(A)
print_round(val, vec)
print(f"\tErrors: {norm(val - closest(val)[0])}, {norm(vec - closest(val)[1])}")
print()

print("[Метод обратных простых итераций] Наименьшее по модулю собственное значение и соотвествующий вектор: ")
val, vec = min_eigen_pair(A)
print_round(val, vec)
print(f"\tErrors: {norm(val - closest(val)[0])}, {norm(vec - closest(val)[1])}")
print()

print("[Метод вращений] Собственные значения и соотвествующие векторы: ")
vals, vecs = eigen_pairs(A)
val_errors, vec_errors = [], []
for i, data in enumerate(sorted(zip(vals, vecs), key=lambda x: x[0])):
    val, vec = data
    val_errors.append(norm(val - closest(val)[0]))
    vec_errors.append(norm(vec - closest(val)[1]))
    print_round(val, vec)
print(f"\tAverage errors: {np.average(val_errors)}, {np.average(vec_errors)}")
print()