from eigenfind import min_eigen_pair, max_eigen_pair, eigen_pairs_symmetric, eigen_values
import numpy as np
from numpy.linalg import norm

# import scipy.linalg as spl


def print_round(val, vec, order=2):
    print(f"\t{val:.{order}f}: [ ", end="")
    for coord in vec:
        print(f"{coord:.{order}f} ", end="")
    print("]")


def eigvec_error(x, y):
    return min(norm(x - y), norm(x + y))


# A = np.array([[-1, 0], [0, -1]], dtype=float)
A = np.array([[-2, 0, 0], [0, 3, 4], [0, 4, 9]], dtype=float)

npvals, npvecs = np.linalg.eig(A)
npvecs = np.swapaxes(npvecs, 0, 1)
true_val_vec = list(sorted(zip(npvals, npvecs), key=lambda x: x[0]))
true_val_vec_abs_sort = list(sorted(zip(npvals, npvecs), key=lambda x: abs(x[0])))


def closest(lam):
    clo_val_idx = 0
    for i, t in enumerate(true_val_vec):
        val, _ = t
        if abs(lam - val) <= abs(lam - true_val_vec[clo_val_idx][0]):
            clo_val_idx = i
    return true_val_vec[clo_val_idx]


print("[Ориентировочные собственные значения и векторы, вычесленные numpy.linalg.eig]:")
for val, vec in true_val_vec:
    print_round(val, vec)
print()

print(
    "[Метод степенных итераций] Наибольшее по модулю собственное значение и соотвествующий вектор: "
)
val, vec = max_eigen_pair(A, deterministic=True)
print_round(val, vec)
print(f"\tErrors: {norm(val - closest(val)[0])}, {eigvec_error(vec, closest(val)[1])}")
print()

print(
    "[Метод итераций Рэлея] Наименьшее по модулю собственное значение и соотвествующий вектор: "
)
val, vec = min_eigen_pair(A, deterministic=True)
print_round(val, vec)
print(f"\tErrors: {norm(val - closest(val)[0])}, {eigvec_error(vec, true_val_vec_abs_sort[0][1])}")
print()

print("[Метод вращений] Собственные значения и соотвествующие векторы: ")
vals, vecs = eigen_pairs_symmetric(A)
val_errors, vec_errors = [], []
for i, data in enumerate(sorted(zip(vals, vecs), key=lambda x: x[0])):
    val, vec = data
    val_errors.append(norm(val - closest(val)[0]))
    vec_errors.append(eigvec_error(vec, closest(val)[1]))
    print_round(val, vec)
print(f"\tAverage errors: {np.average(val_errors)}, {np.average(vec_errors)}")
print()

print("[Метод QR] Собственные значения: ")
vals = eigen_values(A)
for val in sorted(vals):
    val_errors.append(norm(val - closest(val)[0]))
    print(f"\t{val:.2f}")
print(f"\tAverage error: {np.average(val_errors)}")
print()