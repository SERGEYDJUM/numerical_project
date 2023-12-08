from eigenfind import min_eigen_pair, max_eigen_pair
import numpy as np
from numpy.linalg import norm
# import scipy.linalg as spl


def print_round(val, vec, order = 2):
    print(f"\t{val:.{order}f}: [ ", end='')
    for coord in vec:
        print(f"{coord:.{order}f} ", end='')
    print("]")


A = np.array([[2, 0, 0], [0, 3, 4], [0, 4, 9]], dtype=float)

npvals, npvecs = np.linalg.eig(A)
npvecs = np.swapaxes(npvecs, 0, 1)
true_val_vec = list(sorted(zip(npvals, npvecs), key=lambda x: abs(x[0])))

print("Eigenvalues and eigenvectors calculated by numpy.linalg.eig:")
for val, vec in true_val_vec:
    print(f"\t {val}: {vec}")
print()

print("[Метод простых итераций] Наибольшее по модулю собственное значение и соотвествующий вектор: ")
val, vec = max_eigen_pair(A)
print_round(val, vec)
print(f"\tErrors: {abs(val - true_val_vec[-1][0])}, {norm(vec - true_val_vec[-1][1])}")
print()

print("[Метод обратных простых итераций] Наименьшее по модулю собственное значение и соотвествующий вектор: ")
val, vec = min_eigen_pair(A)
print_round(val, vec)
print(f"\tErrors: {abs(val - true_val_vec[0][0])}, {norm(vec - true_val_vec[0][1])}")
print()