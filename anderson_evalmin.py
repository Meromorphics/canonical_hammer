import scipy
import geom
import scipy
import itertools

# Complex eigenvalue minimize... on an Anderson model

# 1 --t1-- 2 --t2-- 3     Uc
# |        |        |
# v1       v2       v3
# |        |        |
# 4        5        6     Ud

n = 4

#updnpairs = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
#             (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
#             (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
#             (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
#             (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
#             (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
#             (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6),
#             (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]

# equivalent to the above (at least for 2*n=6)
updnpairs = list(itertools.product(range(1, n+1), range(n+1))) + list(itertools.product([0], range(1, n+1)))

L = geom.Lattice(2*n)
for i in range(1, n):
    L.sym_couple(i, i+1, f"t{i}")
for i in range(1, n+1):
    L.sym_couple(i, i+n, f"v{i}")
for i in range(1, n+1):
    L.interaction(i, "Uc")
for i in range(1, n+1):
    L.interaction(i+n, "Ud")


Bs = [geom.Basis(2*n, nups, ndns) for nups, ndns in updnpairs]
Hs = [geom.Hamiltonian() for _ in updnpairs]
total = len(updnpairs)
for i, (H, B) in enumerate(zip(Hs, Bs)):
    print(f"Constructing H for (nups, ndns) = {updnpairs[i]}, {i+1}/{total}...")
    H.construct(L, B)
    H.zero_H()
print(f"All H's constructed")

def f(x, *args):
    ts = x[:n-1]
    vs = x[n-1:-2]
    Uc = x[-2]
    Ud = x[-1]

    complex_sum = 0
    for H in Hs:
        for i, t in enumerate(ts):
            H.ts[f"t{i+1}"] = ts[i]
        for i, v in enumerate(vs):
            H.ts[f"v{i+1}"] = vs[i]
        H.Us["Uc"] = Uc
        H.Us["Ud"] = Ud
        H.make_H(fresh=False)
        eigenvalues = scipy.linalg.eig(H.H, b=None, left=False, right=False,
                                       overwrite_a=False, overwrite_b=False,
                                       check_finite=True, homogeneous_eigvals=False)
        #eigenvalues = scipy.sparse.linalg.eigs(scipy.sparse.csr_matrix(H.H),
        #                                       return_eigenvectors=False)
        complex_sum = complex_sum + sum(abs(eigenvalues.imag))
    print(complex_sum)
    return complex_sum


#                           ts                                 vs                          Us
bounds = list(zip([0] * (n-1), [10] * (n-1))) + list(zip([0] * n, [10] * n)) + list(zip([1] * 2, [5] * 2))

result = scipy.optimize.dual_annealing(f, bounds=bounds)
print(result)
