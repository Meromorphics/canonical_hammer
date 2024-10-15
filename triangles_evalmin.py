import scipy
import geom
import itertools

# Complex eigenvalue minimize
#
# Searches for asymmetric hoppings in which the resulting Hamiltonian
# has all real eigenvalues (by minimizing their absolute value sum)
#
#
#             1              4
#           /   \          /   \
#         t12   t13      t45   t46
#         /       \      /       \
#        2 - t23 - 3    5 - t56 - 6
#
# interconnection: 1 - t14 - 4
#                  2 - t26 - 6
#                  3 - t35 - 5
#
# hopping between two sites are free to be different
#

n = 6
U = 4

#updnpairs = [(1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0),
#             (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1),
#             (1, 2), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2),
#             (1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (6, 3),
#             (1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4),
#             (1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5),
#             (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (6, 6),
#             (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6)]

# equivalent to the above (at least for n=6)
updnpairs = list(itertools.product(range(1, n+1), range(n+1))) + list(itertools.product([0], range(1, n+1)))


def f(x, *args):
    t12l, t13l, t23l, t45l, t46l, t56l, t14l, t26l, t35l = x[:9]
    t12r, t13r, t23r, t45r, t46r, t56r, t14r, t26r, t35r = x[9:]
    L = geom.Lattice(n)
    L.two_couple(1, 2, t12l, t12r)
    L.two_couple(1, 3, t13l, t13r)
    L.two_couple(2, 3, t23l, t23r)
    L.two_couple(4, 5, t45l, t45r)
    L.two_couple(4, 6, t46l, t46r)
    L.two_couple(5, 6, t56l, t56r)
    L.two_couple(1, 4, t14l, t14r)
    L.two_couple(2, 6, t26l, t26r)
    L.two_couple(3, 5, t35l, t35r)

    complex_sum = 0
    for pair in updnpairs:
        nups, ndns = pair
        B = geom.Basis(n, nups, ndns)
        H = geom.Hamiltonian()
        H.construct(U, L, B)
        H.make_H()
        eigenvalues = scipy.linalg.eig(H.H, b=None, left=False, right=False,
                                       overwrite_a=False, overwrite_b=False,
                                       check_finite=True, homogeneous_eigvals=False)
        complex_sum = complex_sum + sum(abs(eigenvalues.imag))
    return complex_sum

bounds = list(zip([0] * 18, [100] * 18))

result = scipy.optimize.dual_annealing(f, bounds=bounds)
print(result)
