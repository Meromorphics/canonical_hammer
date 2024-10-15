import scipy
import geom
import scipy
import itertools

# Complex eigenvalue minimize... on an Anderson model

# 1 --t1-- 2 --t2-- 3
# |        |        |
# t3       t4       t5
# |        |        |
# 4        5        6

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
    t1l, t2l, t3l, t4l, t5l = x[0], x[1], x[2], x[3], x[4]
    t1r, t2r, t3r, t4r, t5r = x[5], x[6], x[7], x[8], x[9]
    L = geom.Lattice(n)
    L.two_couple(1, 2, t1l, t1r)
    L.two_couple(2, 3, t2l, t2r)
    L.two_couple(1, 4, t3l, t3r)
    L.two_couple(2, 5, t4l, t4r)
    L.two_couple(3, 6, t5l, t5r)

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

bounds = list(zip([1] * 10, [5] * 10))

result = scipy.optimize.dual_annealing(f, bounds=bounds)
print(result)
