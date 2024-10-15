import geom
import matplotlib.pyplot
import scipy
import numpy

# I forget the picture for this one
# The filename said that it involved triangles
# Prints some eigenvalues of the resulting Hamiltonian

n = 7
nups = 1
ndns = 0
U = 1
t = 1
delta = 0.5
tp = t + delta
tm = t - delta

B = geom.Basis(n, nups, ndns)
L = geom.Lattice(n)

L.two_couple(1, 2, tp, tm)
L.two_couple(1, 3, tm, tp)
L.two_couple(1, 4, tm, tp)
L.two_couple(2, 4, tm, tp)
L.two_couple(2, 5, tm, tp)
L.two_couple(3, 4, tp, tm)
L.two_couple(4, 5, tp, tm)
L.two_couple(3, 6, tm, tp)
L.two_couple(4, 6, tm, tp)
L.two_couple(4, 7, tm, tp)
L.two_couple(5, 7, tm, tp)


H = geom.Hamiltonian()
H.construct(U, L, B)

H.make_H()

eigenvalues = scipy.linalg.eig(H.H, b=None, left=False, right=False,
                               overwrite_a=False, overwrite_b=False,
                               check_finite=True, homogeneous_eigvals=False)
i = 1
for a in numpy.sort(eigenvalues):
    print(i, a)
    i = i + 1
matplotlib.pyplot.matshow(H.H)
matplotlib.pyplot.show()
