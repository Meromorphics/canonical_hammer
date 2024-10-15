import scipy
import geom
import scipy
import numpy

# Optimize hoppings to take an up and down electron from
# the left to the right: quantum state transfer

# 1 --t1-- 2 --t2-- 3 --t3-- 4 --t4-- 5 --t5-- 6

n = 6
nups = 1
ndns = 1
U = 1
time = numpy.pi / 2

B = geom.Basis(n, nups, ndns)
initial_state = geom.State(ups=(1,), dns=(1,))
final_state = geom.State(ups=(n,), dns=(n,))
initial_index = B.index(initial_state)
final_index = B.index(final_state)
psi0 = numpy.zeros(B.N)
psi0[initial_index-1] = 1
global i
i = 0

def f(x, *args):
    t1, t2, t3, t4, t5 = x[0], x[1], x[2], x[3], x[4]
    L = geom.Lattice(n)
    L.sym_couple(1, 2, t1)
    L.sym_couple(2, 3, t2)
    L.sym_couple(3, 4, t3)
    L.sym_couple(4, 5, t4)
    L.sym_couple(5, 6, t5)
    H = geom.Hamiltonian()
    H.construct(U, L, B)
    H.make_H()
    psi1 = scipy.linalg.expm(-1.0j * H.H * time) @ psi0
    final_probability = abs(psi1[final_index-1]) ** 2
    global i
    i = i + 1
    return -final_probability

bounds = list(zip([0] * 5, [20] * 5))

result = scipy.optimize.dual_annealing(f, bounds=bounds)
print(result)
