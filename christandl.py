import scipy
import geom
import scipy
import numpy

# Implements the Christandl hoppings to take an electron down a chain.

# 1 --t1-- 2 --t2-- 3 --t3-- 4 -- ... -- tn-1 -- n

n = 12
nups = 1
ndns = 1
time = numpy.pi / 2
U = 0

B = geom.Basis(n, nups, ndns)
L = geom.Lattice(n)

initial_state = geom.State(ups=(1,), dns=(1,))
final_state = geom.State(ups=(n,), dns=(n,))
initial_index = B.index(initial_state)
final_index = B.index(final_state)
psi0 = numpy.zeros(B.N)
psi0[initial_index-1] = 1


for i in range(n):
    L.sym_couple(i+1, i+2, numpy.sqrt((i+1) * (n-i-1)))

H = geom.Hamiltonian()
H.construct(U, L, B)
H.make_H()

psi1 = scipy.linalg.expm(-1.0j * H.H * time) @ psi0
final_probability = abs(psi1[final_index-1]) ** 2
print(final_probability)