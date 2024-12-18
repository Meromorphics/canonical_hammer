import scipy
import geom
import scipy
import numpy
import time as timer
import exp

# Implements the Christandl hoppings to take an electron down a chain.
# For example purposes, an up and down electron are taken down a chain,
# since the interaction U = 0, each electron should move independently.

# 1 --t1-- 2 --t2-- 3 --t3-- 4 -- ... -- tn-1 -- n

n = 100
nups = 1
ndns = 1
time = numpy.pi / 2
U = 0


start = timer.time()
B = geom.Basis(n, nups, ndns)
finish = timer.time()
print(f"Basis made in {finish-start} seconds")
L = geom.Lattice(n)



initial_state = geom.State(ups=(1,), dns=(1,))
final_state = geom.State(ups=(n,), dns=(n,))
initial_index = B.index(initial_state)
final_index = B.index(final_state)
psi0 = numpy.zeros(B.N)
psi0[initial_index-1] = 1

print(B.N)

for i in range(1, n):
    L.sym_couple(i, i+1, f"t{i}")
for i in range(1, n+1):
    L.interaction(i, f"U{i}")


H = geom.Hamiltonian()

for i in range(1, n):
    H.ts[f"t{i}"] = numpy.sqrt(i * (n-i))
for i in range(1, n+1):
    H.Us[f"U{i}"] = U



start = timer.time()
H.construct(L, B)
finish = timer.time()
print(f"H constructed in {finish-start} seconds")



start = timer.time()
H.make_H()
finish = timer.time()
print(f"Matrix made in {finish-start} seconds")

start = timer.time()
psi1 = exp.expA_v(H.H, psi0, -1.0j * time, "expm_multiply", traceA=H.trace())
finish = timer.time()
print(f"Exponential-vector product done in {finish-start} seconds")

final_probability = abs(psi1[final_index-1]) ** 2
print(f"Fidelity = {final_probability}")
