import scipy
import geom
import scipy
import numpy
import time as timer
import exp

# Optimize hoppings to take an up and down electron from
# the left to the right: quantum state transfer

# 1 --t1-- 2 --t2-- 3 --t3-- 4 --t4-- 5 --t5-- 6

n = 20
nups = 1
ndns = 1
time = numpy.pi / 2


start = timer.time()
B = geom.Basis(n, nups, ndns)
finish = timer.time()
print(f"Basis made in {finish-start} seconds")

initial_state = geom.State(ups=(1,), dns=(1,))
final_state = geom.State(ups=(n,), dns=(n,))
initial_index = B.index(initial_state)
final_index = B.index(final_state)
psi0 = numpy.zeros(B.N)
psi0[initial_index-1] = 1

H = geom.Hamiltonian()
L = geom.Lattice(n)

for i in range(1, n):
    L.sym_couple(i, i+1, f"t{i-i}")
for i in range(1, n+1):
    L.interaction(i, f"U{i-1}")

start = timer.time()
H.construct(L, B)
finish = timer.time()
print(f"H constructed in {finish-start} seconds")

H.zero_H()

def f(x, *args):
    for i in range(n-1):
        H.ts[f"t{i}"] = x[i]
    for i in range(n):
        H.Us[f"U{i}"] = x[i+n-1]
    H.make_H(fresh=False)
    psi1 = exp.expA_v(H.H, psi0, -1.0j * time, "expm_multiply", traceA=H.trace())
    final_probability = abs(psi1[final_index-1]) ** 2
    print(final_probability)
    return -final_probability

bounds = list(zip([0] * (n-1), [40] * (n-1))) + list(zip([1] * n, [5] * n))
print("Starting optimization...")
result = scipy.optimize.dual_annealing(f, bounds=bounds, maxiter=10000)
print(result)
