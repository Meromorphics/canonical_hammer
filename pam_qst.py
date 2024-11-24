import scipy
import geom
import scipy
import numpy
import time as timer
import exp


n = 4
nups = 1
ndns = 1
time = numpy.pi / 2
U = 0


start = timer.time()
B = geom.Basis(2*n, nups, ndns)
finish = timer.time()
print(f"Basis made in {finish-start} seconds")
L = geom.Lattice(2*n)



initial_state = geom.State(ups=(1,), dns=(1,))
final_state = geom.State(ups=(n,), dns=(n,))
initial_index = B.index(initial_state)
final_index = B.index(final_state)
psi0 = numpy.zeros(B.N)
psi0[initial_index-1] = 1


for i in range(1, n):
    L.sym_couple(i, i+1, f"t{i}")
for i in range(1, n+1):
    L.sym_couple(i, i+n, f"v{i}")

for i in range(1, n+1):
    L.interaction(i, "Uc")
for i in range(1, n+1):
    L.interaction(i+n, "Ud")


H = geom.Hamiltonian()


def f(x, *args):
    ts = x[:n-1]
    vs = x[n-1:-2]
    Uc = x[-2]
    Ud = x[-1]
    for i, t in enumerate(ts):
        H.ts[f"t{i+1}"] = ts[i]
    for i, v in enumerate(vs):
        H.ts[f"v{i+1}"] = vs[i]
    H.Us["Uc"] = Uc
    H.Us["Ud"] = Ud
    H.make_H(fresh=False)
    psi1 = exp.expA_v(H.H, psi0, -1.0j * time, "he_diag", traceA=H.trace())
    final_probability = abs(psi1[final_index-1]) ** 2
    print(final_probability)
    return -final_probability




start = timer.time()
H.construct(L, B)
finish = timer.time()
print(f"H constructed in {finish-start} seconds")

H.zero_H()
#                           ts                                 vs                          Us
bounds = list(zip([0] * (n-1), [100] * (n-1))) + list(zip([0] * n, [100] * n)) + list(zip([1] * 2, [5] * 2))
#bounds = [(419, 420), (1188, 1189), (419, 420), (478, 479), (1292, 1293), (1292, 1293), (478, 479), (13683, 13684), (4014, 4015)]

print("Starting optimization...")
result = scipy.optimize.dual_annealing(f, bounds=bounds, maxiter=10000)
print(result)
