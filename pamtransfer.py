import scipy
import geom
import scipy
import numpy
import time as timer
import exp


n = 8
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
final_state = geom.State(ups=(4,), dns=(4,))
initial_index = B.index(initial_state)
final_index = B.index(final_state)
psi0 = numpy.zeros(B.N)
psi0[initial_index-1] = 1


L.sym_couple(1, 2, "t1")
L.sym_couple(2, 3, "t2")
L.sym_couple(3, 4, "t3")

L.sym_couple(1, 5, "v1")
L.sym_couple(2, 6, "v2")
L.sym_couple(3, 7, "v3")
L.sym_couple(4, 8, "v4")

L.interaction(1, "Uc")
L.interaction(2, "Uc")
L.interaction(3, "Uc")
L.interaction(4, "Uc")

L.interaction(5, "Ud")
L.interaction(6, "Ud")
L.interaction(7, "Ud")
L.interaction(8, "Ud")


H = geom.Hamiltonian()

H.ts = {"t1": 418.87,
        "t2": 1188.11,
        "t3": 419.87,
        "v1": 478.51,
        "v2": 1292.36,
        "v3": 1292.36,
        "v4": 478.51}

H.Us = {"Uc": 13683.6,
        "Ud": 4014.55}


start = timer.time()
H.construct(L, B)
finish = timer.time()
print(f"H constructed in {finish-start} seconds")



start = timer.time()
H.make_H()
finish = timer.time()
print(f"Matrix made in {finish-start} seconds")

start = timer.time()
psi1 = exp.expA_v(H.H, psi0, -1.0j * time, "he_diag", traceA=H.trace())
finish = timer.time()
print(f"Exponential-vector product done in {finish-start} seconds")

final_probability = abs(psi1[final_index-1]) ** 2
print(f"Fidelity = {final_probability}")

