import scipy
import geom
import scipy
import numpy
import time as timer
import exp

# Saves the Hamiltonian matrix for the Christandl hoppings.
# To save to a file, run something like:
#   python christandl_save.py > matrix.txt

# Implements the Christandl hoppings to take an electron down a chain.
# For example purposes, an up and down electron are taken down a chain,
# since the interaction U = 0, each electron should move independently.

# 1 --t1-- 2 --t2-- 3 --t3-- 4 -- ... -- tn-1 -- n

n = 6
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

for i in range(1, n):
    L.sym_couple(i, i+1, f"t{i}")
for i in range(1, n+1):
    L.interaction(i, f"U{i}")

H = geom.Hamiltonian()

for i in range(1, n):
    H.ts[f"t{i}"] = numpy.sqrt(i * (n-i))
for i in range(1, n+1):
    H.Us[f"U{i}"] = U

H.construct(L, B)

#
# Prints the Hamiltonian matrix in the following format:
#
# Off diagonal elements (printed first):
#   i j factor value
#   if H is the Hamiltonian matrix, this means:
#       H[i, j] = factor * value
#   since the Hamiltonian is allowed to be non Hermitian (though in most
#   cases it is Hermitian), all nonzero entries are printed.
#
# Diagonal elements (printed second):
#   i value
#   if H is the Hamiltonian matrix, this means:
#       H[i, i] = 0
#       H[i, i] = H[i, i] + value
#   the H[i, i] = 0 part means that the ith diagonal entry is first initialized
#   as 0, but as elements are read in, it is increased by factor each time.
#
# The ishift argument says how much to shift indices by (by default 1-based indexing
# is used).
# The values (True or False) argument says whether to print the actual hopping/interaction
# strength values, or a string corresponding for each unique type (so many Hamiltonians
# can be made in code without having to determine from scratch where they go each time).
#
ishift = -1 # For C/Python indexing
H.print_matrix(ishift=ishift, values=True)

#
# This will print the rules for putting together the Hamiltonian matrix H
# for time evolution. To carry out time evolution, a little more information is
# required.
#
# The time evolution operator is A = exp(-1.0j * t * H),
# where 1.0j is the imaginary number sqrt(-1) and t is the time being evolved to.
# Christandl hoppings were engineered for perfect quantum state transfer at
# t = pi/2, but you should also find that at t = pi the state goes right back to
# the start, and back again to the other side at t = pi + pi/2, and so on.
#
# The initial state is a B.N long vector of 0's with a 1 in its
# initial_index + ishift entry. To print this, you can run:
# print(initial_index + ishift)
#
# Time evolution is done by the matrix-vector multiplication:
# final_vector = A @ initial_vector
#
# The modulus squared of each entry of final_vector gives the probability
# of finding the system in the state corresponding to that entry.
# In this particular case, the Christandl hoppings promise that
# if we examine the entry:
# final_index + ishift
# of final_vector and multiply it by its complex conjugate, we should get 1:
# perfect quantum state transfer.
#
# See some of the setup in this code and christandl.py for more details.
#