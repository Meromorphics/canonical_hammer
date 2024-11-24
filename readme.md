Generates a Hubbard model Hamiltonian in the canonical ensemble, given (potentially asymmetric) hoppings and an interaction energy.

This code originated as a test on what Graph data structure things I would have to implement in a Fortran to get the same functionality there, but it ended up proving quite useful.

The main file is geom.py. Note that sites are indexed starting at 1.

Dependencies: networkx, numpy, scipy.
Some examples use matplotlib.
By default the exp module requires pytorch and tensorflow as well, but these parts can be deleted or commented out (they are there for testing purposes).

TODO:
Implement chemical potential (*should* be somewhat easy).
Make documentation.
Organize code.
Make Fortran io (take Hamiltonian/graph information printed from Python code here and put it in Fortran).
Make better examples.
