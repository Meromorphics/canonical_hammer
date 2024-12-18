import networkx
import itertools
import numpy

class State():
    def __init__(self, **kwargs):
        if "index" in kwargs:
            self.set_index(kwargs["index"])
        if "ups" in kwargs:
            self.set_ups(kwargs["ups"])
        if "dns" in kwargs:
            self.set_dns(kwargs["dns"])
    
    def set_index(self, i):
        self.index = i
    
    def set_ups(self, ups):
        self.ups = tuple(sorted([int(u) for u in ups]))
    
    def set_dns(self, dns):
        self.dns = tuple(sorted([int(d) for d in dns]))
    
    def hopped_state(self, spin, old_site, new_site):
        new_tuple = self.__hop(spin, old_site, new_site)
        if spin == "up":
            return State(ups=new_tuple, dns=self.dns)
        elif spin == "dn":
            return State(ups=self.ups, dns=new_tuple)

    def check_validity(self):
        # Ensure the state is an actual valid state: no repeat ups or downs
        if hasattr(self, "ups"):
            upscheck = len(self.ups) == len(set(self.ups))
        else:
            upscheck = True
        if hasattr(self, "dns"):
            dnscheck = len(self.dns) == len(set(self.dns))
        else:
            dnscheck = True

        return upscheck and dnscheck
    
    def __hop(self, spin, old_site, new_site):
        if spin == "up":
            old_tuple = self.ups
        elif spin == "dn":
            old_tuple = self.dns
        new_tuple = list(old_tuple)
        old_index = new_tuple.index(old_site)
        new_tuple[old_index] = new_site
        return tuple(new_tuple)
    
    def __str__(self):
        try:
            str = f"State index: {self.index}\n" + f"        ups: {self.ups}\n" + f"        dns: {self.dns}\n"
        except:
            str = f"        ups: {self.ups}\n" + f"        dns: {self.dns}\n"
        return str
    
    def __eq__(self, state):
        if hasattr(self, "ups"):
            upscheck = self.ups == state.ups
        else:
            upscheck = True
        if hasattr(self, "dns"):
            dnscheck = self.dns == state.dns
        else:
            dnscheck = True
        return upscheck and dnscheck
    
    def hopover_count(self, state, spin):
        if spin == "up":
            old = list(self.ups)
            new = list(state.ups)
        elif spin == "dn":
            old = list(self.dns)
            new = list(state.dns)
        # x, y = site hopped from/to
        x, y = set(old).symmetric_difference(set(new))
        a = min(x, y)
        b = max(x, y)
        # hopover count = number of sites between x and y
        count = len([x for x in old if a < x < b])
        return count

class Lattice():
    def __init__(self, n, zshift=1):
        self.G = networkx.DiGraph()
        self.G.add_nodes_from(range(zshift, n+zshift))

    def couple(self, i, j, tij):
        self.G.add_edge(i, j, tij=tij)

    def sym_couple(self, i, j, t):
        self.couple(i, j, t)
        self.couple(j, i, t)

    def two_couple(self, i, j, tij, tji):
        self.couple(i, j, tij)
        self.couple(j, i, tji)

    def interaction(self, i, U):
        self.G.nodes[i]["U"] = U


class Basis():
    def __init__(self, n, nups, ndns, zshift=1, ishift=1):
        self.zshift = zshift
        self.ishift = ishift
        self.make_basis(n, nups, ndns)
        self.N = len(self.basis)

    def make_basis(self, n, nups, ndns):
        self.basis = []
        i = self.ishift
        for ups in itertools.combinations(range(self.zshift, n+self.zshift), nups):
            for dns in itertools.combinations(range(self.zshift, n+self.zshift), ndns):
                self.basis.append(State(index=i, ups=ups, dns=dns))
                i = i + 1
    
    def __getitem__(self, i):
        return self.basis[i]
    
    def index(self, state):
        for basis_state in self:
            if state == basis_state:
                return basis_state.index
        return None

class Hamiltonian():
    def __init__(self):
        self.ts = dict()
        self.Us = dict()


    def zero_H(self):
        self.H = numpy.zeros((self.n, self.n))


    def print_matrix(self, ishift=0, values=False):
        for x in self.elements:
            y = x[:]
            y[0], y[1] = y[0] + ishift, y[1] + ishift
            if values:
                y[3] = self.ts[y[3]]
            print(*y)
        for x in self.diagelements:
            y = x[:]
            y[0] = y[0] + ishift
            if values:
                y[1] = self.Us[y[1]]
            print(*y)


    def construct(self, L, B, has_U=True):
        self.n = len(B.basis)
        self.ishift = B.ishift
        self.diagcount = 0
        self.elements = []
        self.diagelements = []
        self.diagentries = dict()
        for state in B:
            i = state.index
            if hasattr(state, "ups"):
                for up in state.ups:
                    for _, j, edge_data in L.G.edges(up, data=True):
                        hop_state = state.hopped_state("up", up, j)
                        if (k := B.index(hop_state)) is not None:
                            tij = edge_data["tij"]
                            hopcount = state.hopover_count(hop_state, "up")
                            sign = (-1) * ((-1) ** hopcount)
                            self.elements.append([i, k, sign, tij])

            if hasattr(state, "dns"):
                for dn in state.dns:
                    for _, j, edge_data in L.G.edges(dn, data=True):
                        hop_state = state.hopped_state("dn", dn, j)
                        if (k := B.index(hop_state)) is not None:
                            tij = edge_data["tij"]
                            hopcount = state.hopover_count(hop_state, "dn")
                            sign = (-1) * ((-1) ** hopcount)
                            self.elements.append([i, k, sign, tij])

            # diagonal U entries
            if has_U and hasattr(state, "ups") and hasattr(state, "dns"):
                for j in set(state.ups).intersection(set(state.dns)):
                    self.diagelements.append([i, L.G.nodes[j]["U"]])
                    self.diagentries[i] = 0


    def make_H(self, fresh=True):
        if fresh:
            self.zero_H()
        for i, j, factor, tij in self.elements:
            i = i - self.ishift
            j = j - self.ishift
            self.H[i, j] = factor * self.ts[tij]
        self.diagentries = dict.fromkeys(self.diagentries, 0)
        for i, U in self.diagelements:
            self.diagentries[i] = self.diagentries[i] + self.Us[U]
        for i, d in self.diagentries.items():
            i = i - self.ishift
            self.H[i, i] = d


    def trace(self):
        return sum(self.diagentries.values())
    
    