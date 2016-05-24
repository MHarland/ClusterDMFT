from pytriqs.operators import c as C, c_dag as Cdag

class HubbardQuantumNumbers:
    """
    Sets up N and Sz for the cluster
    """
    def __init__(self, g_transf_struct, *args, **kwargs):
        self.g_struct = g_transf_struct
        self.N = None
        self.Sz = None
        self._set_qnrs()

    def update_parameters(self, parameters):
        parameters["partition_method"] = "quantum_numbers"
        parameters["quantum_numbers"].append(self.N)
        parameters["quantum_numbers"].append(self.Sz)
        return parameters

    def _set_qnrs(self):
        self._set_Sz()
        self._set_N()

    def _set_Sz(self):
        self.Sz = 0
        for block in self.g_struct:
            if 'up' in block[0]:
                sign = +1
            else:
                assert 'dn' in block[0] or 'down' in block[0], "Sz quantum number could not be set, due to lack of up/down labels in g_transf_struct"
                sign = -1
            for block_ind in block[1]:
                self.Sz += sign * Cdag(block[0], block_ind) * C(block[0], block_ind)

    def _set_N(self):
        self.N = 0
        for block in self.g_struct:
            for block_ind in block[1]:
                self.N += Cdag(block[0], block_ind) * C(block[0], block_ind)

def sum_list(self, l):
    assert len(l) > 1, "list too short to sum"
    s = l[0]
    for li in l[1:]:
        s += li
    return s
