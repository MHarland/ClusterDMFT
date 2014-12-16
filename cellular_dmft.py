from .lattice.superlatticetools import _init_k_sum

class cellular_dmft(object):
    def __init__(self, lattice_vectors, clustersite_pos, hopping, n_kpts):
        self.k_sum = _init_k_sum(lattice_vectors, clustersite_pos, hopping, n_kpts)

    def g_local(self, sigma, mu):
        return self.k_sum(mu = mu, Sigma = sigma)
