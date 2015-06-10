from pytriqs.gf.local import BlockGf, GfImTime

class Observables(object):
    def __init__(self, g_iw, n_tau = 10001):
        self.giw = g_iw
        self.nTau = n_tau
        self.blockNames = list()
        for s, b in g_iw:
            self.nOrbs = len(b.data[0, :, :])
            self.blockNames.append(s)
        self.orbs = range(self.nOrbs)
        self.gtau = None

    def __set_g_tau(self):
        self.gtau = BlockGf(name_list = self.blockNames,
                            block_list = [GfImTime(indices = self.orbs,
                                                   beta = self.giw.beta,
                                                   n_points = self.nTau) for s in self.blockNames],
                            make_copies = False)
        for s, b in self.gtau: b.set_from_inverse_fourier(self.giw[s])

    def local_magnetization(self):
        """returns a list of pairs containing orbital name and Sz"""
        sz = list()
        if not self.gtau:
            self.__set_g_tau()
        for i in self.orbs:
            sz.append([i, -self.gtau[self.blockNames[0]].data[-1, i, i] + self.gtau[self.blockNames[1]].data[-1, i, i]])
        return sz

    def total_magnetization(self):
        sz = 0
        if not self.gtau:
            self.__set_g_tau()
        for i in self.orbs:
            sz += -self.gtau[self.blockNames[0]].data[-1, i, i] + self.gtau[self.blockNames[1]].data[-1, i, i]
        return sz
        
    def occupation(self):
        """returns a list of pairs, of which the first entry is again a pair of blockname, orbital"""
        occ = list()
        if not self.gtau:
            self.__set_g_tau()
        for s, b in self.gtau:
            for i in self.orbs:
                occ.append([[s, i], -b.data[-1, i, i]])
        return occ

    def total_occupation(self):
        occ = 0
        if not self.gtau:
            self.__set_g_tau()
        for s, b in self.gtau:
            for i in self.orbs:
                occ += -b.data[-1, i, i]
        return occ
