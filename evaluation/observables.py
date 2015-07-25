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

    def dosAtFermiLevel(self):
        """returns a list of pairs, of which the first entry is again a pair of blockname, orbital"""
        dos = list()
        if not self.gtau:
            self.__set_g_tau()
        for s, b in self.gtau:
            for i in self.orbs:
                dos.append([[s, i], -1 * b.beta * b.data[int(len(b.data[:, 0, 0])*.5)+1, i, i]])
        return dos

    def totDosAtFermiLevel(self):
        dos = 0
        if not self.gtau:
            self.__set_g_tau()
        for s, b in self.gtau:
            for i in self.orbs:
                dos += -1 * b.beta * b.data[int(len(b.data[:, 0, 0])*.5)+1, i, i]
        return dos

    def charge_order(self, symClasses = [[],[]]):
        if not self.gtau:
            self.__set_g_tau()
        occ = self.occupation()
        assert len(self.blockNames) == 2, 'Gf\'s basis must be spins_sites'
        nSites = len(occ)/2
        trSpinOcc = [0] * nSites
        for i in range(nSites):
            trSpinOcc[i] += (occ[i][1] + occ[nSites+i][1]) * .5
        trSpinSymOcc = [0, 0]
        for i, symClass in enumerate(symClasses):
            for el in symClass:
                trSpinSymOcc[i] += trSpinOcc[el]
            trSpinSymOcc[i] = trSpinSymOcc[i] /float(len(symClass))
        return abs(trSpinSymOcc[1] - trSpinSymOcc[0])
