from numpy import empty, identity, array, zeros

from .other import sum_list, delta

class CoulombTensor(object):

    def __init__(self, u_int, dimension):
        self.u_int = u_int
        self.d = dimension
        self.data = zeros([self.d, self.d, self.d, self.d, 2, 2], dtype = float)
        for i in range(self.d):
            for j in range(self.d):
                for k in range(self.d):
                    for l in range(self.d):
                        for s1 in range(2):
                            for s2 in range(2):
                                self.data[i, j, k, l, s1, s2] = u_int * delta(i, k) * delta(i, l) * delta(i, j) * (1 - delta(s1, s2)) *.5

    def __getitem__(self, key):
        key = list(key)
        for s in [4, 5]:
            if key[s] == 'up':
                key[s] = 0
            elif key[s] == 'down':
                key[s] = 1
            else:
                print 'CoulombTensor did not recognise the spinname'
        return self.data[key[0], key[1], key[2], key[3], key[4], key[5]]

    def show(self, len_per_entry = 4):
        for s1, ss1 in enumerate(['up', 'down']):
            for s2, ss2 in enumerate(['up', 'down']):
                print '(' + ss1 + ', ' + ss2 + ')'
                for i in range(self.d):
                    for j in range(self.d):
                        line = str()
                        for k in range(self.d):
                            for l in range(self.d):
                                x = self.data[i, j, k, l, s1, s2]
                                if x > 0:
                                    c = '+' + str(x)[0:len_per_entry - 1]
                                else:
                                    c = str(x)[0:len_per_entry]
                                while len(c) < len_per_entry:
                                    c += '0'
                                line += c + ' '
                            line += '  '
                        print line
                    print

    def transform(self, u):
        assert len(u) == self.d, 'u must have the same dimension as the supermatrixelements'
        u = array(u)
        udag = u.T.conjugate()
        result = empty([self.d, self.d, self.d, self.d, 2, 2], dtype = float)
        for s1 in range(2):
            for s2 in range(2):
                for i in range(self.d):
                    for j in range(self.d):
                        for k in range(self.d):
                            for l in range(self.d):
                                result[i, j, k, l, s1, s2] = sum_list([sum_list([sum_list([sum_list([self.data[m, n, o, p, s1, s2] * u[i, m] * u[j, n] * u[l, p].conjugate() * u[k, o].conjugate() for m in range(self.d)]) for n in range(self.d)]) for o in range(self.d)]) for p in range(self.d)])

        self.data = result
        return self

class NNCoulombTensor(CoulombTensor):

    def __init__(self, u_int, dimension):
        self.u_int = array(u_int)
        self.d = dimension
        self.data = zeros([self.d, self.d, self.d, self.d, 2, 2], dtype = float)
        for i in range(self.d):
            for j in range(self.d):
                for k in range(self.d):
                    for l in range(self.d):
                        for s1 in range(2):
                            for s2 in range(2):
                                self.data[i, j, k, l, s1, s2] = self.u_int[i, j] * delta(i, k) * delta(j, l) * .5
