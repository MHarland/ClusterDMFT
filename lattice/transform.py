from pytriqs.gf.local import BlockGf, GfImFreq, inverse
from numpy import zeros, array
# TODO move transf methods from cdmft here or del in cdmft

_spins = ['up', 'down'] # TODO no global vars

def exact_inverse(g, u):
    assert type(g) == BlockGf, 'g has to be a BlockGf'
    sym_ind = sym_indices(g, u)
    g_s = g_sym(g, u, sym_ind)
    g_s_inv = g_s.copy()
    g_s_inv << inverse(g_s)
    g_inv = g_c(g_s_inv, u, sym_ind)
    return g_inv

def g_sym(g, u, sym_indices):
    global _spins
    u = array(u)
    udag = u.T.conjugate()
    g_sym = BlockGf(name_block_generator = [(ind, GfImFreq(indices = dict(sym_indices)[ind], mesh = g[_spins[0]].mesh)) for ind in dict(sym_indices).keys()], make_copies = False)

    for ss in _spins:
        i = 0
        for s in [sym_indices[ii][0] for ii in range(len(sym_indices))]:
            if ss in s:
                for ib in dict(sym_indices)[s]:
                    for jb in dict(sym_indices)[s]:
                        g_sym[s][ib, jb] << sum_list([sum_list([u[i + ib, k] * g[ss][k, l] * udag[l, i + jb] for k in range(len(u))]) for l in range(len(u))])
                i += 1
    return g_sym

def g_c(g, u, sym_indices):
    global _spins
    u = array(u)
    udag = u.T.conjugate()
    sites = range(len(u))
    g_c = BlockGf(name_block_generator = [(s, GfImFreq(indices = sites, mesh = g[sym_indices[0][0]].mesh)) for s in _spins], make_copies = False)

    for s in _spins:
        for i in sites:
            for j in sites:
                g_c[s][i, j] << sum_list([sum_list([udag[i, k] * _unblocked_g_sym(g, s, k, l, sym_indices) * u[l, j]  for k in sites]) for l in sites])
    return g_c

def _unblocked_g_sym(g, s, row, col, sym_indices):
    i0 = row if row <= col else col
    i1 = row if row > col else col
    ordered_keys = [sym_indices[ii][0] for ii in range(len(sym_indices))]
    j = 0

    for key in ordered_keys:
        if s in key:
            for k in dict(sym_indices)[key]:
                if i0 == j:
                    (block, ind1) = (key, k)
                j += 1

    if len(dict(sym_indices)[block]) > ind1 + i1 - i0:
        if row <= col:
            return g[block][ind1, ind1 + i1 - i0]
        else:
            return g[block][ind1 + i1 - i0, ind1]
    else:
        return 0

def sym_indices(g_weiss_iw, u):
    """
    Finds the new blockstructure of G_loc using the unitary transformation u.
    """
    global _spins
    almost_zero = 10e-12
    n_sites = len(u)
    sites = range(n_sites)
    g_block_structure = zeros([n_sites, n_sites], int)
    u = array(u)
    udag = u.T.conjugate()
    blockdiag_len = list()
    _blockdiag_len = 0

    for s, b in g_weiss_iw:
        for i in sites:
            for l in sites:
                for m in range(40): # len(b.data[:,0,0])
                    if abs(sum_list([sum_list([u[i, j] * b[j, k] * udag[k, l] for j in sites]) for k in sites]).data[m, 0, 0]) > almost_zero:
                        g_block_structure[i, l] = 1

    for i in sites:
        block_is_zero = True
        for j in range(i, n_sites, 1):
            for k in range(0, i + 1, 1):
                if i != j and (g_block_structure[i - k, j] != 0 or g_block_structure[j, i - k] != 0):
                    block_is_zero = False
        if block_is_zero:
            blockdiag_len.append(_blockdiag_len)
        else:
            _blockdiag_len += 1

    return [(str(i) + '-' + s, range(blockdiag_len[i] + 1)) for s in _spins for i in range(len(blockdiag_len))]

def sum_list(list0):
    assert type(list0) == list, 'Parameter is not a list'
    if list0:
        x = list0.pop(0)
        for i in list0:
            x = x + i
        return x
    else:
        return 0
