from numpy import array, zeros

def m_transform(m, u, i, j):
    """
    returns u.m.u_dag
    """
    return sum_list([sum_list([u[i, k] * m[k, l] * u[j, l].conjugate() for k in range(len(m))]) for l in range(len(m))])

def m_transformed(m, u):
    umu = m.copy()
    indices = range(len(m))
    for i in indices:
        for j in indices:
            umu[i, j] = m_transform(m, u, i, j)
    return umu

def energy_loc_sym(hop_loc, u, sym_inds):
    u = array(u)
    hop_loc = array(hop_loc)
    h_sym = dict()
    n_sites = len(u)
    h_sym_old_space = m_transformed(hop_loc, u)
    n_blocks = len(sym_inds)
    b_pos = 0
    for block_inds in sym_inds:
        blockname = block_inds[0]
        block_len = len(block_inds[1])
        block = zeros([block_len, block_len])
        if b_pos == n_sites: b_pos = 0
        for i in block_inds[1]:
            for j in block_inds[1]:
                block[i, j] = h_sym_old_space[b_pos + i, b_pos + j]
        h_sym.update({blockname : block})
        b_pos += len(block_inds[1])
    return h_sym

def v_transform(v, u, i):
    """
    returns u.v
    """
    return sum_list([u[i, j] * v[j] for j in range(len(v))])

def sum_list(list0):
    assert type(list0) == list, 'Parameter is not a list'
    if list0:
        x = list0.pop(0)
        for i in list0:
            x = x + i
        return x
    else:
        return 0

def delta(x, y):
    if x == y: return 1
    return 0
