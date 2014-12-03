from numpy import array

def m_transform(m, u, i, j):
    """
    returns u.m.u_dag
    """
    return sum_list([sum_list([u[i, k] * m[k, l] * u[j, l].conjugate() for k in range(len(m))]) for l in range(len(m))])

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
