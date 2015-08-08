from cdmft.post_process_g import clip_g
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq

def pade_tr(blockgf_iw, pade_n_omega_n, pade_eta, dos_n_points, dos_window, clip_threshold=0, n_points=1200):
    g_iw = blockgf_iw
    indices = [ind for ind in g_iw.indices]
    tr_g = GfImFreq(indices = [0], mesh = g_iw[indices[0]].mesh)
    for s, b in g_iw:
        for i in range(len(b.data[0, :, :])):
            tr_g << tr_g + b[i, i]
    tr_g_lat_pade = GfReFreq(indices = [0], window = dos_window,
                             n_points = dos_n_points, name = 'Tr$G_{lat}$')
    tr_g_lat_pade.set_from_pade(clip_g(tr_g, clip_threshold), n_points = pade_n_omega_n,
                                freq_offset = pade_eta)
    return tr_g_lat_pade

def pade_spin(blockgf_iw, spin, pade_n_omega_n, pade_eta, dos_n_points, dos_window, clip_threshold=0, n_points=1200):
    g_iw = blockgf_iw[spin]
    indices = g_iw.indices
    g_pade = GfReFreq(indices = indices, window = dos_window, n_points = dos_n_points, name = 'Tr$G_{lat}$')
    g_pade.set_from_pade(g_iw, n_points = pade_n_omega_n, freq_offset = pade_eta)
    return g_pade
