from cdmft.post_process_g import clip_g
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq

def pade_tr(blockgf_iw, pade_n_omega_n, pade_eta, dos_n_points, dos_window, clip_threshold=0, n_points=1200):
    g_iw = blockgf_iw
    tr_spin_g = GfImFreq(indices = range(len(g_iw['up'].data[0, :, :])), mesh = g_iw.mesh)
    tr_spin_g << g_iw['up'] + g_iw['down']
    tr_band_g = GfImFreq(indices = [0], mesh = g_iw.mesh)
    tr_band_g.zero()
    _temp = tr_band_g.copy()
    for i in range(len(tr_spin_g.data[0, :, :])):
        tr_band_g << _temp + tr_spin_g[i, i]
        _temp << tr_band_g
    del _temp

    tr_g = tr_band_g
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
