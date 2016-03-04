from matplotlib import pyplot as plt, rc
from numpy import empty, log
from pytriqs.plot.mpl_interface import oplot
from pytriqs.archive import HDFArchive

def plot_of_loops_from_archive(archive, function, matsubara_freqs = [0], blocks = ['up'], indices = [(0, 0)], RI = str(), **kwargs):
    arch = HDFArchive(archive, 'r')
    n_loops = arch['results']['n_dmft_loops']

    rc('font', size = 10)
    rc('legend', fontsize = 10)
    fig, ax = plt.subplots()

    if 'iw' in function:
        re_f_of_l = empty(n_loops)
        im_f_of_l = empty(n_loops)
        for s in blocks:
            for i in indices:
                f_name = function + '_' + s + '_' + str(i[0]) + str(i[1])
                for mats_nr in matsubara_freqs:
                    for l in range(n_loops):
                        g = arch['results'][str(l)][function][s]
                        freqs = [w for w in g.mesh]
                        nw = len(freqs)
                        re_f_of_l[l] = float(g[i].data[mats_nr+int(nw*.5)].real)
                        im_f_of_l[l] = float(g[i].data[mats_nr+int(nw*.5)].imag)
                    if RI != 'R':
                        ax.plot(im_f_of_l, label = 'Im' + f_name + '_at_ w' + str(mats_nr), **kwargs)
                    if RI != 'I':
                        ax.plot(re_f_of_l, label = 'Re' + f_name + '_at_ w' + str(mats_nr), **kwargs)    
    if 'mu' == function:
        mu = empty(n_loops)
        for l in range(n_loops):
            mu[l] = arch['results'][str(l)][function]
        ax.plot(mu, label = '$\mu$', **kwargs)
    if 'dmu' == function:
        dmu = empty(n_loops)
        for l in range(n_loops):
            dmu[l] = arch['results'][str(l)][function]
        ax.plot(dmu, label = '$\\tilde{\\mu}$', **kwargs)
    if 'density' == function:
        density = empty(n_loops)
        for l in range(n_loops):
            density[l] = arch['results'][str(l)][function]
        ax.plot(density, label = 'density', **kwargs)
    if 'sign' == function:
        density = empty(n_loops)
        for l in range(n_loops):
            density[l] = arch['results'][str(l)][function]
        ax.plot(density, label = '<sign>', **kwargs)
        ax.set_ylim(-1.1, 1.1)

    ax.legend(loc = 2)
    plt.xlabel('DMFT loop')
    #plt.ylabel('f')
    del arch

def plot_from_archive(archive, function, loops = [-1], indices = [(0, 0)], blocks = ['up'], **kwargs):
    archive = HDFArchive(archive, 'r')
    for l in loops:
        if l < 0:
            ll = archive['results']['n_dmft_loops'] + l
        else:
            ll = l
        for ind in indices:
            for s in blocks:
                f_name = s + '_' + str(ind[0]) + str(ind[1]) + '_it' + str(ll)
                if 'raw' in function: f_name += '_raw'
                f = archive['results'][str(ll)][function]
                if 'iw' in function:
                    if 'RI' in kwargs.keys():
                        if kwargs['RI'] == 'R':
                            oplot(f[s][ind], name = 'Re_' + f_name, **kwargs)
                        if kwargs['RI'] == 'I':
                            oplot(f[s][ind], name = 'Im_' + f_name, **kwargs)
                    else:
                        oplot(f[s][ind], name = 'Re_' + f_name, RI = 'R', **kwargs)
                        oplot(f[s][ind], name = 'Im_' + f_name, RI = 'I', **kwargs)
                elif function == 'g_transf_l':
                    plt.plot(f[s].data[:, ind[0], ind[1]], label = f_name, **kwargs)
                    plt.xlabel('$l_n$')
                    plt.ylabel('$\\tilde{G}(l_n)$')
                else:
                    oplot(f[s][ind], name = f_name, **kwargs)
    y_ax_lab = '$'
    if 'transf' in function: y_ax_lab += '\\tilde{'
    if 'sigma' in function: y_ax_lab += '\\Sigma'
    elif 'g' in function: y_ax_lab += 'G'
    elif 'delta' in function: y_ax_lab += '\\Delta'
    else: y_ax_lab += function
    if 'transf' in function: y_ax_lab += '}'
    y_ax_lab += "("
    if 'iw' in function: y_ax_lab += 'i\\omega_n'
    elif 'tau' in function: y_ax_lab += '\\tau'
    y_ax_lab += ')$'
    plt.gca().set_ylabel(y_ax_lab)
    del archive

def checksym_plot(g, fname, **kwargs):
    plt.figure(1)
    for s, b in g:
        mesh = list()
        for iw in b.mesh:
            mesh.append(iw.imag)
        for i in range(len(b.data[0, :, :])):
            for j in range(len(b.data[0, :, :])):
                if i == j:
                    m = '+'
                else:
                    m = 'x'
                plt.subplot(2, 1, 1)
                plt.gca().set_color_cycle(['b'])
                plt.plot(mesh, b.data[:, i, j].imag, label = s + '_' + str(i) + str(j), marker = m, **kwargs)
                plt.subplot(2, 1, 2)
                plt.gca().set_color_cycle(['b'])
                plt.plot(mesh, b.data[:, i, j].real, label = s + '_' + str(i) + str(j), marker = m, **kwargs)

    plt.subplot(2, 1, 1)
    plt.legend(prop = {'size' : 'small'})
    plt.gca().set_xlabel('$i\omega_n$')
    plt.gca().set_ylabel('$\mathrm{Im}G(i\omega_n)$')
    plt.gca().set_xlim(0, 40)

    plt.subplot(2, 1, 2)
    plt.legend(prop = {'size' : 'small'})
    plt.gca().set_xlabel('$i\omega_n$')
    plt.gca().set_ylabel('$\mathrm{Re}G(i\omega_n)$')
    plt.gca().set_xlim(0, 40)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def checktransf_plot(g, fname, xmax = 40, orbitals = None, **kwargs):
    plt.figure(1)
    for s, b in g:
        if orbitals != None:
            if not s in orbitals:
                continue
        mesh = [w for w in b.mesh]
        if mesh[-1].real > mesh[-1].imag:
            mesh = [w.real for w in b.mesh]
        else:
            mesh = [w.imag for w in b.mesh]
        for i in range(len(b.data[0, :, :])):
            for j in range(len(b.data[0, :, :])):
                if i == j:
                    m = '+'
                else:
                    m = 'x'
                plt.subplot(2, 1, 1)
                plt.plot(mesh, b.data[:, i, j].imag, label = s + '_' + str(i) + str(j))#, marker = m, **kwargs)
                plt.subplot(2, 1, 2)
                plt.plot(mesh, b.data[:, i, j].real, label = s + '_' + str(i) + str(j))#, marker = m, **kwargs)

    plt.subplot(2, 1, 1)
    plt.legend(prop = {'size' : 'small'})
    plt.gca().set_xlabel('$i\omega_n$')
    plt.gca().set_ylabel('$\mathrm{Im}G(i\omega_n)$')
    plt.gca().set_xlim(0, xmax)

    plt.subplot(2, 1, 2)
    plt.legend(prop = {'size' : 'small'})
    plt.gca().set_xlabel('$i\omega_n$')
    plt.gca().set_ylabel('$\mathrm{Re}G(i\omega_n)$')
    plt.gca().set_xlim(0, xmax)

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def plot_ln_abs(gl, *args, **kwargs):
    g = gl.copy()
    for s, b in gl:
        for i in range(len(b.data[0, :, :])):
            for j in range(len(b.data[0, :, :])):
                for n in range(len(b.data[:, 0, 0])):
                    g[s].data[n, i, j] = log(abs(b.data[n, i, j]))
    for s, b in g:
        for i in b.indices:
            for j in b.indices:
                oplot(b[i, j], name = s+'_'+str(i)+str(j), *args, **kwargs)
    plt.gca().set_ylabel('$\\mathrm{ln}\\,\\mathrm{abs}\\,\\tilde{G}(l)$')
