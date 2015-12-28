from ClusterDMFT.lattice.superlattices import l_dim as my_superlattice

sl = my_superlattice()
p = dict()

p['u_hubbard'] = 6 # coulomb
p['t'] = sl.get_hopping()
p['beta'] = 20
p['density'] = False
p['n_kpts'] = 32
p['mix_coeff'] = 1
p['mu'] = 3
p['cluster_lattice'] = sl.get_cartesian_clusterlatticevectors()
p['cluster'] = sl.get_clusterlatticebasis()

p['impose_paramagnetism'] = False
p['transformation'] = sl.get_symmetry_transformation()
p['g_transf_struct'] = sl.get_g_transf_struct_orbital()
p['scheme'] = 'cellular_dmft'
p["verbosity"] = 2 # 2 makes plots of intermediate steps

p["max_time"] = -1
p["length_cycle"] = 50
p["n_warmup_cycles"] = 5 * 10**4
p["n_cycles"] = int(10**6 *.5)
p['n_iw'] = 1025
p['n_tau'] = 10001
p['n_legendre'] = 30
p['make_histograms'] = False
p['measure_g_tau'] = True# use tail-fit options if g_l is not measured
p['fit_tail'] = True
p['tail_start'] = 50
p['measure_g_l'] = False
#p['site_symmetries'] =[[(0,0),(1,1)],[(0,1),(1,0)]]
#p['clipping_threshold'] = 0 # not ready
p['use_trace_estimator'] = False
p['measure_pert_order'] = False
p['measure_density_matrix'] = False
p['use_norm_as_weight'] = True
