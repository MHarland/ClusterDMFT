from cdmft.lattice.superlattices import l_dim as my_superlattice

sl = my_superlattice()

p = dict()
p['u'] = 10
p['t'] = -1
p['beta'] = 10
p['density'] = False
p['n_kpts'] = 32
p['mix_coeff'] = 1
p['mu'] = 0
p['lattice_vectors'] = sl.get_cartesian_superlatticevectors()
p['hop'] = sl.get_h_eff()
p['clustersite_pos'] = sl.get_superlatticebasis()
p['impose_paramagnetism'] = True
p["max_time"] = -1
p["verbosity"] = 2
p["length_cycle"] = 25
p["n_warmup_cycles"] = 5000
p["n_cycles"] = 100000
p['n_iw'] = 1025
p['n_tau'] = 10001
p['n_legendre'] = 20
p['make_histograms'] = False
p['symmetry_transformation'] = sl.get_symmetry_transformation()
p['measure_g_tau'] = False
p['measure_g_l'] = True
p['clipping_threshold'] = 0.01
p['use_trace_estimator'] = False
p['measure_pert_order'] = False
