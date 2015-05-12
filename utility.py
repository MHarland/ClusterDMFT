def get_n_sites(parameters):
    return len(parameters['cluster'])

def get_site_nrs(parameters):
    return range(get_n_sites(parameters))

def get_dim(parameters):
    return len(parameters['t'].keys()[0])
