from pytriqs.utility import mpi

def get_n_sites(parameters):
    return len(parameters['cluster'])

def get_site_nrs(parameters):
    return range(get_n_sites(parameters))

def get_dim(parameters):
    return len(parameters['t'].keys()[0])

class Reporter(object):
    def __init__(self, verbosity, **kwargs):
        self.verbosity = verbosity

    def __call__(self, *args):
        if self.verbosity > 0 and mpi.is_master_node():
            for arg in args:
                print arg
