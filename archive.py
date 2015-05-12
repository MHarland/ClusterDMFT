import os
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq, GfImTime

class ArchiveConnected(object):
    """
    allows extraction of data from the associated HDFArchive
    """
    def __init__(self, archive, *args, **kwargs):
        self.archive = archive
        if not os.path.exists(archive):
            archive = HDFArchive(archive, 'w')
            archive.create_group('results')
            archive['results']['n_dmft_loops'] = 0
            del archive

    def next_loop(self):
        """returns the DMFT loop nr. of the next loop"""
        archive = HDFArchive(self.archive, 'r')
        if archive.is_group('results'):
            nl = archive['results']['n_dmft_loops']
        else:
            nl = 0
        del archive
        return nl

    def last_loop(self):
        """returns the last DMFT loop nr."""
        arch = HDFArchive(self.archive, 'r')
        ll = arch['results']['n_dmft_loops'] - 1
        del arch
        return ll

    def load(self, function_name, loop_nr = -1):
        """
        returns a calculated function from archive
        function_name: 'Sigma_c_iw', 'G_c_iw', ...
        loop_nr: int, -1 gives the last loop nr.
        """
        a = HDFArchive(self.archive, 'r')
        if loop_nr < 0:
            function = a['results'][str(self.next_loop() + loop_nr)][function_name]
        else:
            function = a['results'][str(loop_nr)][function_name]
        del a
        return function

    def archive_content(self, group = list(), dont_exp = list(), n_max_subgroups = 50, shift_step_len = 10):
        """
        collects and returns the archive\'s content as a string
        can be applied to a specific group of the archive
        certain group's expansion can be omitted
        recursion depth can be set
        indentation for tree structure can be set
        """
        archive = self.archive
        arch = HDFArchive(archive, 'r')
        content = archive + '\n'
        shift = str()
        for i, g in enumerate(group):
            arch = arch[g]
            shift += ' ' * shift_step_len
            content += shift + g + '\n'
        content = _archive_content(arch, content, shift, shift_step_len, dont_exp, n_max_subgroups)
        del arch
        return content

def _archive_content(group, content, shift, shift_step_len, dont_exp, n_max_subgroups):
    for key in group.keys():
        if group.is_data(key):
            if key in dont_exp:
                content += shift + ' ' * shift_step_len + key + '...\n'
            else:
                content += shift + ' ' * shift_step_len + str(key) + ' = ' +  str(group[key]) + '\n'
        else:
            assert group.is_group(key), 'unkown data in archive'
            if len(group[key]) > n_max_subgroups:
                content += shift + ' ' * shift_step_len + str(key) + '...\n'
            elif key in dont_exp:
                content += shift + ' ' * shift_step_len + key + '...\n'
            elif type(group[key]) in [BlockGf, GfImFreq, GfReFreq, list, tuple, dict]:
                content += shift + ' ' * shift_step_len + str(key) + ' = ' +  str(group[key]) + '\n'
            else: 
                content += _archive_content(group[key], shift + ' ' * shift_step_len + str(key) + '\n', shift + ' '  * shift_step_len, shift_step_len, dont_exp, n_max_subgroups)
    return content
