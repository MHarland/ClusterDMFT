from pytriqs.archive import HDFArchive
from pytriqs.gf.local import BlockGf, GfImFreq, GfReFreq, GfImTime

def dict_to_archive(dic, archive, group_path):
    """
    dic: dict, archive: str, group_name: str, group_path: list of str
    Written especially for dicts that have non str-type keys
    Not written for dicts of dicts
    """
    a = HDFArchive(archive, 'a')
    a_g = a
    for group in group_path:
        if not a_g.is_group(group):
            a_g.create_group(group)
        a_g = a_g[group]
    for key, val in dic.items():
        if not a_g.is_group(str(key)):
            a_g.create_group(str(key))
        a_g[str(key)]['key'] = key
        a_g[str(key)]['val'] = val
    del a

def archive_to_dict(archive, group_path):
    """
    archive: str, group_name: str, group_path: list of str
    Written especially for dicts that have non str-type keys
    Not written for dicts of dicts
    """
    dic = dict()
    a = HDFArchive(archive, 'r')
    a_g = a
    for group in group_path:
        assert a_g.is_group(group), group + ' could not be found in ' + archive
        a_g = a_g[group]
    for val in a_g.values():
        try:
            dic.update({val['key'] : val['val']})
        except KeyError:
            dic.update({val['R'] : val['t']}) # TODO del, keep try case only
    del a
    return dic

def archive_content(archive, group = list(), dont_exp = list(), n_max_subgroups = 20, shift_step_len = 10):
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

def load_sym_indices(archive, loop_nr):
    arch = HDFArchive(archive, 'r')
    if loop_nr < 0:
        l = arch['Results']['n_dmft_loops'] + loop_nr
    else:
        l = loop_nr

    inds = arch['Results'][str(l)]['sym_indices']
    del arch
    return inds
