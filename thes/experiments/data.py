from thes.tools import snippets
from thes.data.external_dataset import DatasetDALY, DatasetVOC2007


def precompute_cache(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset: [~, ['daly', 'voc2007']]
    """)
    cf = cfg.parse()

    if cf['dataset'] == 'daly':
        dataset = DatasetDALY()
    elif cf['dataset'] == 'voc2007':
        dataset = DatasetVOC2007()
    else:
        raise RuntimeError('Wrong dataset')
    dataset.precompute_to_folder(out)
