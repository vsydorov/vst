import logging
import argparse

from thes.tools import snippets


log = logging.getLogger(__name__)


def _set_cfg_defaults_d2dalyobj(cfg):
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    num_gpus: [~, int]
    """)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
    """)
    cfg.set_defaults(D2DICT_GPU_SCALING_DEFAULTS)
    cfg.set_deftype("""
    d2:
        SOLVER.CHECKPOINT_PERIOD: [2500, int]
        TEST.EVAL_PERIOD: [0, int]
        SEED: [42, int]
        # ... anything ...
    """)


def _set_cfg_defaults_pfadet_train(cfg):
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    num_gpus: [~, int]
    """)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', ['train', 'test']]
    """)
    cfg.set_defaults(D2DICT_GPU_SCALING_DEFAULTS)
    cfg.set_deftype("""
    d2:
        SOLVER.CHECKPOINT_PERIOD: [2500, int]
        TEST.EVAL_PERIOD: [0, int]
        SEED: [42, int]
        # ... anything ...
    """)


def train_daly_action(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_cfg_defaults_pfadet_train(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)

    cls_names = dataset.action_names
    num_classes = len(cls_names)
    TRAIN_DATASET_NAME = 'daly_pfadet_train'

    d_cfg = _set_d2config_pfadet(cf, cf_add_d2, TRAIN_DATASET_NAME)
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    d_cfg.freeze()

    num_gpus = cf['num_gpus']

    nargs = argparse.Namespace()
    nargs.datalist = datalist
    nargs.TRAIN_DATASET_NAME = TRAIN_DATASET_NAME
    nargs.cls_names = cls_names
    nargs.resume = True

    dist_url = _figure_out_disturl(add_args)
    launch_w_logging(_train_func_pfadet,
            num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url=dist_url,
            args=(d_cfg, cf, nargs))


def train_daly_object(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_cfg_defaults_d2dalyobj(cfg)
    cfg.set_deftype("""
    hacks:
        dataset: ['normal', ['normal', 'o100', 'action_object']]
        action_object:
            merge: ['sane', ['sane',]]
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    o100_objects, category_map = get_category_map_o100(dataset)
    assert len(o100_objects) == 16

    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = simplest_daly_to_datalist(dataset, split)
        datalist_per_split[split] = datalist

    num_classes, object_names, datalist_converter = \
            _datalist_hacky_converter(cf, dataset)
    datalist_per_split = {k: datalist_converter(datalist)
            for k, datalist in datalist_per_split.items()}

    d_cfg = _set_d2config(cf, cf_add_d2)
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    d_cfg.freeze()

    num_gpus = cf['num_gpus']
    _d2_train_boring_launch(
            object_names, datalist_per_split,
            num_gpus, d_cfg, cf, add_args)
