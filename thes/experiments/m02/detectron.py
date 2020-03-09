import os
import logging
import argparse
from pathlib import Path

import torch

from detectron2.data import (
        DatasetCatalog, MetadataCatalog,)
from detectron2.engine.defaults import DefaultPredictor
from detectron2.layers.nms import nms, batched_nms

from vsydorov_tools import small

from thes.tools import snippets
from thes.data.external_dataset import (
        DatasetDALY)
from thes.detectron.cfg import (
        D2DICT_GPU_SCALING_DEFAULTS,
        d2dict_gpu_scaling,
        set_detectron_cfg_base,
        set_detectron_cfg_train_and_freeze,
        set_detectron_cfg_test_and_freeze,
        )
from thes.detectron.internals import launch_w_logging
from thes.detectron.externals import (
        simple_d2_setup, DalyVideoDatasetTrainer,
        get_frame_without_crashing)
from thes.detectron.daly import (
        get_daly_split_vids, simplest_daly_to_datalist_v2,
        daly_to_datalist_pfadet, get_category_map_o100, make_datalist_o100,
        get_similar_action_objects_DALY, make_datalist_objaction_similar_merged)
from thes.eval_tools import legacy_evaluation


log = logging.getLogger(__name__)


def _set_defcfg_detectron(cfg):
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
    """)
    cfg.set_typecheck(""" dataset.subset: ['train', 'test'] """)


def _set_defcfg_detectron_train(cfg):
    cfg.set_defaults(""" dataset.subset: 'train' """)
    cfg.set_deftype("""
    num_gpus: [~, int]
    d2:
        SOLVER.CHECKPOINT_PERIOD: [2500, int]
        TEST.EVAL_PERIOD: [0, int]
        # ... anything ...
    """)
    cfg.set_defaults(D2DICT_GPU_SCALING_DEFAULTS)


def _set_defcfg_detectron_test(cfg):
    cfg.set_defaults(""" dataset.subset: 'test' """)
    cfg.set_deftype("""
    conf_thresh: [0.0, float]
    nms:
        enable: [True, bool]
        batched: [False, bool]
        thresh: [0.3, float]
    """)


def _set_defcfg_object_hacks(cfg):
    cfg.set_deftype("""
    object_hacks:
        dataset: ['normal', ['normal', 'o100', 'action_object']]
        action_object:
            merge: ['sane', ['sane',]]
    """)


def _figure_out_disturl(add_args):
    if '--port_inc' in add_args:
        ind = add_args.index('--port_inc')
        port_inc = int(add_args[ind+1])
    else:
        port_inc = 0
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14 + port_inc
    dist_url = "tcp://127.0.0.1:{}".format(port)
    return dist_url


def _detectron_train_function(d_cfg, cf, nargs):
    simple_d2_setup(d_cfg)

    dataset_name = nargs.TRAIN_DATASET_NAME
    DatasetCatalog.register(dataset_name, lambda: nargs.datalist)
    MetadataCatalog.get(dataset_name).set(
            thing_classes=nargs.cls_names)

    trainer = nargs.trainer(d_cfg)
    trainer.resume_or_load(resume=nargs.resume)
    trainer.train()


def _datalist_hacky_converter(cf, dataset):
    if cf['object_hacks.dataset'] == 'normal':
        num_classes = 43
        object_names = dataset.object_names

        def datalist_converter(datalist):
            return datalist

    elif cf['object_hacks.dataset'] == 'o100':
        o100_objects, category_map = get_category_map_o100(dataset)
        num_classes = len(o100_objects)
        assert len(o100_objects) == 16
        object_names = o100_objects

        def datalist_converter(datalist):
            datalist = make_datalist_o100(datalist, category_map)
            return datalist

    elif cf['object_hacks.dataset'] == 'action_object':
        assert cf['object_hacks.action_object.merge'] == 'sane'
        action_object_to_object = get_similar_action_objects_DALY()
        object_names = sorted([x
            for x in set(list(action_object_to_object.values())) if x])
        num_classes = len(object_names)

        def datalist_converter(datalist):
            datalist = make_datalist_objaction_similar_merged(
                    datalist, dataset.object_names, object_names,
                    action_object_to_object)
            return datalist

    else:
        raise NotImplementedError()
    return num_classes, object_names, datalist_converter


def _train_routine(cf, cf_add_d2, out,
        cls_names, TRAIN_DATASET_NAME,
        datalist, add_args):

    num_classes = len(cls_names)
    d2_output_dir = str(small.mkdir(out/'d2_output'))
    d_cfg = set_detectron_cfg_base(
            d2_output_dir, num_classes, cf['seed'])
    d_cfg = set_detectron_cfg_train_and_freeze(
            d_cfg, TRAIN_DATASET_NAME, cf_add_d2)

    num_gpus = cf['num_gpus']

    nargs = argparse.Namespace()
    nargs.datalist = datalist
    nargs.TRAIN_DATASET_NAME = TRAIN_DATASET_NAME
    nargs.cls_names = cls_names
    nargs.resume = True
    nargs.trainer = DalyVideoDatasetTrainer

    dist_url = _figure_out_disturl(add_args)
    launch_w_logging(_detectron_train_function,
            num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url=dist_url,
            args=(d_cfg, cf, nargs))


def train_daly_action(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_detectron(cfg)
    _set_defcfg_detectron_train(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')
    cf_add_d2 = d2dict_gpu_scaling(cf, cf_add_d2, cf['num_gpus'])

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)

    cls_names = dataset.action_names

    TRAIN_DATASET_NAME = 'daly_pfadet_train'
    _train_routine(cf, cf_add_d2, out,
        cls_names, TRAIN_DATASET_NAME, datalist, add_args)


def train_daly_object(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_detectron(cfg)
    _set_defcfg_detectron_train(cfg)
    _set_defcfg_object_hacks(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')
    cf_add_d2 = d2dict_gpu_scaling(cf, cf_add_d2, cf['num_gpus'])

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    o100_objects, category_map = get_category_map_o100(dataset)
    assert len(o100_objects) == 16
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = simplest_daly_to_datalist_v2(dataset, split_vids)

    num_classes, cls_names, datalist_converter = \
            _datalist_hacky_converter(cf, dataset)
    datalist = datalist_converter(datalist)

    TRAIN_DATASET_NAME = 'daly_objaction_train'
    _train_routine(cf, cf_add_d2, out,
        cls_names, TRAIN_DATASET_NAME, datalist, add_args)


def _eval_foldname_hack(cf):
    if cf['eval_hacks.model_to_eval'] == 'what':
        model_to_eval = cf['what_to_eval']
    elif cf['eval_hacks.model_to_eval'] == 'what+foldname':
        import dervo
        EPATH = dervo.experiment.EXPERIMENT_PATH
        model_name = EPATH.resolve().name
        model_to_eval = str(Path(cf['what_to_eval'])/model_name)
    else:
        raise NotImplementedError('Wrong eval_hacks.model_to_eval')
    return model_to_eval


def _eval_routine(cf, cf_add_d2, out,
        cls_names, TEST_DATASET_NAME,
        datalist, model_to_eval):

    num_classes = len(cls_names)
    d2_output_dir = str(small.mkdir(out/'d2_output'))
    d_cfg = set_detectron_cfg_base(
            d2_output_dir, num_classes, cf['seed'])
    d_cfg = set_detectron_cfg_test_and_freeze(
            d_cfg, TEST_DATASET_NAME,
            model_to_eval, cf['conf_thresh'], cf_add_d2)

    simple_d2_setup(d_cfg)
    predictor = DefaultPredictor(d_cfg)
    cpu_device = torch.device("cpu")

    def eval_func(dl_item):
        video_path = dl_item['video_path']
        frame_number = dl_item['video_frame_number']
        frame_time = dl_item['video_frame_number']
        frame_u8 = get_frame_without_crashing(
            video_path, frame_number, frame_time)
        predictions = predictor(frame_u8)
        cpu_instances = predictions["instances"].to(cpu_device)
        return cpu_instances

    df_isaver = snippets.Simple_isaver(
            small.mkdir(out/'isaver'), datalist, eval_func, '::50')
    predicted_datalist = df_isaver.run()

    if cf['nms.enable']:
        nms_thresh = cf['nms.thresh']
        nmsed_predicted_datalist = []
        for pred_item in predicted_datalist:
            if cf['nms.batched']:
                keep = batched_nms(pred_item.pred_boxes.tensor,
                        pred_item.scores, pred_item.pred_classes, nms_thresh)
            else:
                keep = nms(pred_item.pred_boxes.tensor,
                        pred_item.scores, nms_thresh)
            nmsed_item = pred_item[keep]
            nmsed_predicted_datalist.append(nmsed_item)
        predicted_datalist = nmsed_predicted_datalist
    legacy_evaluation(cls_names, datalist, predicted_datalist)


def eval_daly_action(workfolder, cfg_dict, add_args):
    """
    Evaluation code with hacks
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_detectron(cfg)
    _set_defcfg_detectron_test(cfg)
    cfg.set_deftype("""
    what_to_eval: [~, str]
    eval_hacks:
        model_to_eval: ['what', ['what', 'what+foldname']]
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)
    cls_names = dataset.action_names

    TEST_DATASET_NAME = 'daly_pfadet_test'

    model_to_eval = _eval_foldname_hack(cf)
    _eval_routine(cf, cf_add_d2, out, cls_names,
        TEST_DATASET_NAME, datalist, model_to_eval)


def eval_daly_object(workfolder, cfg_dict, add_args):
    """
    Evaluation code with hacks
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_detectron(cfg)
    _set_defcfg_detectron_test(cfg)
    _set_defcfg_object_hacks(cfg)
    cfg.set_deftype("""
    what_to_eval: [~, str]
    eval_hacks:
        model_to_eval: ['what', ['what', 'what+foldname']]
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = simplest_daly_to_datalist_v2(dataset, split_vids)

    num_classes, cls_names, datalist_converter = \
            _datalist_hacky_converter(cf, dataset)
    datalist = datalist_converter(datalist)

    TEST_DATASET_NAME = 'daly_objaction_test'

    model_to_eval = _eval_foldname_hack(cf)
    _eval_routine(cf, cf_add_d2, out, cls_names,
        TEST_DATASET_NAME, datalist, model_to_eval)
