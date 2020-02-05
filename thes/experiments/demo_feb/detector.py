import argparse
import os
import yacs
import copy
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, cast

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

import torch

from detectron2.engine import (
        DefaultTrainer, hooks)
from detectron2.data import (
        DatasetCatalog, MetadataCatalog,
        build_detection_train_loader,
        build_detection_test_loader)
from detectron2.data import detection_utils as d2_dutils
from detectron2.data import transforms as d2_transforms
from detectron2.engine.defaults import DefaultPredictor
from detectron2.layers.nms import nms, batched_nms
from detectron2.structures import BoxMode

from thes.data.external_dataset import (
        DatasetDALY, DALY_action_name, DALY_object_name)
from thes.tools import snippets
from thes.det2 import (
        YAML_Base_RCNN_C4, YAML_faster_rcnn_R_50_C4,
        launch_w_logging, launch_without_logging, simple_d2_setup,
        base_d2_frcnn_config, set_d2_cthresh, d2dict_gpu_scaling,
        D2DICT_GPU_SCALING_DEFAULTS,
        get_frame_without_crashing,)
from thes.daly_d2 import (
        simplest_daly_to_datalist,
        get_daly_split_vids,
        DalyVideoDatasetMapper,)
from thes.eval_tools import legacy_evaluation


log = logging.getLogger(__name__)


def daly_to_datalist_pfadet(dataset, split_vids):
    d2_datalist = []
    for vid in split_vids:
        v = dataset.video_odict[vid]
        vmp4 = dataset.source_videos[vid]
        video_path = vmp4['video_path']
        height = vmp4['height']
        width = vmp4['width']
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                for keyframe in instance['keyframes']:
                    frame_number = keyframe['frameNumber']
                    frame_time = keyframe['time']
                    image_id = '{}_A{}_FN{}_FT{:.3f}'.format(
                            vid, action_name, frame_number, frame_time)
                    box_unscaled = keyframe['boundingBox'].squeeze()
                    bbox = box_unscaled * np.tile([width, height], 2)
                    bbox_mode = BoxMode.XYXY_ABS
                    action_id = dataset.action_names.index(action_name)
                    act_obj = {
                            'bbox': bbox,
                            'bbox_mode': bbox_mode,
                            'category_id': action_id}
                    annotations = [act_obj]
                    record = {
                            'video_path': video_path,
                            'video_frame_number': frame_number,
                            'video_frame_time': frame_time,
                            'action_name': action_name,
                            'image_id': image_id,
                            'height': height,
                            'width': width,
                            'annotations': annotations}
                    d2_datalist.append(record)
    return d2_datalist


class DalyFrameDetectionsTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DalyVideoDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg,
                mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        raise NotImplementedError()


def _set_cfg_defaults_pfadet(cfg):
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


PRETRAINED_WEIGHTS_MODELPATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/horus/pytorch_model_zoo/pascal_voc_baseline/model_final_b1acc2.pkl'


def _set_d2config_pfadet(cf, cf_add_d2, TRAIN_DATASET_NAME):
    # // d2_config
    d_cfg = base_d2_frcnn_config()
    d_cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS_MODELPATH
    d_cfg.DATASETS.TRAIN = (
            TRAIN_DATASET_NAME,)
    d_cfg.DATASETS.TEST = ()
    if cf['gpu_scaling.enabled']:
        d2dict_gpu_scaling(cf, cf_add_d2, cf['num_gpus'])
    # Merge additional keys
    yacs_add_d2 = yacs.config.CfgNode(
            snippets.unflatten_nested_dict(cf_add_d2), [])
    d_cfg.merge_from_other_cfg(yacs_add_d2)
    return d_cfg


def _train_func_pfadet(d_cfg, cf, nargs):
    simple_d2_setup(d_cfg)

    name = nargs.TRAIN_DATASET_NAME
    DatasetCatalog.register(name, lambda: nargs.datalist)
    MetadataCatalog.get(name).set(
            thing_classes=nargs.cls_names)

    trainer = DalyFrameDetectionsTrainer(d_cfg)
    trainer.resume_or_load(resume=nargs.resume)
    trainer.train()


def _figure_out_disturl(add_args):
    if '--port_inc' in add_args:
        ind = add_args.index('--port_inc')
        port_inc = int(add_args[ind+1])
    else:
        port_inc = 0
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14 + port_inc
    dist_url = "tcp://127.0.0.1:{}".format(port)
    return dist_url


def train_d2_framewise_action_detector(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_cfg_defaults_pfadet(cfg)
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
