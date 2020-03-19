import argparse
import itertools
import cv2
import os
import yacs
import copy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import namedtuple
from typing import (Dict, Tuple, List, cast, NewType,
        Iterable, NamedTuple, Any, TypeVar, Callable)
from mypy_extensions import TypedDict

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

from thes.data.external_dataset import (
        DatasetDALY, DALY_action_name, DALY_object_name, DALY_vid)
from thes.tools import snippets
from thes.det2 import (
        YAML_Base_RCNN_C4, YAML_faster_rcnn_R_50_C4,
        launch_w_logging, launch_without_logging, simple_d2_setup,
        base_d2_frcnn_config, set_d2_cthresh, d2dict_gpu_scaling,
        D2DICT_GPU_SCALING_DEFAULTS,
        get_frame_without_crashing)
from thes.daly_d2 import (
        simplest_daly_to_datalist, get_daly_split_vids,
        get_daly_gt_tubes, gt_tubes_to_df)
from thes.eval_tools import (legacy_evaluation, datalist_to_voclike,
        oldcode_evaluate_voc_detections)
from thes.experiments.demo_december.load_daly import (
        DALY_wein_tube, DALY_tube_index)
from thes.experiments.demo_jan.d2_dalytrain import (
        make_datalist_objaction_similar_merged,
        get_similar_action_objects_DALY)
from thes.eval_tools import voc_ap


log = logging.getLogger(__name__)


def filter_tube_keyframes_only_gt(dataset, tubes_per_video):
    gt_tubes = get_daly_gt_tubes(dataset)
    gt_df = gt_tubes_to_df(dataset, gt_tubes)
    # Query good inds per vid
    good_inds_per_vid = {}
    for vid, gindices in gt_df.groupby('vid').groups.items():
        qdf = gt_df.loc[gindices]
        sorted_inds = sorted(
                itertools.chain.from_iterable(qdf.frame_inds.tolist()))
        good_inds_per_vid[vid] = sorted_inds
    # Filter tubes to only gt keyframes
    filtered_tubes = {}
    for k, v in tqdm(tubes_per_video.items(), 'filter_tubes'):
        (vid, bunch_id, tube_id) = k
        good_inds = good_inds_per_vid[vid]
        intersecting_inds, comm1, comm2 = \
            np.intersect1d(v['frame_inds'], good_inds, return_indices=True)
        if len(intersecting_inds):
            v_intersect = {}
            for k0, v0 in v.items():
                v_intersect[k0] = v0[comm1]
            filtered_tubes[k] = v_intersect
    return filtered_tubes


def get_subset_tubes_(split_vids, tubes_per_video):
    subset_tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube] = {}
    for k, v in tubes_per_video.items():
        (vid, bunch_id, tube_id) = k
        if vid in split_vids:
            subset_tubes_per_video[k] = v
    return subset_tubes_per_video


def get_subset_tubes(dataset, split_label, tubes_per_video):
    split_vids = get_daly_split_vids(dataset, split_label)
    return get_subset_tubes_(split_vids, tubes_per_video)


def sample_some_tubes(tubes_per_video, N=10, NP_SEED=0):
    np_rstate = np.random.RandomState(NP_SEED)
    prm_key_indices = np_rstate.permutation(
            np.arange(len(tubes_per_video)))
    key_list = list(tubes_per_video.keys())
    some_keys = [key_list[i] for i in prm_key_indices[:N]]
    some_tubes = {k: tubes_per_video[k] for k in some_keys}
    return some_tubes


def _set_tubes(cf, dataset):
    tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['tubes.imported_wein_tubes'])
    if cf['tubes.filter_gt']:
        tubes_per_video = filter_tube_keyframes_only_gt(
                dataset, tubes_per_video)
    split_label = cf['dataset.subset']
    tubes_per_video = \
            get_subset_tubes(dataset, split_label, tubes_per_video)
    return tubes_per_video


def eval_daly_tubes_RGB_demovis(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def eval_daly_tubes_RGB(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def hacky_gather_evaluated_tubes(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def actual_eval_of_nicphil_etubes(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def actual_eval_of_action_object_predictions(workfolder, cfg_dict, add_args):
    raise NotImplementedError()


def assign_objactions_to_tubes(workfolder, cfg_dict, add_args):
    raise NotImplementedError()
