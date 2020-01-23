import argparse
import cv2
import os
import yacs
import copy
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, cast, NewType

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
        DatasetDALY, DALY_action_name, DALY_object_name)
from thes.tools import snippets
from thes.det2 import (
        YAML_Base_RCNN_C4, YAML_faster_rcnn_R_50_C4,
        launch_w_logging, launch_without_logging, simple_d2_setup,
        base_d2_frcnn_config, set_d2_cthresh, d2dict_gpu_scaling,
        D2DICT_GPU_SCALING_DEFAULTS,
        get_frame_without_crashing)
from thes.daly_d2 import (
        simplest_daly_to_datalist, get_daly_split_vids)
from thes.eval_tools import legacy_evaluation
from thes.experiments.demo_december.load_daly import (
        DALY_wein_tube, DALY_tube_index)


log = logging.getLogger(__name__)
# Some scary caffe manipulations


FrameNumber0 = NewType('FrameNumber0', int)  # 0-based frame number
Scores_per_frame = Dict[FrameNumber0, np.ndarray]


# NICPHIL_RCNN_CAFFE_PATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/30_nicolas_rcnn_hackery/caffe/py-faster-rcnn/caffe-fast-rcnn/python'
NICPHIL_RCNN_CAFFE_PATH = Path('/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/40_recompiled_caffe/caffe/py-faster-rcnn/caffe-fast-rcnn/python')
NICPHIL_RCNN_MODEL_PATH = Path('/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/30_nicolas_rcnn_hackery/models')


def revive_nicolas_caffe():
    caffe_root = NICPHIL_RCNN_CAFFE_PATH
    small.add_pypath(caffe_root)
    os.environ['GLOG_minloglevel'] = '3'  # Stop caffe outputs
    import caffe
    caffe.set_device(0)
    caffe.set_mode_gpu()
    return caffe


def nicolas_net():
    caffe = revive_nicolas_caffe()
    nico_root = NICPHIL_RCNN_MODEL_PATH
    model_weights = nico_root/'net_VGG16_iter_70000PHIL.caffemodel'
    model_proto = nico_root/'net_VGG16_test.prototxt'

    net = caffe.Net(str(model_proto), str(model_weights), caffe.TEST)
    return net


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def model_test_get_image_blob(im, PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True) - PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


@profile
def get_scores_per_frame_RGB(
        net, tube, frames_u8,
        PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE):
    """
    Score per frame for NIC RGB
    """
    tube_scores = []
    for i, frame_BGR in enumerate(frames_u8):
        # Image resizing
        blob_, im_scale_factors = model_test_get_image_blob(
                frame_BGR, PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE)
        blob = blob_.transpose(0, 3, 1, 2)  # 1, H, W, 3 --> 1, 3, H, W
        im_scale_factor = im_scale_factors[0]

        net.blobs['data'].reshape(*blob.shape)
        net.blobs['data'].data[...] = blob

        # Put in the sole ROI
        box = tube['boxes'][i]
        box5 = np.r_[0, box]
        box5 = box5 * im_scale_factor
        _, l, t_, r, d = map(int, box5)
        net.blobs['rois'].reshape(1, 5)
        net.blobs['rois'].data[...] = box5
        cls_prob = net.forward()['cls_prob']
        """
        import cv2
        imshow(np.clip(blob_[0] + PIXEL_MEANS, 1, 254)/255)
        im2 = cv2.rectangle(im.copy(), (l, t_), (r, d), (255, 255, 0), thickness=2)
        # printing
        """
        scores = cls_prob.flatten()
        tube_scores.append(scores)
    return tube_scores


@profile
def eval_daly_tubes_RGB(workfolder, cfg_dict, add_args):
    """
    Run Philippes/Nicolas caffe model to extract 'rcnn scores'
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    inputs:
        imported_wein_tubes: [~, ~]
    """)
    cfg.set_defaults("""
    outputs:
        scores_for_tubes_per_video: 'scores_for_tubes_per_video.pkl'
        multiscored_tubes_per_video: 'multiscored_tubes_per_video.pkl'
    rcnn:
        PIXEL_MEANS: [102.9801, 115.9465, 122.7717]
        TEST_SCALES: [600,]
        TEST_MAX_SIZE: 1000

    eval:
        nms:
            enabled: False
            thresh: 0.5
        iou_thresholds: [0.3, 0.5, 0.75]
        spatiotemporal: False
        use_07_metric: False
        use_diff: False
    """)
    cf = cfg.parse()
    PIXEL_MEANS = cf['rcnn.PIXEL_MEANS']
    TEST_SCALES = cf['rcnn.TEST_SCALES']
    TEST_MAX_SIZE = cf['rcnn.TEST_MAX_SIZE']

    # // Loading inputs //
    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    all_actions: List[DALY_action_name] = dataset.action_names
    tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['inputs.imported_wein_tubes'])

    # // Input manipulations //
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    subset_tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube] = {}
    for k, v in tubes_per_video.items():
        (vid, bunch_id, tube_id) = k
        if vid in split_vids:
            subset_tubes_per_video[k] = v

    # // Computation of the parcels inside chosen chunk
    net = nicolas_net()
    for k, tube in tqdm(subset_tubes_per_video.items(), 'nicphil on tubes'):
        (vid, bunch_id, tube_id) = k
        vmp4 = dataset.source_videos[vid]
        video_path = vmp4['video_path']
        frame_inds = tube['frame_inds']

        # scorefold/f'scores_{video_name}_{tube_id:04d}.pkl')(
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)
        scores_per_frame = get_scores_per_frame_RGB(
                net, tube, frames_u8,
                PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE)
