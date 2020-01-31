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


FrameNumber0 = NewType('FrameNumber0', int)  # 0-based frame number
Scores_per_frame = Dict[FrameNumber0, np.ndarray]


# This tube has boxes only for some frame indices
DALY_sparse_frametube = TypedDict('DALY_sparse_frametube', {
    'frame_inds': np.ndarray,
    'boxes': np.ndarray,  # LTRD
    'start_frame': int,
    'end_frame': int
    })

DALY_sparse_frametube_scored = TypedDict('DALY_sparse_frametube_scored', {
    'frame_inds': np.ndarray,
    'boxes': np.ndarray,  # LTRD
    'start_frame': int,
    'end_frame': int,
    'score': float
    })

Coverage_tube = TypedDict('Coverage_tube', {
    'vid': DALY_vid,
    'gt_tube_id': int,
    'spatial_iou': float,
    'spatial_temp_iou': float,
    'N_viable': int  # wrt SpatioTemporal
})


Options_tube_ap = TypedDict('Options_tube_ap', {
    'iou_thresh': float,
    'spatiotemporal': bool,
    'use_07_metric': bool,
    'use_diff': bool
})


Objaction_dets = TypedDict('Objaction_dets', {
    'pred_boxes': np.ndarray,
    'scores': np.ndarray,
    'pred_classes': np.ndarray  # strings
})


Flat_tube_temp_annotation = NamedTuple('Flat_tube_temp_annotation', [
    ('vid', str),
    ('id_tube', int),
    ('diff', bool),
    ('tube_temp', DALY_sparse_frametube),
])
Flat_tube_detection = NamedTuple('Flat_tube_detection', [
    ('vid', str),
    ('id_tube', int),
    ('score', float),
    ('tube', DALY_sparse_frametube)
])


Stats_daly_ap = TypedDict('Stats_daly_ap', {
    'flat_annotations': List[Flat_tube_temp_annotation],
    'flat_detections': List[Flat_tube_detection],
    'detection_matched': np.ndarray,
    'gt_already_matched': np.ndarray,
    'possible_matches': List[Dict[int, float]],
    'iou_coverages_per_detection_ind': Dict[int, List[float]],
    'detection_matched_to_which_gt': np.ndarray,
    'sorted_inds': np.ndarray,
    'fp': np.ndarray,
    'tp': np.ndarray,
    'npos': int,
    'rec': np.ndarray,
    'prec': np.ndarray,
    'ap': float
})


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


def cv_put_box_with_text(
        image: np.ndarray,
        box_ltrd: Iterable[float],
        # Rectangle params
        rec_color=(255, 255, 255),  # White
        rec_thickness=4,
        # Text params
        text=None,
        text_size=0.6,
        text_color=None,
        text_thickness=2,
        text_position='left_down'
            ) -> np.ndarray:
    """
    Overwrites in place
    """

    l, t, r, d = map(int, box_ltrd)
    result = cv2.rectangle(
            image,
            (l, t), (r, d),
            color=rec_color,
            thickness=rec_thickness)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    if text:
        if text_color is None:
            text_color = rec_color
        # Text Positioning

        retval, baseline = cv2.getTextSize(
                text, fontFace, text_size, text_thickness)
        if text_position == 'left_down':
            text_pos = (l, d-5)
        elif text_position == 'left_up':
            text_pos = (l, t-5)
        elif text_position == 'right_down':
            text_pos = (r-retval[0], d-5)
        elif text_position == 'right_up':
            text_pos = (r-retval[0], t-5)
        else:
            raise ValueError('Wrong text position')
        cv2.putText(
                image,
                text,
                text_pos,
                fontFace=fontFace,
                fontScale=text_size,
                color=text_color,
                thickness=text_thickness)
    return result


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


def perform_tube_demovis(dataset, some_tubes, out,
        PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE):
    net = nicolas_net()
    nicolas_labels = ['background', ] + dataset.action_names
    for k, tube in tqdm(some_tubes.items(), 'nicphil on tubes'):
        (vid, bunch_id, tube_id) = k
        vmp4 = dataset.source_videos[vid]
        # video = dataset.video_odict[vid]
        video_path = vmp4['video_path']
        frame_inds = tube['frame_inds']

        # scorefold/f'scores_{video_name}_{tube_id:04d}.pkl')(
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)
        scores_per_frame = get_scores_per_frame_RGB(
                net, tube, frames_u8,
                PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE)
        txt_output = []
        video_fold = small.mkdir(out/'{}_{}_{}'.format(
            vid, bunch_id, tube_id))
        for i, (frame, score) in enumerate(zip(frames_u8, scores_per_frame)):
            image = frame.copy()
            box = tube['boxes'][i]
            real_framenum = tube['frame_inds'][i]
            best_score_id = np.argmax(score)
            best_score = score[best_score_id]
            best_nicolas_label = nicolas_labels[best_score_id]
            cv_put_box_with_text(
                    image, box,
                    text='{} {} {} {:.2f}'.format(
                        i, real_framenum,
                        best_nicolas_label, best_score))
            line = ' '.join([f'{y}: {x:.3f}'
                for x, y in zip(score, nicolas_labels)])
            txt_output.append(line)
            cv2.imwrite(
                    str(video_fold/'frame{:03d}_{:03d}.jpg'.format(
                        i, real_framenum)),
                    image)
        with (video_fold/'scores.txt').open('w') as f:
            f.write('\n'.join(txt_output))


def _set_tubecfg(cfg):
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    """)
    cfg.set_defaults("""
    rcnn:
        PIXEL_MEANS: [102.9801, 115.9465, 122.7717]
        TEST_SCALES: [600,]
        TEST_MAX_SIZE: 1000
    """)
    return cfg


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
    """
    Run Philippes/Nicolas caffe model to extract 'rcnn scores'
    DEMO version that displays pretty stuff
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_tubecfg(cfg)
    cf = cfg.parse()
    PIXEL_MEANS = cf['rcnn.PIXEL_MEANS']
    TEST_SCALES = cf['rcnn.TEST_SCALES']
    TEST_MAX_SIZE = cf['rcnn.TEST_MAX_SIZE']

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    tubes_per_video = _set_tubes(cf, dataset)

    some_tubes = sample_some_tubes(
            tubes_per_video, N=10, NP_SEED=0)

    perform_tube_demovis(dataset, some_tubes, out,
            PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE)


def eval_daly_tubes_RGB(workfolder, cfg_dict, add_args):
    """
    Run Philippes/Nicolas caffe model to extract 'rcnn scores'
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_tubecfg(cfg)
    cfg.set_deftype("""
    compute:
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
    save_period: ['::10', str]
    """)
    cf = cfg.parse()
    PIXEL_MEANS = cf['rcnn.PIXEL_MEANS']
    TEST_SCALES = cf['rcnn.TEST_SCALES']
    TEST_MAX_SIZE = cf['rcnn.TEST_MAX_SIZE']

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    tubes_per_video = _set_tubes(cf, dataset)

    # // Computation of the parcels inside chosen chunk
    chunk = (cf['compute.chunk'], cf['compute.total'])
    cc, ct = chunk

    key_indices = np.arange(len(tubes_per_video))
    key_list = list(tubes_per_video.keys())
    chunk_indices = np.array_split(key_indices, ct)[cc]
    chunk_keys = [key_list[i] for i in chunk_indices]
    # chunk_tubes = {k: tubes_per_video[k] for k in chunk_keys}
    log.info('Chunk {}: {} -> {}'.format(
        chunk, len(key_indices), len(chunk_indices)))

    net = nicolas_net()

    def tube_eval_func(k):
        tube = tubes_per_video[k]
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
        return scores_per_frame

    df_isaver = snippets.Simple_isaver(
            small.mkdir(out/'tube_eval_isaver'),
            chunk_keys, tube_eval_func, cf['save_period'], 120)
    predicted_tubescores = df_isaver.run()
    tubescores_dict = dict(zip(chunk_keys, predicted_tubescores))
    small.save_pkl(out/'tubescores_dict.pkl', tubescores_dict)


def hacky_gather_evaluated_tubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    cfg.set_deftype("""
    etubes: [~, ~]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    """)
    cf = cfg.parse()

    # Read tubes, merge dicts
    tubescores_dict = {}
    for tubepath in cf['etubes']:
        tubes = small.load_pkl(tubepath)
        tubescores_dict.update(tubes)

    # Confirm that keys match
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    tubes_per_video = _set_tubes(cf, dataset)

    assert tubes_per_video.keys() == tubescores_dict.keys(), \
            "Keys should match"

    small.save_pkl(out/'tubescores_dict.pkl', tubescores_dict)


from detectron2.structures import BoxMode


def convert_daly_to_action_based_datalist(dataset, split_label):
    split_vids = get_daly_split_vids(dataset, split_label)

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


def numpy_box_area(box):
    assert box.shape == (4,)
    return np.prod(box[2:] - box[:2])


def numpy_iou(box1, box2):
    assert box1.shape == (4,)
    assert box2.shape == (4,)
    # Computing IOU
    inter = np.r_[
        np.maximum(box1[:2], box2[:2]),
        np.minimum(box1[2:], box2[2:])]
    if np.any(inter[:2] > inter[2:]):
        iou = 0.0
    else:
        inter_area = numpy_box_area(inter)
        union_area = (
            np.prod(box1[2:] - box1[:2]) +
            np.prod(box2[2:] - box2[:2]) - inter_area)
        iou = inter_area/union_area
    return iou


def numpy_inner_overlap_11(box1, box2):
    assert box1.shape == (4,)
    assert box2.shape == (4,)
    inter = np.r_[
        np.maximum(box1[:2], box2[:2]),
        np.minimum(box1[2:], box2[2:])]
    if np.any(inter[:2] > inter[2:]):
        inter = 0.0
    else:
        inter_area = numpy_box_area(inter)
        inter = inter_area/numpy_box_area(box1)
    return inter


def numpy_inner_overlap_N1(boxes1, box2):
    assert len(boxes1.shape) == 2
    assert boxes1.shape[-1] == 4
    assert box2.shape == (4,)

    inter = np.c_[
        np.maximum(boxes1[..., :2], box2[:2]),
        np.minimum(boxes1[..., 2:], box2[2:])]

    # Intersection area
    inter_subs = inter[..., 2:] - inter[..., :2]
    inter_area = np.prod(inter_subs, axis=1)
    inter_area[(inter_subs < 0).any(axis=1)] = 0.0

    # Divide
    boxes1_areas = np.prod(boxes1[..., 2:] - boxes1[..., :2], axis=1)
    ioverlaps = inter_area / boxes1_areas
    return ioverlaps


def daly_coverage_tube_phil(
        gt_dict: Dict[DALY_vid, List[DALY_sparse_frametube]],
        proposal_dict: Dict[DALY_vid, List[DALY_sparse_frametube_scored]]
            ) -> List[Coverage_tube]:
    """ Best coverage per each GT tube """

    # Walk over videos, try to cover GT tubes
    coverages_tube: List[Coverage_tube] = []
    for vid, gt_tubes in gt_dict.items():
        proposal_tubes: List[DALY_sparse_frametube_scored] = \
                proposal_dict.get(vid, [])
        if len(proposal_tubes) == 0:
            continue
        # Extract min/max frames
        proposal_frame_ranges = np.array([
            (x['start_frame'], x['end_frame'])
            for x in proposal_tubes])  # (N, 2)
        for gt_tube_id, gt_tube in enumerate(gt_tubes):
            gt_beginFrame = gt_tube['start_frame']
            gt_endFrame = gt_tube['end_frame']
            begin = np.maximum(proposal_frame_ranges[:, 0], gt_beginFrame)
            end = np.minimum(proposal_frame_ranges[:, 1], gt_endFrame)
            temp_inters = end-begin+1
            # Prepare iou values
            possible_spatial_ious = np.zeros(len(proposal_tubes))
            possible_temp_ious = np.zeros(len(proposal_tubes))
            # Loop over proposal tubes that have at least some temporal
            # intersection
            for proposal_tube_id in np.where(temp_inters > 0)[0]:
                proposal_tube: DALY_sparse_frametube_scored = \
                        proposal_tubes[proposal_tube_id]
                # Spatial
                spatial_ious: List[float] = []
                for gt_keyframe_id, gt_box in zip(
                        gt_tube['frame_inds'],
                        gt_tube['boxes']):
                    # Spatial IOU
                    if gt_keyframe_id in proposal_tube['frame_inds']:
                        found_ind = np.searchsorted(
                                proposal_tube['frame_inds'],
                                gt_keyframe_id)
                        proposal_box = proposal_tube['boxes'][found_ind]
                        spatial_iou = numpy_iou(gt_box, proposal_box)
                    else:
                        log.debug('Proposal tube ({}/{}) lacks '
                            'GT keyframe {}'.format(
                                vid, proposal_tube_id, gt_keyframe_id))
                        spatial_iou = 0
                    spatial_ious.append(spatial_iou)
                avg_spatial_iou = np.mean(spatial_ious)
                # Temporal
                temp_inter = temp_inters[proposal_tube_id]
                proposal_begin, proposal_end = \
                        proposal_frame_ranges[proposal_tube_id]
                temp_union = (gt_endFrame - gt_beginFrame + 1) + \
                        (proposal_end - proposal_begin + 1) - temp_inter
                temp_iou = temp_inter/temp_union
                # Report upstairs
                possible_spatial_ious[proposal_tube_id] = avg_spatial_iou
                possible_temp_ious[proposal_tube_id] = temp_iou
            # Decide on best IOU
            possible_spatial_temp_ious = \
                    possible_spatial_ious * possible_temp_ious
            best_spatial_iou = possible_spatial_ious.max()
            best_spatial_temp_iou = possible_spatial_temp_ious.max()
            N_viable = len(np.where(possible_spatial_temp_ious > 0)[0])
            # Report upstairs
            coverages_tube.append(Coverage_tube(
                vid=vid,
                gt_tube_id=gt_tube_id,
                spatial_iou=best_spatial_iou,
                spatial_temp_iou=best_spatial_temp_iou,
                N_viable=N_viable))
    return coverages_tube


def compute_daly_coverage(
        all_actions: List[DALY_action_name],
        scored_tubes_per_video_per_action: Dict[DALY_action_name,
            Dict[DALY_vid, List[DALY_sparse_frametube_scored]]],
        daly_tubes_temp_per_action: Dict[DALY_action_name,
            Dict[DALY_vid, List[DALY_sparse_frametube]]],
            ):

    """ Tube Coverage stats (for tubes above 0 score) """

    coverage_per_action = {}
    for action_cls in all_actions:
        daly_tubes_temp = daly_tubes_temp_per_action[action_cls]
        # Tubes > 0 score
        scored_tubes_per_video = scored_tubes_per_video_per_action[action_cls]
        tubes_above0_per_video: Dict[DALY_vid,
                List[DALY_sparse_frametube_scored]] = {}
        for vid, tubes in scored_tubes_per_video.items():
            tubes_above0 = [t for t in tubes if t['score'] > 0.]
            tubes_above0_per_video[vid] = tubes_above0
        coverage_per_action[action_cls] = pd.DataFrame(
            daly_coverage_tube_phil(daly_tubes_temp, tubes_above0_per_video))
    return coverage_per_action


def _daly_recall_as_series(
        coverage_per_action,
        iou_thresh: float
            ) -> pd.Series:

    # Recall (T `thresh)
    recall_s_thresh = {}
    for k, v in coverage_per_action.items():
        if len(v) == 0:
            recall = 0.
        else:
            recall = (v.spatial_iou > iou_thresh).mean()*100
        recall_s_thresh[k] = recall
    s_recall = pd.Series(recall_s_thresh,
            name=f'S recall@{iou_thresh:.2f}').round(2)
    s_recall.loc['AVERAGE'] = s_recall.mean()

    # Recall (ST `thresh`)
    recall_st_thresh = {}
    for k, v in coverage_per_action.items():
        if len(v) == 0:
            recall = 0.
        else:
            recall = (v.spatial_temp_iou > iou_thresh).mean()*100
        recall_st_thresh[k] = recall
    st_recall = pd.Series(recall_st_thresh,
            name=f'ST recall@{iou_thresh:.2f}').round(2)
    st_recall.loc['AVERAGE'] = st_recall.mean()

    return s_recall, st_recall


def _daly_ap_as_series(
    all_actions: List[DALY_action_name],
    scored_tubes_per_video_per_action: Dict[DALY_action_name,
        Dict[DALY_vid, List[DALY_sparse_frametube_scored]]],
    daly_tubes_temp_per_action: Dict[DALY_action_name,
        Dict[DALY_vid, List[DALY_sparse_frametube]]],
    options_tube_ap: Options_tube_ap
        ) -> Tuple[pd.Series, Dict[DALY_action_name, float]]:

    """ Evaluation of AP per action """

    ap_per_cls = {}
    # //// Evaluation per class ////
    with small.QTimer('Evaluation AP took %(time)s sec'):
        for action_cls in all_actions:
            # GT tubes
            daly_tubes_temp: \
                    Dict[DALY_vid, List[DALY_sparse_frametube]] = \
                    daly_tubes_temp_per_action[action_cls]
            # Proposals
            scored_tubes_per_video = \
                    scored_tubes_per_video_per_action[action_cls]
            # AP computations
            stats = daly_average_precision_stats(
                    daly_tubes_temp, scored_tubes_per_video, **options_tube_ap)
            ap = stats['ap']
            ap_per_cls[action_cls] = ap

    # Result printed nicely via pd.Series
    thr = options_tube_ap['iou_thresh']
    x = pd.Series(ap_per_cls, name=f'AP @{thr:.2f}')*100
    x.loc['AVERAGE'] = x.mean()
    return x, ap_per_cls


def temporal_IOU(
        b1, e1, b2, e2):

    begin = max(b1, b2)
    end = min(e1, e2)
    inter = end-begin+1
    if inter <= 0:
        return 0.0
    else:
        union = (e1 - b1 + 1) + (e2 - b2 + 1) - inter
        return inter/union


def spatial_tube_iou(
        tube1: DALY_sparse_frametube,
        tube2: DALY_sparse_frametube,
            ) -> float:

    tubedict1 = dict(zip(tube1['frame_inds'], tube1['boxes']))
    tubedict2 = dict(zip(tube2['frame_inds'], tube2['boxes']))

    spatial_ious: List[float] = []
    for kf_id1, box1 in tubedict1.items():
        if kf_id1 in tubedict2:
            box2 = tubedict2[kf_id1]
            iou = numpy_iou(box1, box2)
            spatial_ious.append(iou)
    if len(spatial_ious):
        avg_iou = np.mean(spatial_ious)
    else:
        avg_iou = 0.
    return avg_iou


def daly_average_precision_stats(
        gt_dict: Dict[DALY_vid, List[DALY_sparse_frametube]],
        proposal_dict: Dict[DALY_vid, List[DALY_sparse_frametube_scored]],
        iou_thresh: float,
        spatiotemporal: bool,
        use_07_metric: bool,
        use_diff: bool
            ) -> Stats_daly_ap:
    """
    We always compute stats
    """
    # Flatten tube representations
    flat_annotations: List[Flat_tube_temp_annotation] = []
    flat_detections: List[Flat_tube_detection] = []
    # Extract GT annotation_list
    for vid, gt_tubes in sorted(gt_dict.items()):
        for id_tube, gt_tube_temp in enumerate(gt_tubes):
            flat_annotations.append(Flat_tube_temp_annotation(
                vid, id_tube, False, gt_tube_temp))
    # Extract Proposal list
    for vid, proposal_tubes in sorted(proposal_dict.items()):
        for id_tube, prop_tube in enumerate(proposal_tubes):
            prop_score = prop_tube['score']
            flat_detections.append(Flat_tube_detection(
                vid, id_tube, prop_score, prop_tube))

    # Precompute 'temporal iou' and indices of tubesp
    possible_matches_per_detection: List[Dict[int, float]] = []
    for _detection in flat_detections:
        ind_to_iou: Dict[int, float] = {}
        begin_det = _detection.tube['start_frame']
        end_det = _detection.tube['end_frame']
        for annotation_id, annotation in enumerate(flat_annotations):
            if annotation.vid == _detection.vid:
                tube_ann = annotation.tube_temp
                begin_ann = tube_ann['start_frame']
                end_ann = tube_ann['end_frame']
                temp_iou = temporal_IOU(
                        begin_ann, end_ann, begin_det, end_det)
                if temp_iou > 0.0:
                    ind_to_iou[annotation_id] = temp_iou
        possible_matches_per_detection.append(ind_to_iou)

    # Preparation
    detection_matched = np.zeros(len(flat_detections), dtype=bool)
    gt_already_matched = np.zeros(len(flat_annotations), dtype=bool)
    # Provenance
    detection_matched_to_which_gt = np.ones(len(flat_detections), dtype=int)*-1
    iou_coverages_per_detection_ind = {}  # type: Dict[int, List[float]]

    # VOC2007 preparation
    nd = len(flat_detections)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    if use_diff:
        npos = len(flat_annotations)
    else:
        npos = len([x for x in flat_annotations if not x.diff])

    # Go through ordered detections
    detection_scores = np.array([x.score for x in flat_detections])
    detection_scores = detection_scores.round(3)
    sorted_inds = np.argsort(-detection_scores)
    for d, detection_ind in enumerate(sorted_inds):
        # Check available GTs
        gt_ids_that_overlap = possible_matches_per_detection[detection_ind]
        if len(gt_ids_that_overlap) == 0:
            fp[d] = 1
            continue

        detection: Flat_tube_detection = flat_detections[detection_ind]
        detection_tube: DALY_sparse_frametube = detection.tube

        # Compute IOUs
        iou_coverages: List[float] = []
        for gt_id, temp_iou in gt_ids_that_overlap.items():
            gt_tube_anno: Flat_tube_temp_annotation = flat_annotations[gt_id]
            gt_tube = gt_tube_anno.tube_temp
            # gt_beginFrame = gt_tube['start_frame']
            # gt_endFrame = gt_tube['end_frame']
            spatial_iou = spatial_tube_iou(gt_tube, detection_tube)
            # SpatioTemporal or Spatial iou
            if spatiotemporal:
                iou = temp_iou * spatial_iou
            else:
                iou = spatial_iou
            iou_coverages.append(iou)
        # Provenance
        iou_coverages_per_detection_ind[detection_ind] = iou_coverages

        max_coverage_id = np.argmax(iou_coverages)
        max_coverage = iou_coverages[max_coverage_id]
        max_coverage_gt_id = list(gt_ids_that_overlap.keys())[max_coverage_id]

        # Mirror VOC eval
        if max_coverage > iou_thresh:
            if (not use_diff) and flat_annotations[max_coverage_gt_id].diff:
                continue
            if not gt_already_matched[max_coverage_gt_id]:
                tp[d] = 1
                detection_matched[detection_ind] = True
                gt_already_matched[max_coverage_gt_id] = True
                detection_matched_to_which_gt[detection_ind] = max_coverage_gt_id
            else:
                fp[d] = 1
        else:
            fp[d] = 1
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # All kinds of stats gathered together
    stats = Stats_daly_ap(flat_annotations=flat_annotations,
            flat_detections=flat_detections,
            detection_matched=detection_matched,
            gt_already_matched=gt_already_matched,
            possible_matches=possible_matches_per_detection,
            iou_coverages_per_detection_ind=iou_coverages_per_detection_ind,
            detection_matched_to_which_gt=detection_matched_to_which_gt,
            sorted_inds=sorted_inds, fp=fp, tp=tp, npos=npos, rec=rec,
            prec=prec, ap=ap)

    return stats


def df_to_table(df: pd.DataFrame, indexname=None) -> str:
    # Header
    if indexname is None:
        indexname = df.index.name
    if indexname is None:
        indexname = 'index'
    header = [indexname, ] + list(df.columns)
    # Col formats
    col_formats = ['{}']
    for dt in df.dtypes:
        form = '{}'
        if dt in ['float32', 'float64']:
            form = '{:.2f}'
        col_formats.append(form)

    table = snippets.string_table(
            np.array(df.reset_index()),
            header=header,
            col_formats=col_formats,
            pad=2)
    return table


def _get_gt_sparsetubes(dataset, split_vids, gt_tubes):
    daly_tubes_temp_per_action: \
            Dict[DALY_action_name,
                Dict[DALY_vid,
                    List[DALY_sparse_frametube]]] = {}

    for ckey, gt_tube in gt_tubes.items():
        (vid, action_name, ins_ind) = ckey
        if vid not in split_vids:
            continue
        vmp4 = dataset.source_videos[vid]
        height = vmp4['height']
        width = vmp4['width']
        ocv_video_fps = vmp4['frames_reached']/vmp4['length_reached']
        frame_inds = np.array(gt_tube['frame_inds'])
        unscaled_boxes = np.array(gt_tube['boxes'])
        boxes = unscaled_boxes * np.tile([width, height], 2)
        start_frame = int(gt_tube['start_time'] * ocv_video_fps)
        end_frame = int(gt_tube['end_time'] * ocv_video_fps)
        sparse_tube = {
                'frame_inds': frame_inds,
                'boxes': boxes,
                'start_frame': start_frame,
                'end_frame': end_frame}
        (daly_tubes_temp_per_action
                .setdefault(action_name, {})
                .setdefault(vid, [])).append(sparse_tube)
    return daly_tubes_temp_per_action


def overlap_DALY_sparse_frametube(
        x: DALY_sparse_frametube_scored,
        y: DALY_sparse_frametube_scored) -> float:
    """
    - We utilize simple spatio-temporal overlap here:
        - S = Mean spatial IOU for overlapping frames
        - T = Temporal IOU of (start, end)
    - We return S*T
    """
    # Find overlapping keyframes
    union_keys = np.intersect1d(list(x), list(y))
    if len(union_keys) == 0:
        return 0.0
    # Spatial
    spat = spatial_tube_iou(x, y)
    # Temporal
    temp = temporal_IOU(
            x['start_frame'], x['end_frame'],
            y['start_frame'], y['end_frame'])
    return spat * temp


T = TypeVar('T')


def custom_nms(
        element_list: List[T],
        overlap_func: Callable[[T, T], float],
        score_func: Callable[[T], float],
        thresh
            ) -> List[T]:
    """
    Provided with function to compare overlap between two elements performs NMS
    on tubes
    """

    scores = [score_func(e) for e in element_list]
    sorted_ids = np.argsort(scores)[::-1]  # In decreasing order
    sorted_candidates = [element_list[i] for i in sorted_ids]
    results = []
    while len(sorted_candidates):
        taken = sorted_candidates.pop(0)
        results.append(taken)
        overlaps = [overlap_func(taken, c) for c in sorted_candidates]
        sorted_candidates = [c for c, o in zip(sorted_candidates, overlaps) if o < thresh]

    return results


def nms_on_scored_tubes(
        scored_tubes_per_video: Dict[str, List[DALY_sparse_frametube_scored]],
        thresh: float
            ) -> Dict[str, List[DALY_sparse_frametube_scored]]:

    def score_func(x: DALY_sparse_frametube_scored):
        return x['score']

    def overlap_func(
            x: DALY_sparse_frametube_scored,
            y: DALY_sparse_frametube_scored) -> float:
        return overlap_DALY_sparse_frametube(x, y)

    nmsed_scored_tubes_per_video: \
            Dict[str, List[DALY_sparse_frametube_scored]] = {}

    for video_name, tubes in tqdm(scored_tubes_per_video.items(), desc='nms'):
        nmsed_tubes: List[DALY_sparse_frametube_scored] = custom_nms(
                tubes, overlap_func, score_func, thresh)
        nmsed_scored_tubes_per_video[video_name] = nmsed_tubes
    return nmsed_scored_tubes_per_video


def scored_tube_nms(
        stubes_va, thresh, nms_folder):
    nmsed_stubes_va = {}
    for action_name, scored_tubes_per_video in stubes_va.items():
        nmsed_stubes_v = small.stash2(
                nms_folder/f'scored_tubes_nms_{thresh:.2f}_at_{action_name}.pkl')(
                nms_on_scored_tubes,
                scored_tubes_per_video, thresh)
        nmsed_stubes_va[action_name] = nmsed_stubes_v
    return nmsed_stubes_va


def _daly_tube_map(cf, out, dataset, stubes_va, gttubes_va):

    # Apply per-class NMS
    if cf['tube_nms.enabled']:
        tube_nms_thresh = cf['tube_nms.thresh']
        stubes_va = scored_tube_nms(stubes_va, tube_nms_thresh, out)

    all_actions = dataset.action_names

    options_tube_ap: Options_tube_ap = {
            'iou_thresh': None,
            'spatiotemporal': cf['eval.spatiotemporal'],
            'use_07_metric': cf['eval.use_07_metric'],
            'use_diff': cf['eval.use_diff'],
    }
    iou_thresholds = cf['eval.iou_thresholds']
    for iou_thresh in iou_thresholds:
        options_tube_ap['iou_thresh'] = iou_thresh
        coverage = compute_daly_coverage(all_actions,
                stubes_va, gttubes_va)
        series_rec_s, series_rec_st = _daly_recall_as_series(coverage, iou_thresh)
        series_ap, ap_per_cls = _daly_ap_as_series(
                all_actions, stubes_va, gttubes_va, options_tube_ap)
        df_recap = pd.concat((series_rec_s, series_rec_st, series_ap), axis=1)
        log.info('For thresh {:.2f}\n{}'.format(iou_thresh, df_to_table(df_recap)))


def _daly_tube_map_v2(cf, out, dataset, stubes_va, gttubes_va):

    # Apply per-class NMS
    if cf['tube_eval.nms.enabled']:
        tube_nms_thresh = cf['tube_eval.nms.thresh']
        stubes_va = scored_tube_nms(stubes_va, tube_nms_thresh, out)

    all_actions = dataset.action_names

    options_tube_ap: Options_tube_ap = {
            'iou_thresh': None,
            'spatiotemporal': cf['tube_eval.params.spatiotemporal'],
            'use_07_metric': cf['tube_eval.params.use_07_metric'],
            'use_diff': cf['tube_eval.params.use_diff'],
    }
    iou_thresholds = cf['tube_eval.params.iou_thresholds']
    for iou_thresh in iou_thresholds:
        options_tube_ap['iou_thresh'] = iou_thresh
        coverage = compute_daly_coverage(all_actions,
                stubes_va, gttubes_va)
        series_rec_s, series_rec_st = _daly_recall_as_series(coverage, iou_thresh)
        series_ap, ap_per_cls = _daly_ap_as_series(
                all_actions, stubes_va, gttubes_va, options_tube_ap)
        df_recap = pd.concat((series_rec_s, series_rec_st, series_ap), axis=1)
        log.info('For thresh {:.2f}\n{}'.format(iou_thresh, df_to_table(df_recap)))


def nicphil_evaluations_to_tubes(
        dataset, tubes_per_video, original_tubes_per_video, tubescores_dict):
    stubes_va: \
            Dict[DALY_action_name,
                    Dict[DALY_vid, List[DALY_sparse_frametube_scored]]] = {}
    for ckey, tube in tubes_per_video.items():
        (vid, bunch_id, tube_id) = ckey
        original_tube = original_tubes_per_video[ckey]
        tubescores = tubescores_dict[ckey]
        agg_tubescores = np.vstack(tubescores).sum(0)[1:]
        frame_inds = tube['frame_inds']
        boxes = tube['boxes']
        start_frame = original_tube['frame_inds'].min()
        end_frame = original_tube['frame_inds'].max()
        # Sum the perframe scores
        for action_name, score in zip(
                dataset.action_names, agg_tubescores):
            sparse_scored_tube = {
                    'frame_inds': frame_inds,
                    'boxes': boxes,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'score': score}
            (stubes_va
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(sparse_scored_tube))
    return stubes_va


def _create_objdetection_helper_structure(
        dataset, split_label, actobjects_evaluated):
    # Assign scores to these tubes, based on intersections
    # with objaction detections
    datalist = simplest_daly_to_datalist(dataset, split_label)
    action_object_to_object = get_similar_action_objects_DALY()
    object_names = sorted([x
        for x in set(list(action_object_to_object.values())) if x])
    # num_classes = len(object_names)
    def datalist_converter(datalist):  # NOQA
        datalist = make_datalist_objaction_similar_merged(
                datalist, dataset.object_names, object_names,
                action_object_to_object)
        return datalist
    datalist = datalist_converter(datalist)

    # // Create a helper structure
    preds_per_framevid: Dict[DALY_vid, Dict[int, Dict]] = {}
    for dl_item, pred_item in zip(datalist, actobjects_evaluated):
        pred_boxes = pred_item.pred_boxes.tensor.numpy()
        scores = pred_item.scores.numpy()
        pred_classes = pred_item.pred_classes.numpy()
        detections = {
                'pred_boxes': pred_boxes,
                'scores': scores,
                'pred_classes': pred_classes}
        (preds_per_framevid
            .setdefault(dl_item['vid'], {})
            [dl_item['video_frame_number']]) = detections
    return preds_per_framevid


def _match_objectdetections_to_tubes(
        dataset, tubes_per_video, original_tubes_per_video,
        preds_per_framevid,
        overlap_type, overlap_cutoff, score_cutoff):
    # To every tube, find matching keyframes
    n_classes = len(dataset.action_names)
    cls_scores_per_tube: Dict[DALY_tube_index, np.ndarray] = {}
    for k, wein_tube in tqdm(tubes_per_video.items(), 'match_keyframes'):
        (vid, bunch_id, tube_id) = k
        cls_scores = np.zeros(n_classes)
        for frame_ind, tube_box in zip(
                wein_tube['frame_inds'],
                wein_tube['boxes']):
            matching_keyframe = preds_per_framevid.get(vid, {}).get(frame_ind)
            if matching_keyframe is None:
                continue
            # Check overlap
            if overlap_type == 'iou':
                overlaps = []
                for pred_box in matching_keyframe['pred_boxes']:
                    overlaps.append(numpy_iou(pred_box, tube_box))
                overlaps = np.array(overlaps)
            elif overlap_type == 'inner_overlap':
                overlaps = numpy_inner_overlap_N1(
                        matching_keyframe['pred_boxes'], tube_box)
            else:
                raise RuntimeError()
            good_overlaps = overlaps > overlap_cutoff
            # Check score
            good_score = matching_keyframe['scores'] > score_cutoff
            # Merge
            good = good_overlaps & good_score

            for score, cls in zip(
                    matching_keyframe['scores'][good],
                    matching_keyframe['pred_classes'][good]):
                cls_scores[cls] += score
        cls_scores_per_tube[k] = cls_scores

    # Create scored_tubes
    stubes_va: \
            Dict[DALY_action_name,
                    Dict[DALY_vid, List[DALY_sparse_frametube_scored]]] = {}
    for ckey, tube in tubes_per_video.items():
        (vid, bunch_id, tube_id) = ckey
        agg_tubescores = cls_scores_per_tube[ckey]

        original_tube = original_tubes_per_video[ckey]
        frame_inds = tube['frame_inds']
        boxes = tube['boxes']
        start_frame = original_tube['frame_inds'].min()
        end_frame = original_tube['frame_inds'].max()
        # Sum the perframe scores
        for action_name, score in zip(
                dataset.action_names, agg_tubescores):
            sparse_scored_tube = {
                    'frame_inds': frame_inds,
                    'boxes': boxes,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'score': score}
            if score < 0.05:
                continue
            (stubes_va
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(sparse_scored_tube))
    return stubes_va


def _objectdetections_into_scores(
        objactions_vf, tubes_per_video,
        overlap_type, overlap_cutoff, score_cutoff
        ) -> Dict[DALY_tube_index, Dict[DALY_action_name, float]]:
    # To every tube, find matching keyframes
    cls_scores_per_tube: Dict[DALY_tube_index,
            Dict[DALY_action_name, float]] = {}
    for k, wein_tube in tqdm(tubes_per_video.items(), 'match_keyframes'):
        (vid, bunch_id, tube_id) = k
        cls_scores: Dict[DALY_action_name, float] = {}
        for frame_ind, tube_box in zip(
                wein_tube['frame_inds'],
                wein_tube['boxes']):
            mobjactions: Objaction_dets = \
                    objactions_vf.get(vid, {}).get(frame_ind)
            if mobjactions is None:
                continue
            # Check overlap
            if overlap_type == 'iou':
                overlaps = []
                for pred_box in mobjactions['pred_boxes']:
                    overlaps.append(numpy_iou(pred_box, tube_box))
                overlaps = np.array(overlaps)
            elif overlap_type == 'inner_overlap':
                overlaps = numpy_inner_overlap_N1(
                        mobjactions['pred_boxes'], tube_box)
            else:
                raise RuntimeError()
            good_overlaps = overlaps > overlap_cutoff
            # Check score
            good_score = mobjactions['scores'] > score_cutoff
            # Merge
            good = good_overlaps & good_score

            for score, cls in zip(
                    mobjactions['scores'][good],
                    mobjactions['pred_classes'][good]):
                cls_scores[cls] = cls_scores.get(cls, 0.0) + score
        cls_scores_per_tube[k] = cls_scores
    return cls_scores_per_tube


def actual_eval_of_nicphil_etubes(workfolder, cfg_dict, add_args):
    """
    Evaluate "tubes" as if they were VOC objects
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    tubescores_dict: [~, ~]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    tube_nms:
        enabled: [True, bool]
        thresh: [0.5, float]
    eval:
        iou_thresholds: [[0.3, 0.5, 0.7], list]
        spatiotemporal: [False, bool]
        use_07_metric: [False, bool]
        use_diff: [False, bool]
    """)
    cf = cfg.parse()

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # // Obtain GT tubes
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    gt_tubes = get_daly_gt_tubes(dataset)
    gttubes_va = \
            _get_gt_sparsetubes(dataset, split_vids, gt_tubes)

    # // Obtain detected tubes
    tubes_per_video: \
            Dict[DALY_tube_index, DALY_wein_tube] = _set_tubes(cf, dataset)
    # Refer to the originals for start_frame/end_frame
    original_tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['tubes.imported_wein_tubes'])
    split_label = cf['dataset.subset']
    original_tubes_per_video = \
        get_subset_tubes(dataset, split_label, original_tubes_per_video)

    tubescores_dict = small.load_pkl(cf['tubescores_dict'])
    stubes_va = nicphil_evaluations_to_tubes(
            dataset, tubes_per_video,
            original_tubes_per_video, tubescores_dict)
    _daly_tube_map(cf, out, dataset, stubes_va, gttubes_va)


def actual_eval_of_action_object_predictions(workfolder, cfg_dict, add_args):
    """
    Evaluate "tubes" as if they were VOC objects
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    tube_nms:
        enabled: [True, bool]
        thresh: [0.5, float]
    eval:
        iou_thresholds: [[0.3, 0.5, 0.7], list]
        spatiotemporal: [False, bool]
        use_07_metric: [False, bool]
        use_diff: [False, bool]
    actobjects_evaluated: [~, ~]
    obj_to_tube:
        overlap_type: ['inner_overlap', ['inner_overlap', 'iou']]
        overlap_cutoff: [0.2, float]
        score_cutoff: [0.2, float]
    """)
    cf = cfg.parse()

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # // Obtain GT tubes
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    gt_tubes = get_daly_gt_tubes(dataset)
    gttubes_va = \
            _get_gt_sparsetubes(dataset, split_vids, gt_tubes)

    # // Obtain detected tubes
    tubes_per_video: \
            Dict[DALY_tube_index, DALY_wein_tube] = _set_tubes(cf, dataset)
    # Refer to the originals for start_frame/end_frame
    original_tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['tubes.imported_wein_tubes'])
    split_label = cf['dataset.subset']
    original_tubes_per_video = \
        get_subset_tubes(dataset, split_label, original_tubes_per_video)

    actobjects_evaluated = small.load_pkl(cf['actobjects_evaluated'])

    preds_per_framevid = _create_objdetection_helper_structure(
        dataset, split_label, actobjects_evaluated)
    overlap_type = cf['obj_to_tube.overlap_type']
    overlap_cutoff = cf['obj_to_tube.overlap_cutoff']
    score_cutoff = cf['obj_to_tube.score_cutoff']
    stubes_va = _match_objectdetections_to_tubes(
        dataset, tubes_per_video, original_tubes_per_video,
        preds_per_framevid, overlap_type, overlap_cutoff, score_cutoff)
    _daly_tube_map(cf, out, dataset, stubes_va, gttubes_va)


def assign_objactions_to_tubes(workfolder, cfg_dict, add_args):
    """
    Assign objactions (real or gt-faked) to tubes (philippe or gt-faked)
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]

    actobjects:
        source: ['detected', ['detected', 'gt']]
        detected:
            path: [~, ~]

    tubes:
        source: ['wein', ['wein', 'gt']]
        wein:
            path: [~, ~]
            filter_gt:
                enabled: [False, bool]
                keep_temporal: [True, bool]

    obj_to_tube:
        overlap_type: ['inner_overlap', ['inner_overlap', 'iou']]
        overlap_cutoff: [0.2, float]
        score_cutoff: [0.2, float]

    tube_eval:
        nms:
            enabled: [True, bool]
            thresh: [0.5, float]
        params:
            iou_thresholds: [[0.3, 0.5, 0.7], list]
            spatiotemporal: [False, bool]
            use_07_metric: [False, bool]
            use_diff: [False, bool]
    """)
    cf = cfg.parse()

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']

    # / GT tubes
    gt_tubes = get_daly_gt_tubes(dataset)
    split_vids = get_daly_split_vids(dataset, split_label)
    gttubes_va: Dict[DALY_action_name,
            Dict[DALY_vid, List[DALY_sparse_frametube]]] = \
            _get_gt_sparsetubes(dataset, split_vids, gt_tubes)

    # / Assign objects to tubes
    # // Create objaction_dets in video frames
    objactions_vf: Dict[DALY_vid, Dict[int, Objaction_dets]] = {}
    if cf['actobjects.source'] == 'detected':
        # /// Recreate the datalist that was used for detections
        action_object_to_object = get_similar_action_objects_DALY()
        new_object_names = sorted([x
            for x in set(list(action_object_to_object.values())) if x])
        datalist = simplest_daly_to_datalist(dataset, split_label)
        datalist = make_datalist_objaction_similar_merged(
                datalist, dataset.object_names, new_object_names,
                action_object_to_object)
        # /// Load detections themselves
        actobjects_evaluated = small.load_pkl(cf['actobjects.detected.path'])
        # /// Assign objactions
        for dl_item, pred_item in zip(datalist, actobjects_evaluated):
            pred_boxes = pred_item.pred_boxes.tensor.numpy()
            scores = pred_item.scores.numpy()
            pred_classes = pred_item.pred_classes.numpy()
            pred_classes = np.array([dataset.action_names[i]
                for i in pred_classes])
            detections: Objaction_dets = {
                    'pred_boxes': pred_boxes,
                    'scores': scores,
                    'pred_classes': pred_classes}
            (objactions_vf
                .setdefault(dl_item['vid'], {})
                [dl_item['video_frame_number']]) = detections
    elif cf['actobjects.source'] == 'gt':
        pass
    else:
        raise NotImplementedError()
    # // Define tubes
    tubes_per_video: Dict[DALY_tube_index, DALY_wein_tube]
    stubes_per_video: Dict[DALY_tube_index, DALY_sparse_frametube]
    if cf['tubes.source'] == 'wein':
        orig_tubes_per_video = small.load_pkl(cf['tubes.wein.path'])
        orig_tubes_per_video = get_subset_tubes_(
                split_vids, orig_tubes_per_video)
        if cf['tubes.wein.filter_gt.enabled']:
            tubes_per_video = filter_tube_keyframes_only_gt(
                    dataset, orig_tubes_per_video)
        else:
            tubes_per_video = orig_tubes_per_video
        # DALY_wein_tube -> DALY_sparse_frametube
        stubes_per_video = {}
        for k, tube in tubes_per_video.items():
            if cf['tubes.wein.filter_gt.enabled'] and \
                    cf['tubes.wein.filter_gt.keep_temporal']:
                otube = orig_tubes_per_video[k]
                temporal_frame_inds = otube['frame_inds']
            else:
                temporal_frame_inds = tube['frame_inds']
            stube: DALY_sparse_frametube = {
                'frame_inds': tube['frame_inds'],
                'boxes': tube['boxes'],
                'start_frame': temporal_frame_inds.min(),
                'end_frame': temporal_frame_inds.max()}
            stubes_per_video[k] = stube
    elif cf['tubes.source'] == 'gt':
        pass
    else:
        raise NotImplementedError()
    # // Assignment itself
    # /// Sum the scores
    overlap_type = cf['obj_to_tube.overlap_type']
    overlap_cutoff = cf['obj_to_tube.overlap_cutoff']
    score_cutoff = cf['obj_to_tube.score_cutoff']
    scores_t: Dict[DALY_tube_index, Dict[DALY_action_name, float]] = \
            _objectdetections_into_scores(
                objactions_vf, tubes_per_video,
                overlap_type, overlap_cutoff, score_cutoff)
    # /// Create "sparse scores tubes"
    stubes_va: Dict[DALY_action_name,
                Dict[DALY_vid, List[DALY_sparse_frametube_scored]]] = {}
    for ckey, stube in stubes_per_video.items():
        (vid, bunch_id, tube_id) = ckey
        scores: Dict[DALY_action_name, float] = scores_t[ckey]
        # Sum the perframe scores
        for action_name, score in scores.items():
            sstube = stube.copy()
            sstube['score'] = score
            if score < 0.05:
                continue
            (stubes_va
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(sstube))
    _daly_tube_map_v2(cf, out, dataset, stubes_va, gttubes_va)
