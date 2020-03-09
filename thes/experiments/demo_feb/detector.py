import argparse
import os
import yacs
import copy
import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, cast, List
from tqdm import tqdm

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
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances, pairwise_iou

from detectron2.utils.visualizer import ColorMode, Visualizer

from thes.data.external_dataset import (
        DatasetDALY, DALY_action_name, DALY_object_name,
        DALY_vid)
from thes.tools import snippets
from thes.det2 import (
        YAML_Base_RCNN_C4, YAML_faster_rcnn_R_50_C4,
        launch_w_logging, launch_without_logging, simple_d2_setup,
        base_d2_frcnn_config, set_d2_cthresh, d2dict_gpu_scaling,
        get_frame_without_crashing,)
from thes.daly_d2 import (
        simplest_daly_to_datalist,
        get_daly_split_vids,
        DalyVideoDatasetMapper,
        gt_tubes_to_df,
        get_daly_gt_tubes)
from thes.eval_tools import legacy_evaluation
from thes.experiments.horror_jan.nicphil import (
        _set_tubes, sample_some_tubes, _get_gt_sparsetubes,
        get_subset_tubes, DALY_sparse_frametube_scored,
        _daly_tube_map)
from thes.experiments.demo_december.load_daly import (
        DALY_wein_tube, DALY_tube_index)


log = logging.getLogger(__name__)


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


PRETRAINED_WEIGHTS_MODELPATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/horus/pytorch_model_zoo/pascal_voc_baseline/model_final_b1acc2.pkl'


def train_d2_framewise_action_detector(workfolder, cfg_dict, add_args):
    raise NotImplementedError('Refactoring in progress')


def eval_d2_framewise_action_detector(workfolder, cfg_dict, add_args):
    pass


def _cpu_and_thresh_instances(predictions, post_thresh, cpu_device):
    instances = predictions["instances"].to(cpu_device)
    good_scores = instances.scores > post_thresh
    tinstances = instances[good_scores]
    return tinstances


def _d2vis_draw_gtboxes(frame_u8, tinstances, metadata):
    visualizer = Visualizer(frame_u8, metadata=metadata, scale=1)
    img_vis = visualizer.draw_instance_predictions(
            predictions=tinstances)
    img = img_vis.get_image()
    return img


def _predict_rcnn_given_box_resized_proposals(
        box4, frame_u8, transform_gen, model):

    o_height, o_width = frame_u8.shape[:2]
    got_transform = transform_gen.get_transform(frame_u8)

    # Transform image
    image = got_transform.apply_image(frame_u8)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    imshape = tuple(image.shape[1:3])

    # / Transform box
    assert box4.shape == (4,)
    boxes_unscaled = box4[None]
    t_boxes = torch.as_tensor(boxes_unscaled.astype("float32"))
    transformed_t_boxes = got_transform.apply_box(t_boxes)
    # // Proposals w.r.t transformed imagesize
    proposal = Instances(imshape)
    tb_boxes = Boxes(transformed_t_boxes)
    proposal.proposal_boxes = tb_boxes

    inputs = {
            "image": image,
            "proposals": proposal,
            "height": o_height,
            "width": o_width}

    with torch.no_grad():
        predictions = model([inputs])[0]
    return predictions


def eval_daly_tubes_RGB_with_pfadet_demovis(workfolder, cfg_dict, add_args):
    """
    Run out own trained model on tubes
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    """)
    cfg.set_deftype("""
    compute:
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
    save_period: ['::10', str]
    """)
    cfg.set_deftype("""
    some_tubes:
        N: [50, int]
        seed: [0, int]
    conf_thresh: [0.0, float]
    trained_d2_model: [~, ~]
    d2:
        SEED: [42, int]
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    cls_names = dataset.action_names
    num_classes = len(cls_names)

    # Dataset
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)
    name = 'daly_pfadet_train'
    DatasetCatalog.register(name, lambda: datalist)
    MetadataCatalog.get(name).set(
            thing_classes=cls_names)
    metadata = MetadataCatalog.get(name)

    # / Define d2 conf
    d_cfg = base_d2_frcnn_config()
    d_cfg.MODEL.WEIGHTS = cf['trained_d2_model']
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    set_d2_cthresh(d_cfg, cf['conf_thresh'])
    yacs_add_d2 = yacs.config.CfgNode(
            snippets.unflatten_nested_dict(cf_add_d2), [])
    d_cfg.merge_from_other_cfg(yacs_add_d2)
    d_cfg2 = copy.deepcopy(d_cfg)
    d_cfg.freeze()
    # / Start d2
    simple_d2_setup(d_cfg)

    # Predictor with proposal generator
    predictor = DefaultPredictor(d_cfg)

    # Predictor without proposal generator
    d_cfg2.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
    d_cfg2.freeze()
    model = build_model(d_cfg2)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(d_cfg2.MODEL.WEIGHTS)
    MIN_SIZE_TEST = d_cfg2.INPUT.MIN_SIZE_TEST
    MAX_SIZE_TEST = d_cfg2.INPUT.MAX_SIZE_TEST
    transform_gen = d2_transforms.ResizeShortestEdge(
        [MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST)
    cpu_device = torch.device("cpu")

    # Load tubes
    tubes_per_video = _set_tubes(cf, dataset)
    some_tubes = sample_some_tubes(
            tubes_per_video, N=cf['some_tubes.N'],
            NP_SEED=cf['some_tubes.seed'])
    # k = ('S_PwpNZWgpk', 0, 19)
    # some_tubes = {k: tubes_per_video[k]}

    post_thresh = 0.2

    for k, tube in tqdm(some_tubes.items(), 'tubes vis'):
        (vid, bunch_id, tube_id) = k
        vmp4 = dataset.source_videos[vid]
        video_path = vmp4['video_path']
        frame_inds = tube['frame_inds']
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)
        tube_prefix = '{}_B{:02d}T{:02d}'.format(vid, bunch_id, tube_id)
        tube_fold = small.mkdir(out/tube_prefix)
        for i, (frame_ind, frame_u8) in enumerate(zip(frame_inds, frames_u8)):
            frame_prefix = tube_fold/'I{}_F{:03d}'.format(i, frame_ind)

            predictions = predictor(frame_u8)
            tinstances = \
                    _cpu_and_thresh_instances(predictions, post_thresh, cpu_device)
            img = _d2vis_draw_gtboxes(frame_u8, tinstances, metadata)
            cv2.imwrite(str(out/f'{frame_prefix}_frcnn.jpg'), img)

            # Get tube box, pass tube box through the rcnn part
            box4 = tube['boxes'][i]
            predictions = _predict_rcnn_given_box_resized_proposals(
                    box4, frame_u8, transform_gen, model)
            tinstances = \
                    _cpu_and_thresh_instances(predictions, post_thresh, cpu_device)
            img = _d2vis_draw_gtboxes(frame_u8, tinstances, metadata)
            snippets.cv_put_box_with_text(
                    img, box4, text='philtube')
            cv2.imwrite(str(out/f'{frame_prefix}_rcnn.jpg'), img)


def equal_tube_split(tubes_per_video, ct, split_kind):
    key_indices = np.arange(len(tubes_per_video))
    key_list = list(tubes_per_video.keys())

    # Simple tube df
    nframes_df = []
    for k, v in tubes_per_video.items():
        vid = k[0]
        nframes = len(v['frame_inds'])
        nframes_df.append([vid, nframes])
    nframes_df = pd.DataFrame(nframes_df, columns=['vid', 'nframes'])
    nframes_df['keys'] = key_list

    # Divide indices
    if split_kind == 'tubes':
        equal_split = np.array_split(key_indices, ct)
    elif split_kind == 'frames':
        approx_nframes_per_split = nframes_df.nframes.sum() // ct
        approx_split_indices = approx_nframes_per_split * np.arange(1, ct)
        split_indices = np.searchsorted(
                nframes_df.nframes.cumsum(), approx_split_indices)
        equal_split = np.array_split(key_indices, split_indices)
    else:
        raise NotImplementedError()

    # Assign splits
    for i, inds in enumerate(equal_split):
        nframes_df.loc[inds, 'split'] = i
    nframes_df['split'] = nframes_df['split'].astype(int)

    # Compute stats
    gb_chunk = nframes_df.groupby('split')
    all_nvids = gb_chunk['vid'].unique().apply(len)
    all_nframes = gb_chunk['nframes'].sum()
    split_stats = pd.concat((all_nvids, all_nframes), axis=1)

    # Divide tubes
    split_tubes = [{} for i in range(ct)]
    for i, group in gb_chunk.groups.items():
        keys = nframes_df.loc[group, 'keys'].tolist()
        for k in keys:
            split_tubes[i][k] = tubes_per_video[k]
    return split_tubes, split_stats


def _parcel_management(cf, tubes_per_video):
    # // Computation of parcels
    cc, ct = (cf['compute.chunk'], cf['compute.total'])
    split_kind = cf['compute.equal_split']
    split_tubes, split_stats = \
            equal_tube_split(tubes_per_video, ct, split_kind)
    ctubes_per_video = split_tubes[cc]
    # Logging part
    log.info('Chunk {}/{}: {} -> {}'.format(
        cc, ct, len(tubes_per_video), len(ctubes_per_video)))
    log.info('split_stats:\n{}'.format(split_stats))
    return ctubes_per_video


def eval_daly_tubes_RGB_with_pfadet(workfolder, cfg_dict, add_args):
    """
    Run out own trained model on tubes
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    # _set_tubecfg
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    """)
    cfg.set_deftype("""
    compute:
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
        equal_split: ['frames', ['frames', 'tubes']]
    save_period: ['::10', str]
    """)
    cfg.set_deftype("""
    some_tubes:
        N: [50, int]
        seed: [0, int]
    conf_thresh: [0.0, float]
    trained_d2_model: [~, ~]
    d2:
        SEED: [42, int]
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    cls_names = dataset.action_names
    num_classes = len(cls_names)

    # Dataset
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    datalist = daly_to_datalist_pfadet(dataset, split_vids)
    name = 'daly_pfadet_train'
    DatasetCatalog.register(name, lambda: datalist)
    MetadataCatalog.get(name).set(
            thing_classes=cls_names)
    metadata = MetadataCatalog.get(name)

    # / Define d2 conf
    d_cfg = base_d2_frcnn_config()
    d_cfg.MODEL.WEIGHTS = cf['trained_d2_model']
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    d_cfg.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
    set_d2_cthresh(d_cfg, cf['conf_thresh'])
    yacs_add_d2 = yacs.config.CfgNode(
            snippets.unflatten_nested_dict(cf_add_d2), [])
    d_cfg.merge_from_other_cfg(yacs_add_d2)
    d_cfg.freeze()
    # / Start d2
    simple_d2_setup(d_cfg)

    # Predictor without proposal generator
    model = build_model(d_cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(d_cfg.MODEL.WEIGHTS)
    MIN_SIZE_TEST = d_cfg.INPUT.MIN_SIZE_TEST
    MAX_SIZE_TEST = d_cfg.INPUT.MAX_SIZE_TEST
    transform_gen = d2_transforms.ResizeShortestEdge(
        [MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST)
    cpu_device = torch.device("cpu")

    # Load tubes
    tubes_per_video = _set_tubes(cf, dataset)
    ctubes_per_video = _parcel_management(cf, tubes_per_video)

    def tube_eval_func(k):
        tube = ctubes_per_video[k]
        (vid, bunch_id, tube_id) = k
        vmp4 = dataset.source_videos[vid]
        video_path = vmp4['video_path']
        frame_inds = tube['frame_inds']
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)

        instances_per_frame = []
        for i, (frame_ind, frame_u8) in enumerate(zip(frame_inds, frames_u8)):
            # Get tube box, pass tube box through the rcnn part
            box4 = tube['boxes'][i]
            predictions = _predict_rcnn_given_box_resized_proposals(
                    box4, frame_u8, transform_gen, model)
            instances = predictions["instances"].to(cpu_device)
            # Simply record all predictions
            instances_per_frame.append(instances)
        return instances_per_frame

    df_isaver = snippets.Simple_isaver(
            small.mkdir(out/'tube_eval_isaver'),
            list(ctubes_per_video.keys()),
            tube_eval_func, cf['save_period'], 120)
    predicted_tube_instances = df_isaver.run()
    tube_instances_dict = dict(zip(
        ctubes_per_video.keys(),
        predicted_tube_instances))
    small.save_pkl(out/'tube_instances_dict.pkl', tube_instances_dict)


def eval_daly_tubes_RGB_with_pfadet_gather_evaluated(
        workfolder, cfg_dict, add_args):
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
    for tubepath in tqdm(cf['etubes'], 'loading etubes'):
        tubes = small.load_pkl(tubepath)
        tubescores_dict.update(tubes)

    # Confirm that keys match
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    tubes_per_video = _set_tubes(cf, dataset)

    assert tubes_per_video.keys() == tubescores_dict.keys(), \
            "Keys should match"

    small.save_pkl(out/'tube_instances_dict.pkl', tubescores_dict)


def map_score_tubes_and_pfadet_rcnn_scores(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    tube_instances_dict: [~, ~]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    rcnn_assignment:
        use_boxes: [False, bool]
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

    tube_instances_dict = small.load_pkl(cf['tube_instances_dict'])
    stubes_va: \
            Dict[DALY_action_name,
                    Dict[DALY_vid, List[DALY_sparse_frametube_scored]]] = {}

    assert cf['rcnn_assignment.use_boxes'] is False
    # # Only record scores > 0.01
    # score_record_thresh = 0.01
    for ckey, tube in tubes_per_video.items():
        (vid, bunch_id, tube_id) = ckey
        original_tube = original_tubes_per_video[ckey]
        tube_instances = tube_instances_dict[ckey]
        # Ignore boxes in tube instances
        scores_per_actid = np.zeros(len(dataset.action_names))
        for i, ins in enumerate(tube_instances):
            for pred_cls, score in zip(ins.pred_classes, ins.scores):
                scores_per_actid[pred_cls] += score
        start_frame = original_tube['frame_inds'].min()
        end_frame = original_tube['frame_inds'].max()
        for action_name, score in zip(
                dataset.action_names, scores_per_actid):
            sparse_scored_tube = {
                    'frame_inds': tube['frame_inds'],
                    'boxes': tube['boxes'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'score': score}
            (stubes_va
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(sparse_scored_tube))
    _daly_tube_map(cf, out, dataset, stubes_va, gttubes_va)
