import argparse
import os
import yacs
import copy
import logging
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, cast
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
from detectron2.structures import BoxMode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import Boxes, Instances, pairwise_iou

from detectron2.utils.visualizer import ColorMode, Visualizer

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
from thes.experiments.horror_jan.nicphil import (
        _set_tubes, sample_some_tubes)


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


def eval_d2_framewise_action_detector(workfolder, cfg_dict, add_args):
    """
    Evaluation code with hacks
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    # from _set_cfg_defaults_pfadet_train
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', ['train', 'test']]
    """)
    cfg.set_deftype("""
    d2:
        SEED: [42, int]
    """)
    cfg.set_deftype("""
    what_to_eval: [~, str]
    nms:
        enable: [True, bool]
        batched: [False, bool]
        thresh: [0.3, float]
    conf_thresh: [0.0, float]
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
    num_classes = len(cls_names)

    # from _set_d2config_pfadet
    d_cfg = base_d2_frcnn_config()
    d_cfg.MODEL.WEIGHTS = cf['what_to_eval']
    set_d2_cthresh(d_cfg, cf['conf_thresh'])
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    yacs_add_d2 = yacs.config.CfgNode(
            snippets.unflatten_nested_dict(cf_add_d2), [])
    d_cfg.merge_from_other_cfg(yacs_add_d2)
    d_cfg.freeze()

    # evaluation
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


# def eval_daly_tubes_RGB_with_pfadet(workfolder, cfg_dict, add_args):
#     """
#     Run out own trained model on tubes
#     """
#     out, = snippets.get_subfolders(workfolder, ['out'])
#     cfg = snippets.YConfig(cfg_dict)
#     # _set_tubecfg
#     cfg.set_deftype("""
#     dataset:
#         name: [~, ['daly']]
#         cache_folder: [~, str]
#         subset: ['train', str]
#     tubes:
#         imported_wein_tubes: [~, ~]
#         filter_gt: [False, bool]
#     """)
#     _set_tubecfg(cfg)
#     cfg.set_deftype("""
#     compute:
#         chunk: [0, "VALUE >= 0"]
#         total: [1, int]
#     save_period: ['::10', str]
#     """)
