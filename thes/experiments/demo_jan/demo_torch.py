import cv2
import sys
import copy
import torch
import logging
import logging.handlers
import threading
import argparse
import time
import yacs
import re
import yaml
import collections
import pprint
import os
import random
import numpy as np
import pandas as pd
import multiprocessing
from collections import OrderedDict
from collections import deque
from tqdm import tqdm
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Dict, List, Tuple
from mypy_extensions import TypedDict

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image
from detectron2.engine import (
        DefaultTrainer, default_argument_parser,
        default_setup, hooks, launch)
from detectron2.data.datasets.pascal_voc import register_pascal_voc
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
import detectron2.utils.comm as comm
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as d2_dutils
from detectron2.data import transforms as d2_transforms
from detectron2.data import (
        build_detection_train_loader,
        build_detection_test_loader)
from PIL import Image
from fvcore.common.file_io import PathManager

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.data import video_utils
from thes.eval_tools import legacy_evaluation
from thes.tools import snippets
from thes.data.external_dataset import DatasetVOC2007, DatasetDALY
from thes.det2 import (
        YAML_Base_RCNN_C4, YAML_faster_rcnn_R_50_C4,
        launch_w_logging, launch_without_logging, simple_d2_setup,
        base_d2_frcnn_config, set_d2_cthresh, D2DICT_GPU_SCALING_DEFAULTS,
        get_frame_without_crashing)
from thes.daly_d2 import simplest_daly_to_datalist


log = logging.getLogger(__name__)


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes
            from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata,
                instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(
                        predictions=instances)

        return predictions, vis_output


def load_voc2007(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    imported_wein_tubes: [~, str]
    dataset:
        name: [~, ['voc2007']]
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    dataset = DatasetVOC2007()
    dataset.populate_from_folder(cf['dataset.cache_folder'])


def eval_detectron2_rcnn_visdemo(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    imported_wein_tubes: [~, str]
    dataset:
        name: [~, ['voc2007']]
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    dataset = DatasetVOC2007()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    CONFIDENCE_THRESHOLD = 0.25

    DETECTRON_CONFIG_PATH = '/scratch/horus/vsydorov/bulk/deployed/2019_12_Thesis/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml'
    PRETRAINED_WEIGHTS = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/horus/pytorch_model_zoo/pascal_voc_baseline/model_final_b1acc2.pkl'
    d_cfg = get_cfg()
    d_cfg.merge_from_file(DETECTRON_CONFIG_PATH)
    d_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
            CONFIDENCE_THRESHOLD
    d_cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS
    d_cfg.freeze()
    demo = VisualizationDemo(d_cfg)

    test_split = dataset.annotations_per_split['test']
    k10 = list(test_split.keys())[:10]
    filepaths = []
    for k in k10:
        anno = test_split[k]
        filepaths.append(str(anno['filepath']))

    output_fold = str(small.mkdir(out/'img_outputs'))
    WINDOW_NAME = "COCO detections"

    for path in tqdm(filepaths):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        log.info(
            "{}: detected {} instances in {:.2f}s".format(
                path, len(predictions["instances"]), time.time() - start_time
            )
        )

        if output_fold is not None:
            out_filename = os.path.join(output_fold, os.path.basename(path))
            visualized_output.save(out_filename)
        else:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, d_cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(d_cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=d_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=d_cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(
                dataset_name, d_cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(
                dataset_name, output_folder))
        elif evaluator_type == "cityscapes":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, d_cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, d_cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(d_cfg, model)
        evaluators = [
            cls.build_evaluator(
                d_cfg, name, output_folder=os.path.join(d_cfg.OUTPUT_DIR,
                    "inference_TTA")
            )
            for name in d_cfg.DATASETS.TEST
        ]
        res = cls.test(d_cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def eval_detectron2_rcnn_voc07eval(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    imported_wein_tubes: [~, str]
    dataset:
        name: [~, ['voc2007']]
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    dataset = DatasetVOC2007()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    CONFIDENCE_THRESHOLD = 0.25
    DETECTRON_CONFIG_PATH = '/scratch/horus/vsydorov/bulk/deployed/2019_12_Thesis/detectron2/configs/PascalVOC-Detection/faster_rcnn_R_50_C4.yaml'
    PRETRAINED_WEIGHTS = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/horus/pytorch_model_zoo/pascal_voc_baseline/model_final_b1acc2.pkl'
    d_cfg = get_cfg()
    d_cfg.merge_from_file(DETECTRON_CONFIG_PATH)
    d_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
            CONFIDENCE_THRESHOLD
    d_cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'output_dir'))
    # Change datasets
    d_cfg.DATASETS.TRAIN = (
            'voc_2007_trainval_absolute',
            'voc_2012_trainval_absolute')
    d_cfg.DATASETS.TEST = ('voc_2007_test_absolute',)
    d_cfg.freeze()

    class ARGS:
        resume = True

    # Register "absolute" VOC
    VOC_ROOT = '/home/vsydorov/projects/datasets/detection'
    SPLITS = [
        ("voc_2007_test_absolute", "VOC2007/VOCdevkit/VOC2007", "test"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(VOC_ROOT, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    model = Trainer.build_model(d_cfg)
    DetectionCheckpointer(
            model, save_dir=d_cfg.OUTPUT_DIR
            ).resume_or_load(d_cfg.MODEL.WEIGHTS, resume=ARGS.resume)
    res = Trainer.test(d_cfg, model)
    if comm.is_main_process():
        verify_results(d_cfg, res)
    if d_cfg.TEST.AUG.ENABLED:
        res.update(Trainer.test_with_TTA(d_cfg, model))


def _train_func(d_cfg, cf, args):
    simple_d2_setup(d_cfg)

    VOC_ROOT = '/home/vsydorov/projects/datasets/detection'
    SPLITS = [
        ("voc_2007_trainval_absolute", "VOC2007/VOCdevkit/VOC2007", "trainval"),
        ("voc_2007_test_absolute", "VOC2007/VOCdevkit/VOC2007", "test"),
        ("voc_2012_trainval_absolute", "VOC2012/VOCdevkit/VOC2012", "trainval"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(VOC_ROOT, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

    trainer = Trainer(d_cfg)
    trainer.resume_or_load(resume=cf['resume'])
    if d_cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(d_cfg, trainer.model))]
        )
    trainer.train()


def train_d2_rcnn_voc0712(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    resume: [True, bool]
    num_gpus: [~, int]
    solver_gpuscale: [true, bool]
    """)
    cf = cfg.parse()

    CONFIDENCE_THRESHOLD = 0.25
    d_cfg = get_cfg()
    # Base keys
    loaded_cfg = yacs.config.CfgNode.load_cfg(YAML_Base_RCNN_C4)
    d_cfg.merge_from_other_cfg(loaded_cfg)
    # FRCCN keys
    loaded_cfg = yacs.config.CfgNode.load_cfg(YAML_faster_rcnn_R_50_C4)
    d_cfg.merge_from_other_cfg(loaded_cfg)
    # Manual overwrite
    d_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
            CONFIDENCE_THRESHOLD
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    # d_cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS
    # 2GPUs only
    if cf['solver_gpuscale']:
        d_cfg.SOLVER.BASE_LR = 0.0025 * cf['num_gpus']
        d_cfg.SOLVER.IMS_PER_BATCH = 2 * cf['num_gpus']
        imult = 8 / cf['num_gpus']
        d_cfg.SOLVER.STEPS = \
            (np.r_[12000, 16000] * imult).astype(int).tolist()
        d_cfg.SOLVER.MAX_ITER = \
            int(18000 * imult)
        d_cfg.WARMUP_ITERS = int(100 * imult)
    else:
        raise NotImplementedError()
    # Change datasets
    d_cfg.DATASETS.TRAIN = (
            'voc_2007_trainval_absolute',
            'voc_2012_trainval_absolute')
    d_cfg.DATASETS.TEST = ('voc_2007_test_absolute',)
    d_cfg.freeze()

    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    dist_url = "tcp://127.0.0.1:{}".format(port)

    args = argparse.Namespace()

    launch_w_logging(_train_func,
            cf['num_gpus'],
            num_machines=1,
            machine_rank=0,
            dist_url=dist_url,
            args=(d_cfg, cf, args))


def demo_d2_dalyobj_vis(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    resume: [True, bool]
    num_gpus: [~, int]
    solver_gpuscale: [true, bool]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = simplest_daly_to_datalist(dataset, split)
        datalist_per_split[split] = datalist

    for split, datalist in datalist_per_split.items():
        d2_dataset_name = f'dalyobjects_{split}'
        DatasetCatalog.register(d2_dataset_name,
                lambda split=split: datalist_per_split[split])
        MetadataCatalog.get(d2_dataset_name).set(
                thing_classes=dataset.object_names)

    # Visualize
    metadata = MetadataCatalog.get("dalyobjects_train")
    train_datalist = datalist_per_split['train']

    NP_SEED = 0
    np_rstate = np.random.RandomState(NP_SEED)
    prm_train_datalist = np_rstate.permutation(train_datalist)
    dl_items = prm_train_datalist[:50]

    for i, dl_item in enumerate(dl_items):
        video_path = dl_item['video_path']
        frame_number = dl_item['video_frame_number']
        with vt_cv.video_capture_open(video_path) as vcap:
            frame_u8 = vt_cv.video_sample(vcap, [frame_number])[0]

        visualizer = Visualizer(frame_u8, metadata=metadata, scale=0.5)
        img_vis = visualizer.draw_dataset_dict(dl_item)
        filename = 'i{:02d}_id.jpg'.format(i, dl_item['image_id'])
        cv2.imwrite(str(out/filename), img_vis.get_image())


def get_single_frame_robustly(video_path, frame_number, OVERALL_ATTEMPTS):
    def _get(video_path, frame_number):
        with vt_cv.video_capture_open(video_path) as vcap:
            frame_u8 = vt_cv.video_sample(
                    vcap, [frame_number], debug_filename=video_path)[0]
        return frame_u8

    i = 0
    while i < OVERALL_ATTEMPTS:
        try:
            frame_u8 = _get(video_path, frame_number)
            return frame_u8
        except (IOError, vt_cv.VideoCaptureError) as e:
            mp_name = multiprocessing.current_process().name
            WARN_MESSAGE = 'Caught "{}", retrying {}/{}. File {} frame {} mp_name {}'.format(
                    e, i, OVERALL_ATTEMPTS, video_path, frame_number, mp_name)
            d2_logger = logging.getLogger('detectron2')
            log.warning(WARN_MESSAGE)
            d2_logger.warning(WARN_MESSAGE)
            print(WARN_MESSAGE)
            time.sleep(1)
            i += 1
    raise IOError('Never managed to open {}, frame {}'.format(
        video_path, frame_number))


def vis_evaldemo_dalyobj(predictor, datalist, metadata, visfold, size=50):
    NP_SEED = 0
    np_rstate = np.random.RandomState(NP_SEED)
    prm_train_datalist = np_rstate.permutation(datalist)
    dl_items = prm_train_datalist[:size]

    cpu_device = torch.device("cpu")
    for i, dl_item in tqdm(enumerate(dl_items)):
        video_path = dl_item['video_path']
        frame_number = dl_item['video_frame_number']
        frame_time = dl_item['video_frame_number']
        frame_u8 = get_frame_without_crashing(
            video_path, frame_number, frame_time)

        # Detect

        predictions = predictor(frame_u8)
        instances = predictions["instances"].to(cpu_device)

        visualizer1 = Visualizer(frame_u8, metadata=metadata, scale=0.5)
        vis_output_gt = visualizer1.draw_instance_predictions(predictions=instances)

        visualizer2 = Visualizer(frame_u8, metadata=metadata, scale=0.5)
        vis_output_det = visualizer2.draw_dataset_dict(dl_item)

        img1 = vis_output_gt.get_image()
        img2 = vis_output_det.get_image()
        img_side_by_side = np.hstack((img1, img2))

        filename = 'i{:02d}_id{}_side.jpg'.format(i, dl_item['image_id'])
        cv2.imwrite(str(visfold/filename), img_side_by_side)


def evaldemo_d2_dalyobj_old(workfolder, cfg_dict, add_args):
    """
    Just a demo
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
    conf_thresh: [0.0, float]
    model_to_eval: [~, str]
    subset: [train, [train, test]]
    N: [50, int]
    seed: [42, int]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # D2 dataset compatible list of keyframes
    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = simplest_daly_to_datalist(dataset, split)
        datalist_per_split[split] = datalist

    for split, datalist in datalist_per_split.items():
        d2_dataset_name = f'dalyobjects_{split}'
        DatasetCatalog.register(d2_dataset_name,
                lambda split=split: datalist_per_split[split])
        MetadataCatalog.get(d2_dataset_name).set(
                thing_classes=dataset.object_names)

    # d2_config
    d_cfg = base_d2_frcnn_config()
    set_d2_cthresh(d_cfg, cf['conf_thresh'])
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.WEIGHTS = cf['model_to_eval']
    d_cfg.DATASETS.TRAIN = ()
    d_cfg.DATASETS.TEST = ('dalyobjects_test',)
    # Different number of classes
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 43
    # Defaults:
    d_cfg.SEED = cf['seed']
    d_cfg.freeze()

    # Visualizer
    predictor = DefaultPredictor(d_cfg)

    subset = cf['subset']
    if subset == 'train':
        datalist = datalist_per_split['train']
        metadata = MetadataCatalog.get("dalyobjects_train")
        visfold = small.mkdir(out/'exp3/train')
    elif subset == 'test':
        datalist = datalist_per_split['test']
        metadata = MetadataCatalog.get("dalyobjects_test")
        visfold = small.mkdir(out/'exp3/test')
    else:
        raise RuntimeError()
    vis_evaldemo_dalyobj(predictor, datalist, metadata, visfold, cf['N'])
