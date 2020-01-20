from abc import abstractmethod, ABC
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
from thes.tools import snippets
from thes.data.external_dataset import DatasetVOC2007, DatasetDALY
from thes.det2 import (
        launch_w_logging, launch_without_logging, simple_d2_setup)


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


# Base-RCNN-C4.yaml
YAML_Base_RCNN_C4 = """
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  RPN:
    PRE_NMS_TOPK_TEST: 6000
    POST_NMS_TOPK_TEST: 1000
  ROI_HEADS:
    NAME: "Res5ROIHeads"
DATASETS:
  TRAIN: ("coco_2017_train",)
  TEST: ("coco_2017_val",)
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.02
  STEPS: (60000, 80000)
  MAX_ITER: 90000
INPUT:
  MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
VERSION: 2
"""
# faster_rcnn_R_50_C4.yaml
YAML_faster_rcnn_R_50_C4 = """
MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  MASK_ON: False
  RESNETS:
    DEPTH: 50
  ROI_HEADS:
    NUM_CLASSES: 20
INPUT:
  MIN_SIZE_TRAIN: (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
  MIN_SIZE_TEST: 800
DATASETS:
  TRAIN: ('voc_2007_trainval', 'voc_2012_trainval')
  TEST: ('voc_2007_test',)
SOLVER:
  STEPS: (12000, 16000)
  MAX_ITER: 18000  # 17.4 epochs
  WARMUP_ITERS: 100
"""


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


def daly_to_datalist(dataset, split_label):
    split_vids = [vid for vid, split in dataset.split.items()
            if split == split_label]

    if split_label == 'train':
        split_size = 310
    elif split_label == 'test':
        split_size = 200
    else:
        split_size = None
    assert len(split_vids) == split_size

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
                    kf_objects = keyframe['objects']
                    annotations = []
                    for kfo in kf_objects:
                        [xmin, ymin, xmax, ymax,
                            objectID, isOccluded, isHallucinate] = kfo
                        isOccluded = bool(isOccluded)
                        isHallucinate = bool(isHallucinate)
                        if isHallucinate:
                            continue
                        box_unscaled = np.array([xmin, ymin, xmax, ymax])
                        bbox = box_unscaled * np.tile([width, height], 2)
                        bbox_mode = BoxMode.XYXY_ABS
                        obj = {
                                'bbox': bbox,
                                'bbox_mode': bbox_mode,
                                'category_id': int(objectID),
                                'is_occluded': isOccluded}
                        annotations.append(obj)
                    if len(annotations) == 0:
                        continue
                    record = {
                            'video_path': video_path,
                            'video_frame_number': frame_number,
                            'video_frame_time': frame_time,
                            'image_id': image_id,
                            'height': vmp4['height'],
                            'width': vmp4['width'],
                            'annotations': annotations}
                    d2_datalist.append(record)
    return d2_datalist


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
        datalist = daly_to_datalist(dataset, split)
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


def get_frame_without_crashing(
        video_path, frame_number, frame_time,
        OCV_ATTEMPTS=3,
        PYAV_ATTEMPTS=0):
    """
    Plz don't crash
    """
    MP_NAME = multiprocessing.current_process().name
    THREAD_NAME = threading.get_ident()

    def _get_via_opencv(video_path, frame_number):
        with vt_cv.video_capture_open(video_path, tries=2) as vcap:
            vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame_BGR = vcap.retrieve()
            if ret == 0:
                raise OSError(f"Can't read frame {frame_number} from {video_path}")
        return frame_BGR

    def _get_via_pyav(video_path, frame_time):
        raise NotImplementedError()
        # import torchvision.io
        # pyav_video = torchvision.io.read_video(
        #         video_path, pts_unit='sec', start_pts=frame_time)
        # return frame_BGR

    def _fail_message(via, e, i, attempts):
        MESSAGE = (
            'Failed frame read via {} mp_name {} thread {}. '
            'Caught "{}", retrying {}/{}. '
            'File {} frame {}').format(
                    via, MP_NAME, THREAD_NAME,
                    e, i, attempts,
                    video_path, frame_number)
        log.warning('WARN ' + MESSAGE)

    for i in range(OCV_ATTEMPTS):
        try:
            frame_u8 = _get_via_opencv(video_path, frame_number)
            return frame_u8
        except (IOError, RuntimeError, NotImplementedError) as e:
            _fail_message('opencv', e, i, OCV_ATTEMPTS)
            time.sleep(1)

    for i in range(PYAV_ATTEMPTS):
        try:
            frame_u8 = _get_via_pyav(video_path, frame_time)
            return frame_u8
        except (IOError, RuntimeError, NotImplementedError) as e:
            _fail_message('pyav', e, i, OCV_ATTEMPTS)
            time.sleep(1)

    raise IOError('Never managed to open {}, f_num {} f_time {}'.format(
        video_path, frame_number, frame_time))


class DalyVideoDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = d2_transforms.RandomCrop(
                    cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info(
                    "CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = d2_dutils.build_transform_gen(cfg, is_train)
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # it will be modified by code below
        dataset_dict = copy.deepcopy(dataset_dict)

        # Robust video sampling
        video_path = dataset_dict['video_path']
        frame_number = dataset_dict['video_frame_number']
        frame_time = dataset_dict['video_frame_time']
        OVERALL_ATTEMPTS = 5

        image = get_frame_without_crashing(
                video_path, frame_number, frame_time, OVERALL_ATTEMPTS)

        d2_dutils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = d2_transforms.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = d2_dutils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = \
                    d2_transforms.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        ).contiguous()
        # Can use uint8 if it turns out to be slow some day

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other types of data
            annos = [
                d2_dutils.transform_instance_annotations(
                    obj, transforms, image_shape,
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = d2_dutils.annotations_to_instances(
                annos, image_shape,
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = d2_dutils.filter_empty_instances(instances)
        return dataset_dict


class DalyObjTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = DalyVideoDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg,
                mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        mapper = DalyVideoDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name,
                mapper=mapper)

    # @classmethod
    # def build_evaluator(cls, d_cfg, dataset_name, output_folder=None):
    #     if output_folder is None:
    #         output_folder = os.path.join(d_cfg.OUTPUT_DIR, "inference")
    #     return PascalVOCDetectionEvaluator(dataset_name)


def _train_func_dalyobj(d_cfg, cf, args,):
    simple_d2_setup(d_cfg)
    datalist_per_split = args.datalist_per_split
    dataset = args.dataset

    for split, datalist in datalist_per_split.items():
        d2_dataset_name = f'dalyobjects_{split}'
        DatasetCatalog.register(d2_dataset_name,
                lambda split=split: datalist_per_split[split])
        MetadataCatalog.get(d2_dataset_name).set(
                thing_classes=dataset.object_names)

    trainer = DalyObjTrainer(d_cfg)
    trainer.resume_or_load(resume=args.resume)
    if d_cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(d_cfg, trainer.model))]
        )
    trainer.train()


def base_d2_frcnn_config():
    d_cfg = get_cfg()
    # Base keys
    loaded_cfg = yacs.config.CfgNode.load_cfg(YAML_Base_RCNN_C4)
    d_cfg.merge_from_other_cfg(loaded_cfg)
    # FRCCN keys
    loaded_cfg = yacs.config.CfgNode.load_cfg(YAML_faster_rcnn_R_50_C4)
    d_cfg.merge_from_other_cfg(loaded_cfg)
    return d_cfg


def set_d2_cthresh(d_cfg, CONFIDENCE_THRESHOLD=0.25):
    d_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
            CONFIDENCE_THRESHOLD


def d2dict_gpu_scaling(cf, cf_add_d2, num_gpus):
    imult = 8 / num_gpus
    prepared = {}
    prepared['SOLVER.BASE_LR'] = \
        cf['gpu_scaling.base.SOLVER.BASE_LR'] * num_gpus
    prepared['SOLVER.IMS_PER_BATCH'] = \
        cf['gpu_scaling.base.SOLVER.IMS_PER_BATCH'] * num_gpus
    prepared['SOLVER.MAX_ITER'] = \
        int(cf['gpu_scaling.base.SOLVER.MAX_ITER'] * imult)
    prepared['SOLVER.STEPS'] = tuple((np.array(
            cf['gpu_scaling.base.SOLVER.STEPS']
            ) * imult).astype(int).tolist())
    return cf_add_d2.update(prepared)


def train_d2_dalyobj(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    num_gpus: [~, int]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
    seed: [42, int]
    """)
    cfg.set_defaults("""
    gpu_scaling:
        enabled: True
        base:
            # more per gpu
            SOLVER.BASE_LR: 0.0025
            SOLVER.IMS_PER_BATCH: 2
            # less per gpu
            SOLVER.MAX_ITER: 18000
            SOLVER.STEPS: [12000, 16000]
    """)
    cfg.set_deftype("""
    d2:
        SOLVER.CHECKPOINT_PERIOD: [2500, int]
        TEST.EVAL_PERIOD: [0, int]
        SEED: [42, int]
        # ... anything ...
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')
    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = daly_to_datalist(dataset, split)
        datalist_per_split[split] = datalist

    PRETRAINED_WEIGHTS_MODELPATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/horus/pytorch_model_zoo/pascal_voc_baseline/model_final_b1acc2.pkl'

    # // d2_config
    d_cfg = base_d2_frcnn_config()
    d_cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS_MODELPATH
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.DATASETS.TRAIN = (
            'dalyobjects_train',)
    d_cfg.DATASETS.TEST = ('dalyobjects_test',)
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 43
    num_gpus = cf['num_gpus']
    if cf['gpu_scaling.enabled']:
        d2dict_gpu_scaling(cf, cf_add_d2, num_gpus)
    # Merge additional keys
    yacs_add_d2 = yacs.config.CfgNode(
            snippets.unflatten_nested_dict(cf_add_d2), [])
    d_cfg.merge_from_other_cfg(yacs_add_d2)
    d_cfg.freeze()

    if '--port_inc' in add_args:
        ind = add_args.index('--port_inc')
        port_inc = int(add_args[ind+1])
    else:
        port_inc = 0
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14 + port_inc
    dist_url = "tcp://127.0.0.1:{}".format(port)
    args = argparse.Namespace()
    args.datalist_per_split = datalist_per_split
    args.dataset = dataset
    args.resume = True

    launch_w_logging(_train_func_dalyobj,
            num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url=dist_url,
            args=(d_cfg, cf, args))


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
    model_to_eval: [~, str]
    seed: [42, int]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # D2 dataset compatible list of keyframes
    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = daly_to_datalist(dataset, split)
        datalist_per_split[split] = datalist

    for split, datalist in datalist_per_split.items():
        d2_dataset_name = f'dalyobjects_{split}'
        DatasetCatalog.register(d2_dataset_name,
                lambda split=split: datalist_per_split[split])
        MetadataCatalog.get(d2_dataset_name).set(
                thing_classes=dataset.object_names)

    # d2_config
    d_cfg = base_d2_frcnn_config()
    set_d2_cthresh(d_cfg, 0.25)
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

    datalist = datalist_per_split['test']
    metadata = MetadataCatalog.get("dalyobjects_test")
    visfold = small.mkdir(out/'exp3/test')
    vis_evaldemo_dalyobj(predictor, datalist, metadata, visfold, 50)

    datalist = datalist_per_split['train']
    metadata = MetadataCatalog.get("dalyobjects_train")
    visfold = small.mkdir(out/'exp3/train')
    vis_evaldemo_dalyobj(predictor, datalist, metadata, visfold, 50)


class Base_isaver(ABC):
    def __init__(self, folder, total):
        self._re_finished = (
            r'item_(?P<i>\d+)_of_(?P<N>\d+).finished')
        self._fmt_finished = 'item_{:04d}_of_{:04d}.finished'
        self._history_size = 3

        self._folder = folder
        self._total = total

    def _get_filenames(self, i) -> Dict[str, Path]:
        base_filenames = {
            'finished': self._fmt_finished.format(i, self._total)}
        base_filenames['pkl'] = Path(base_filenames['finished']).with_suffix('.pkl')
        filenames = {k: self._folder/v for k, v in base_filenames.items()}
        return filenames

    def _get_intermediate_files(self) -> Dict[int, Dict[str, Path]]:
        """Check re_finished, query existing filenames"""
        intermediate_files = {}
        for ffilename in self._folder.iterdir():
            matched = re.match(self._re_finished, ffilename.name)
            if matched:
                i = int(matched.groupdict()['i'])
                # Check if filenames exist
                filenames = self._get_filenames(i)
                all_exist = all([v.exists() for v in filenames.values()])
                assert ffilename == filenames['finished']
                if all_exist:
                    intermediate_files[i] = filenames
        return intermediate_files

    def _purge_intermediate_files(self):
        """Remove old saved states"""
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        inds_to_purge = np.sort(np.fromiter(
            intermediate_files.keys(), np.int))[:-self._history_size]
        files_purged = 0
        for ind in inds_to_purge:
            filenames = intermediate_files[ind]
            for filename in filenames.values():
                filename.unlink()
                files_purged += 1
        log.debug('Purged {} states, {} files'.format(
            len(inds_to_purge), files_purged))


class Simple_isaver(Base_isaver):
    """
    Will process a list with a func
    """
    def __init__(self, folder, in_list, func,
            save_period='::25'):
        assert sys.version_info >= (3, 6), 'Dicts must keep insertion order'
        super().__init__(folder, len(in_list))
        self.in_list = in_list
        self.result = []
        self.func = func
        self._save_period = save_period

    def _restore(self):
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        start_i, ifiles = max(intermediate_files.items(),
                default=(-1, None))
        if ifiles is not None:
            restore_from = ifiles['pkl']
            self.result = small.load_pkl(restore_from)
            log.info('Restore from {}'.format(restore_from))
        return start_i

    def _save(self, i):
        ifiles = self._get_filenames(i)
        savepath = ifiles['pkl']
        small.save_pkl(savepath, self.result)
        ifiles['finished'].touch()

    def run(self):
        start_i = self._restore()
        run_range = np.arange(start_i+1, self._total)
        for i in tqdm(run_range):
            self.result.append(self.func(self.in_list[i]))
            if snippets.check_step_v2(i, self._save_period) or \
                    (i+1 == self._total):
                self._save(i)
                self._purge_intermediate_files()
        return self.result


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t_ in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t_) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t_])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


from collections import namedtuple


BoxLTRD = namedtuple('BoxLTRD', 'l t r d')
Flat_annotation = namedtuple('Flat_annotation', 'id diff l t r d')
Flat_detection = namedtuple('Flat_detection', 'id id_box score l t r d')


def box_area(box: BoxLTRD) -> float:
    return float((box.d-box.t+1)*(box.r-box.l+1))


def box_intersection(
        bb1: BoxLTRD,
        bb2: BoxLTRD
            ) -> BoxLTRD:
    """ Intersection of two bbs
    Returned bb has same type as first argument"""

    return BoxLTRD(
            l=max(bb1.l, bb2.l),
            t=max(bb1.t, bb2.t),
            r=min(bb1.r, bb2.r),
            d=min(bb1.d, bb2.d))


def t_box_iou(
        bb1: BoxLTRD,
        bb2: BoxLTRD
            ) -> float:

    inter = box_intersection(bb1, bb2)
    if (inter.t >= inter.d) or (inter.l >= inter.r):
        return 0.0  # no intersection
    else:
        intersection_area = box_area(inter)
        union_area = box_area(bb1) + box_area(bb2) - intersection_area
        return intersection_area/union_area


VOClike_object = TypedDict('VOClike_object', {
    'name': str,
    'difficult': bool,
    'box': BoxLTRD
})

VOClike_image_annotation = TypedDict('VOClike_image_annotation', {
    'filepath': Path,
    'frameNumber': int,  # if '-1' we treat filename is image, otherwise as video
    'size_WHD': Tuple[int, int, int],
    'objects': List[VOClike_object]
})


def oldcode_evaluate_voc_detections(
        annotation_list: List[VOClike_image_annotation],
        all_boxes: "List[OrderedDict[str, np.array]]",
        object_classes: List[str],
        iou_thresh,
        use_07_metric,
        use_diff
            ) -> Dict[str, float]:
    """
    Params:
        use_07_metric: If True, will evaluate AP over 11 points, like in VOC2007 evaluation code
        use_diff: If True, will eval difficult proposals in the same way as real ones
    """

    assert len(annotation_list) == len(all_boxes)

    ap_per_cls: Dict[str, float] = {}
    for obj_cls in object_classes:
        # Extract GT annotation_list (id diff l t r d)
        flat_annotations: List[Flat_annotation] = []
        for ind, image_anno in enumerate(annotation_list):
            for obj in image_anno['objects']:
                if obj['name'] == obj_cls:
                    flat_annotations.append(Flat_annotation(
                        ind, obj['difficult'], *obj['box']))
        # Extract detections (id id_box score l t r d)
        flat_detections: List[Flat_detection] = []
        for id, dets_per_cls in enumerate(all_boxes):
            for id_box, det in enumerate(dets_per_cls[obj_cls]):
                flat_detections.append(
                        Flat_detection(id, id_box, det[4], *det[:4]))

        # Prepare
        detection_matched = np.zeros(len(flat_detections), dtype=bool)
        gt_already_matched = np.zeros(len(flat_annotations), dtype=bool)
        gt_assignment = {}  # type: Dict[int, List[int]]
        for i, fa in enumerate(flat_annotations):
            gt_assignment.setdefault(fa.id, []).append(i)
        # Provenance
        detection_matched_to_which_gt = np.ones(
                len(flat_detections), dtype=int)*-1

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
            detection: Flat_detection = flat_detections[detection_ind]

            # Check available GTs
            gt_ids_that_share_image = \
                    gt_assignment.get(detection.id, [])  # type: List[int]
            if not len(gt_ids_that_share_image):
                fp[d] = 1
                continue

            # Compute IOUs
            det_box = BoxLTRD(*(np.array(detection[3:]).round(1)+1))
            gt_boxes: List[BoxLTRD] = [BoxLTRD(*(np.array(
                flat_annotations[gt_ind][2:]).round(1)+1))
                for gt_ind in gt_ids_that_share_image]
            iou_coverages = [t_box_iou(det_box, gtb) for gtb in gt_boxes]

            max_coverage_id = np.argmax(iou_coverages)
            max_coverage = iou_coverages[max_coverage_id]
            gt_id = gt_ids_that_share_image[max_coverage_id]

            # Mirroring voc_eval
            if max_coverage > iou_thresh:
                if (not use_diff) and flat_annotations[gt_id].diff:
                    continue
                if not gt_already_matched[gt_id]:
                    tp[d] = 1
                    detection_matched[detection_ind] = True
                    gt_already_matched[gt_id] = True
                    detection_matched_to_which_gt[detection_ind] = gt_id
                else:
                    fp[d] = 1
            else:
                fp[d] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        ap_per_cls[obj_cls] = ap
        # sklearn version does not work, since it assumes 'recall=1'
        # ap = sklearn.metrics.average_precision_score(detection_matched, detection_scores)
    return ap_per_cls


def eval_d2_dalyobj_old(workfolder, cfg_dict, add_args):
    """
    Here we'll follow the old evaluation protocol
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
    model_to_eval: [~, str]
    subset: ['train', str]
    seed: [42, int]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # D2 dataset compatible list of keyframes
    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = daly_to_datalist(dataset, split)
        datalist_per_split[split] = datalist

    for split, datalist in datalist_per_split.items():
        d2_dataset_name = f'dalyobjects_{split}'
        DatasetCatalog.register(d2_dataset_name,
                lambda split=split: datalist_per_split[split])
        MetadataCatalog.get(d2_dataset_name).set(
                thing_classes=dataset.object_names)

    # d2_config
    d_cfg = base_d2_frcnn_config()
    set_d2_cthresh(d_cfg, 0.25)
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

    # Define subset
    subset = cf['subset']
    if subset == 'train':
        datalist = datalist_per_split['train']
        metadata = MetadataCatalog.get("dalyobjects_train")
    elif subset == 'test':
        datalist = datalist_per_split['test']
        metadata = MetadataCatalog.get("dalyobjects_test")
    else:
        raise RuntimeError('wrong subset')

    # Evaluate all images
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

    df_isaver = Simple_isaver(
            small.mkdir(out/'isaver'), datalist, eval_func, '::50')
    predicted_datalist = df_isaver.run()

    # // Transform to oldshool data format
    # //// GroundTruth
    voclike_annotation_list = []
    for dl_item in datalist:
        objects = []
        for dl_anno in dl_item['annotations']:
            name = dataset.object_names[dl_anno['category_id']]
            box = BoxLTRD(*dl_anno['bbox'])
            difficult = dl_anno['is_occluded']
            o = VOClike_object(name=name, difficult=difficult, box=box)
            objects.append(o)
        filepath = dl_item['video_path']
        frameNumber = dl_item['video_frame_number']
        size_WHD = (dl_item['width'], dl_item['height'], 3)
        voclike_annotation = VOClike_image_annotation(
                filepath=filepath, frameNumber=frameNumber,
                size_WHD=size_WHD, objects=objects)
        voclike_annotation_list.append(voclike_annotation)
    # //// Detections
    all_boxes = []
    for dl_item, pred_item in zip(datalist, predicted_datalist):
        pred_boxes = pred_item.pred_boxes.tensor.numpy()
        scores = pred_item.scores.numpy()
        pred_classes = pred_item.pred_classes.numpy()
        # dets = {k: [] for k in dataset.object_names}
        dets = {}
        for b, s, c_ind in zip(pred_boxes, scores, pred_classes):
            cls = dataset.object_names[c_ind]
            dets.setdefault(cls, []).append(np.r_[b, s])
        dets = {k: np.vstack(v) for k, v in dets.items()}
        dets = {k: dets.get(k, np.array([])) for k in dataset.object_names}
        all_boxes.append(dets)

    use_07_metric = False
    use_diff = False
    iou_thresholds = [0.3, 0.5]
    iou_thresh = 0.5
    object_classes = dataset.object_names
    ap_per_cls: Dict[str, float] = oldcode_evaluate_voc_detections(
            voclike_annotation_list,
            all_boxes,
            object_classes,
            iou_thresh, use_07_metric, use_diff)

    # Results printed nicely via pd.Series
    x = pd.Series(ap_per_cls)*100
    x.loc['AVERAGE'] = x.mean()
    table = snippets.string_table(
            np.array(x.reset_index()),
            header=['Object', 'AP'],
            col_formats=['{}', '{:.2f}'], pad=2)
    log.info(f'AP@{iou_thresh:.3f}:\n{table}')
