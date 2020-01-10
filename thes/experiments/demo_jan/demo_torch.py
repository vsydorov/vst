import logging
import logging.handlers
import threading
import argparse
import time
import yacs
from tqdm import tqdm
from abc import abstractmethod, ABC
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
from pathlib import Path
from typing import Dict, List, Tuple
from mypy_extensions import TypedDict

from detectron2.config import get_cfg

from thes.tools import snippets
from thes.data.external_dataset import DatasetVOC2007

from vsydorov_tools import small

log = logging.getLogger(__name__)


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data.detection_utils import read_image

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
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
    from detectron2.data.datasets.pascal_voc import register_pascal_voc
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


import sys
import functools
from fvcore.common.file_io import PathManager
from detectron2.utils.logger import (
        setup_logger, _ColorfulFormatter, colored, _cached_log_stream)
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng


@functools.lru_cache()
def setup_logger_d2mine(
    output=None, distributed_rank=0, *, color=True, name="detectron2",
    abbrev_name=None
):
    """
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = "d2" if name == "detectron2" else name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManager.mkdirs(os.path.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


def default_setup_mine(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger_d2mine(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger_d2mine(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(
        rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file,
                PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead.
    # It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK


def logger_thread(q):
    while True:
        record = q.get()
        if record is not None:
            print('Got record {}, name {}:'.format(record, record.name))
        if record is None:
            print('Got NONE record')
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


def _train_func(d_cfg, cf, args, logmsg_queue):
    # default_setup_mine(d_cfg, args)
    #
    # d2_log = logging.getLogger("detectron2")
    # d2_log.info("d2 test")
    #
    print('before_handlers')
    print('__name__: {}'.format(__name__))

    root_log = logging.getLogger()
    print('ROOT handlers: {}'.format(root_log.handlers))

    q_handler = logging.handlers.QueueHandler(logmsg_queue)
    q_log = logging.getLogger(__name__)
    q_log.addHandler(q_handler)
    q_log.setLevel(logging.DEBUG)
    # q_log.propagate = False
    print('q_log handlers: {}'.format(q_log.handlers))
    #
    name = multiprocessing.current_process().name

    q_log.info("q_log test, name {}".format(name))
    log.info("log test, name {}".format(name))

    print('after everything')
    # log.info("my test")

    #     ("voc_2007_trainval_absolute", "VOC2007/VOCdevkit/VOC2007", "trainval"),
    #     ("voc_2007_test_absolute", "VOC2007/VOCdevkit/VOC2007", "test"),
    #     ("voc_2012_trainval_absolute", "VOC2012/VOCdevkit/VOC2012", "trainval"),
    # ]
    # for name, dirname, split in SPLITS:
    #     year = 2007 if "2007" in name else 2012
    #     register_pascal_voc(name, os.path.join(VOC_ROOT, dirname), split, year)
    #     MetadataCatalog.get(name).evaluator_type = "pascal_voc"
    #
    # trainer = Trainer(d_cfg)
    # trainer.resume_or_load(resume=cf['resume'])
    # if d_cfg.TEST.AUG.ENABLED:
    #     trainer.register_hooks(
    #         [hooks.EvalHook(
    #             0, lambda: trainer.test_with_TTA(d_cfg, trainer.model))]
    #     )
    # trainer.train()


from detectron2.engine.launch import _find_free_port, _distributed_worker
import torch.multiprocessing as torch_mp
import torch.distributed as dist


def _my_distributed_worker(
        local_rank, main_func, world_size,
        num_gpus_per_machine, machine_rank,
        dist_url, args):
    assert torch.cuda.is_available(), \
            "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL", init_method=dist_url,
            world_size=world_size, rank=global_rank
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after calling
    # init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # Setup the local process group (which contains ranks within the same
    # machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine,
            (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    main_func(*args)


def my_launch(main_func, num_gpus_per_machine, num_machines=1, machine_rank=0,
        dist_url=None, args=()):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        torch_mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(main_func, world_size, num_gpus_per_machine, machine_rank,
                dist_url, args),
            daemon=False,
        )
    else:
        main_func(*args)


def train_d2_rcnn_voc0712(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    num_gpus: [~, int]
    resume: [True, bool]
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
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'output_dir'))
    # d_cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS
    # 2GPUs only
    d_cfg.SOLVER.BASE_LR = 0.0025 * cf['num_gpus']
    d_cfg.SOLVER.IMS_PER_BATCH = 2 * cf['num_gpus']
    # Change datasets
    d_cfg.DATASETS.TRAIN = (
            'voc_2007_trainval_absolute',
            'voc_2012_trainval_absolute')
    d_cfg.DATASETS.TEST = ('voc_2007_test_absolute',)
    d_cfg.freeze()

    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    dist_url = "tcp://127.0.0.1:{}".format(port)

    args = argparse.Namespace()

    # if cf['num_gpus'] > 1:
    #     launch(_train_func,
    #             cf['num_gpus'],
    #             num_machines=1,
    #             machine_rank=0,
    #             dist_url=dist_url,
    #             args=(d_cfg, cf, args))
    # else:
    #     _train_func(d_cfg, cf, args)

    mp = multiprocessing.get_context('spawn')
    logmsg_queue = mp.Queue()
    lp = threading.Thread(target=logger_thread, args=(logmsg_queue,))
    lp.start()
    my_launch(_train_func,
            cf['num_gpus'],
            num_machines=1,
            machine_rank=0,
            dist_url=dist_url,
            args=(d_cfg, cf, args, logmsg_queue))
    # _train_func(d_cfg, cf, args, logmsg_queue)
    logmsg_queue.put(None)
    lp.join()

    root_log = logging.getLogger()
    print('ROOT handlers: {}'.format(root_log.handlers))

    # _train_func(d_cfg, cf, args)
