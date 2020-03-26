import os
import yacs
import multiprocessing
import threading
import logging
import logging.handlers
import cv2
import time
import numpy as np
import copy
import pandas as pd
from pathlib import Path

from vsydorov_tools import cv as vt_cv

import torch

from fvcore.common.file_io import PathManager  # type: ignore
import detectron2.utils.comm as comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.config import get_cfg
from detectron2.data import detection_utils as d2_dutils
from detectron2.data import transforms as d2_transforms
from detectron2.structures import BoxMode
from detectron2.engine import (DefaultTrainer)
from detectron2.data import (
        build_detection_train_loader,
        build_detection_test_loader)

from thes.detectron.internals import setup_logger_d2

log = logging.getLogger(__name__)


def get_frame_without_crashing(
        video_path: Path,
        frame_number: int,
        frame_time: float,
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


def simple_d2_setup(d_cfg):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory
    """
    output_dir = d_cfg.OUTPUT_DIR
    seed = d_cfg.SEED

    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)

    rank = comm.get_rank()
    setup_logger_d2(
            output_dir, distributed_rank=rank, name="fvcore")
    d2_log = setup_logger_d2(output_dir, distributed_rank=rank)

    # Print some stuff
    d2_log.info("Rank of current process: {}. World size: {}".format(
        rank, comm.get_world_size()))
    d2_log.info("Environment info:\n" + collect_env_info())
    d2_log.info("Running with full config:\n{}".format(d_cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(d_cfg.dump())
        d2_log.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if seed < 0 else seed + rank)


class DalyVideoDatasetMapper:
    """
    Taken from detectron2.data.dataset_mapper.DatasetMapper
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

        # Pytorch's dataloader is efficient on torch.Tensor due to
        # shared-memory, but not efficient on large generic data structures due
        # to the use of pickle & mp.Queue.  Therefore it's important to use
        # torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        ).contiguous()
        # Can use uint8 if it turns out to be slow some day

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Implement additional transformations if you have other
            # types of data
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
            dataset_dict["instances"] = \
                    d2_dutils.filter_empty_instances(instances)
        return dataset_dict


class DalyVideoDatasetTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DalyVideoDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DalyVideoDatasetMapper(cfg, is_train=False)
        return build_detection_test_loader(
                cfg, dataset_name, mapper=mapper)
