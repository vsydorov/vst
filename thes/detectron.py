"""
Tools for detectron2
"""
import os
import yacs
import functools
import multiprocessing
import threading
import logging
import logging.handlers
import cv2
import time
import numpy as np
import copy
import pandas as pd

from vsydorov_tools import cv as vt_cv

import torch
import torch.multiprocessing as torch_mp
import torch.distributed as dist

from fvcore.common.file_io import PathManager
import detectron2.utils.comm as comm
from detectron2.engine.launch import _find_free_port, _distributed_worker
from detectron2.utils.logger import (
        setup_logger, _ColorfulFormatter, colored, _cached_log_stream)
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.env import seed_all_rng
from detectron2.config import get_cfg
from detectron2.data import detection_utils as d2_dutils
from detectron2.data import transforms as d2_transforms
from detectron2.structures import BoxMode

log = logging.getLogger(__name__)


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


def launch_w_logging(
        main_func, num_gpus_per_machine, num_machines=1,
        machine_rank=0, dist_url=None, args=()):
    """
    Variation of detectron2.engine.launch that logs to a queue

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including
        protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port
                       on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, \
                    "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        mp = multiprocessing.get_context('spawn')
        logmsg_queue = mp.Queue()

        lp = threading.Thread(target=my_logger_thread, args=(logmsg_queue,))
        lp.start()

        try:
            torch_mp.spawn(
                _distributed_worker_w_logging,
                nprocs=num_gpus_per_machine,
                args=(main_func, world_size, num_gpus_per_machine, machine_rank,
                    dist_url, logmsg_queue, args),
                daemon=False,
            )
        finally:
            logmsg_queue.put(None)
            lp.join(1)
            print('Kill logging thread')

    else:
        main_func(*args)


def launch_without_logging(
        main_func, num_gpus_per_machine, num_machines=1,
        machine_rank=0, dist_url=None, args=()):
    """
    Variation of detectron2.engine.launch that logs to a queue

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including
        protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port
                       on localhost
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert num_machines == 1, \
                    "dist_url=auto cannot work with distributed training."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        torch_mp.spawn(
            _distributed_worker_without_logging,
            nprocs=num_gpus_per_machine,
            args=(main_func, world_size, num_gpus_per_machine, machine_rank,
                dist_url, args),
            daemon=False,
        )
    else:
        main_func(*args)


def _distributed_worker_without_logging(
        local_rank, main_func, world_size,
        num_gpus_per_machine, machine_rank,
        dist_url, args):
    """
    Based on "detectron2.engine._distributed_worker'
    """
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


def _distributed_worker_w_logging(
        local_rank, main_func, world_size,
        num_gpus_per_machine, machine_rank,
        dist_url, logmsg_queue, args):
    """
    Based on "detectron2.engine._distributed_worker'

    """
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

    _distributed_queue_logging(global_rank, logmsg_queue)
    main_func(*args)


"""
My multiprocess logging setup
"""


def _distributed_queue_logging(global_rank, logmsg_queue):
    """
    Send logs to queue (only first process)
    """
    print('log_queue: setup loggin')
    root_log = logging.getLogger()
    root_log.setLevel(logging.NOTSET)
    assert len(root_log.handlers) == 0, \
            ('root handlers should be empty '
              'for multiprocessing, instead got {}'.format(root_log.handlers))
    # Add filter to ROOT LOGGER that would record global tank
    filter_record_rank = LoggingFilter_record_rank(global_rank)
    # Add filter qhandler that would only keep rank=0
    # == Add q_handler ==
    q_handler = logging.handlers.QueueHandler(logmsg_queue)
    filter_keep_rank0 = LoggingFilter_keep_rank0()
    q_handler.addFilter(filter_record_rank)
    q_handler.addFilter(filter_keep_rank0)
    root_log.addHandler(q_handler)
    print('root handlers {}'.format(root_log))


@functools.lru_cache()
def setup_logger_d2(
        output=None, distributed_rank=0, *, color=True,
        name="detectron2", abbrev_name=None):
    """
    Stub of "detectron2.utils.logger.setup_logger"
      - Remove stdout part
      - Only file logging

    Args:
        output (str): a file name or a directory to save log. If None, will not
        save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names
        in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True   # We handle the rest
    if abbrev_name is None:
        abbrev_name = "d2" if name == "detectron2" else name
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%m/%d %H:%M:%S"
    )
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


def my_logger_thread(q):
    while True:
        record = q.get()
        if record is None:
            break
        logger = logging.getLogger(record.name)
        logger.handle(record)


class LoggingFilter_record_rank(logging.Filter):
    def __init__(self, rank):
        self.rank = rank

    def filter(self, record):
        mp_name = multiprocessing.current_process().name
        record.rank = self.rank
        record.mp_name = mp_name
        return True


class LoggingFilter_keep_rank0(logging.Filter):
    def filter(self, record):
        if record.rank == 0:
            return True
        else:
            return False


def get_daly_split_vids(dataset, split_label):
    split_vids = [vid for vid, split in dataset.split.items()
            if split == split_label]
    if split_label == 'train':
        split_size = 310
    elif split_label == 'test':
        split_size = 200
    else:
        split_size = None
    assert len(split_vids) == split_size
    return split_vids


def get_daly_gt_tubes(dataset):
    # Daly GT tubes
    gt_tubes = {}
    for vid, v in dataset.video_odict.items():
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                # Read keyframes
                frame_inds = []
                times = []
                boxes = []
                for keyframe in instance['keyframes']:
                    frame_inds.append(keyframe['frameNumber'])
                    times.append(keyframe['time'])
                    boxes.append(keyframe['boundingBox'].squeeze())
                tube = {
                    'start_time': instance['beginTime'],
                    'end_time': instance['endTime'],
                    'frame_inds': frame_inds,
                    'times': times,
                    'boxes': boxes}
                gt_tubes[(vid, action_name, ins_ind)] = tube
    return gt_tubes


def ex_tubes_to_df(extracted_tubes):
    ex_df = []
    for k, v in extracted_tubes.items():
        min_frame = v['frame_inds'].min()
        max_frame = v['frame_inds'].max()
        ex_df.append([*k, min_frame, max_frame])
    ex_df = pd.DataFrame(ex_df)
    ex_df.columns = ['vid', 'bunch_id', 'tube_id', 'min_frame', 'max_frame']
    return ex_df


def gt_tubes_to_df(dataset, gt_tubes):
    gt_df = []
    for k, v in gt_tubes.items():
        vmp4 = dataset.source_videos[k[0]]
        ocv_video_fps = vmp4['frames_reached']/vmp4['length_reached']
        # vmeta = dataset.video_odict[k[0]]
        # meta_video_fps = vmeta['fps']
        # if ocv_video_fps != meta_video_fps:
        #     log.info('FPS mismatch at {}: OCV: {} META: {}'.format(
        #         k, ocv_video_fps, meta_video_fps))
        min_kframe = min(v['frame_inds'])
        max_kframe = max(v['frame_inds'])
        start_frame = int(v['start_time']*ocv_video_fps)
        end_frame = int(v['end_time']*ocv_video_fps)
        frame_inds = v['frame_inds']
        n_frames = len(frame_inds)
        gt_df.append([
            *k, min_kframe, max_kframe,
            start_frame, end_frame, frame_inds, n_frames])
    gt_df = pd.DataFrame(gt_df)
    gt_df.columns = ['vid', 'action', 'ins_id',
            'min_kframe', 'max_kframe',
            'start_frame', 'end_frame',
            'frame_inds', 'n_frames']
    return gt_df


def simplest_daly_to_datalist(dataset, split_label):
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
                            'vid': vid,
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
            dataset_dict["instances"] = d2_dutils.filter_empty_instances(instances)
        return dataset_dict
