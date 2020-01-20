import argparse
import os
import yacs
import copy
import logging
import numpy as np
import pandas as pd

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

from thes.data.external_dataset import DatasetDALY
from thes.tools import snippets
from thes.det2 import (
        YAML_Base_RCNN_C4, YAML_faster_rcnn_R_50_C4,
        launch_w_logging, launch_without_logging, simple_d2_setup,
        base_d2_frcnn_config, set_d2_cthresh, d2dict_gpu_scaling,
        D2DICT_GPU_SCALING_DEFAULTS,
        get_frame_without_crashing)
from thes.daly_d2 import simplest_daly_to_datalist


log = logging.getLogger(__name__)


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


class DalyObjTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DalyVideoDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg,
                mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
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
    object_names = args.object_names

    for split, datalist in datalist_per_split.items():
        d2_dataset_name = f'dalyobjects_{split}'
        DatasetCatalog.register(d2_dataset_name,
                lambda split=split: datalist_per_split[split])
        MetadataCatalog.get(d2_dataset_name).set(
                thing_classes=object_names)

    trainer = DalyObjTrainer(d_cfg)
    trainer.resume_or_load(resume=args.resume)
    if d_cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(
                0, lambda: trainer.test_with_TTA(d_cfg, trainer.model))]
        )
    trainer.train()


def _set_cfg_defaults_d2dalyobj(cfg):
    cfg.set_defaults_handling(['d2.'])
    cfg.set_deftype("""
    num_gpus: [~, int]
    """)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
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


def _set_d2config(cf, cf_add_d2):
    # // d2_config
    d_cfg = base_d2_frcnn_config()
    d_cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS_MODELPATH
    d_cfg.DATASETS.TRAIN = (
            'dalyobjects_train',)
    d_cfg.DATASETS.TEST = ('dalyobjects_test',)
    if cf['gpu_scaling.enabled']:
        d2dict_gpu_scaling(cf, cf_add_d2, cf['num_gpus'])
    # Merge additional keys
    yacs_add_d2 = yacs.config.CfgNode(
            snippets.unflatten_nested_dict(cf_add_d2), [])
    d_cfg.merge_from_other_cfg(yacs_add_d2)
    return d_cfg


def _d2_train_boring_launch(
        object_names, datalist_per_split, num_gpus, d_cfg, cf, add_args):
    if '--port_inc' in add_args:
        ind = add_args.index('--port_inc')
        port_inc = int(add_args[ind+1])
    else:
        port_inc = 0
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14 + port_inc
    dist_url = "tcp://127.0.0.1:{}".format(port)
    args = argparse.Namespace()
    args.datalist_per_split = datalist_per_split
    args.object_names = object_names
    args.resume = True

    launch_w_logging(_train_func_dalyobj,
            num_gpus,
            num_machines=1,
            machine_rank=0,
            dist_url=dist_url,
            args=(d_cfg, cf, args))


def train_d2_dalyobj(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_cfg_defaults_d2dalyobj(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    object_names = dataset.object_names

    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = simplest_daly_to_datalist(dataset, split)
        datalist_per_split[split] = datalist

    d_cfg = _set_d2config(cf, cf_add_d2)
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 43
    d_cfg.freeze()

    num_gpus = cf['num_gpus']
    _d2_train_boring_launch(
            object_names, datalist_per_split,
            num_gpus, d_cfg, cf, add_args)


def get_category_map(dataset):
    # o100 computations
    gt_objects = []
    for vid, v in dataset.video_odict.items():
        vmp4 = dataset.source_videos[vid]
        height = vmp4['height']
        width = vmp4['width']
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                for keyframe in instance['keyframes']:
                    kf_objects = keyframe['objects']
                    frame_number = keyframe['frameNumber']
                    for kfo in kf_objects:
                        [xmin, ymin, xmax, ymax,
                            objectID, isOccluded, isHallucinate] = kfo
                        box_unscaled = np.array([xmin, ymin, xmax, ymax])
                        bbox = box_unscaled * np.tile([width, height], 2)
                        objectID = int(objectID)
                        object_name = dataset.object_names[objectID]
                        obj = {
                                'vid': vid,
                                'ins_ind': ins_ind,
                                'frame_number': frame_number,
                                'action_name': action_name,
                                'bbox': bbox,
                                'object_name': object_name,
                                'is_occluded': isOccluded,
                                'is_hallucinate': isHallucinate}
                        gt_objects.append(obj)
    odf = pd.DataFrame(gt_objects)
    ocounts = odf.object_name.value_counts()
    o100_objects = sorted(ocounts[ocounts>100].index)
    category_map = []
    for obj_name in dataset.object_names:
        if obj_name in o100_objects:
            mapped = o100_objects.index(obj_name)
        else:
            mapped = None
        category_map.append(mapped)
    return o100_objects, category_map


def make_datalist_o100(d2_datalist, category_map):
    filtered_datalist = []
    for record in d2_datalist:
        filtered_annotations = []
        for obj in record['annotations']:
            new_category_id = category_map[obj['category_id']]
            if new_category_id is not None:
                new_obj = copy.copy(obj)
                new_obj['category_id'] = new_category_id
                filtered_annotations.append(new_obj)
        if len(filtered_annotations) != 0:
            new_record = copy.copy(record)
            new_record['annotations'] = filtered_annotations
            filtered_datalist.append(new_record)
    return filtered_datalist


def train_d2_dalyobj_o100(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_cfg_defaults_d2dalyobj(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    o100_objects, category_map = get_category_map(dataset)
    assert len(o100_objects) == 16

    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = simplest_daly_to_datalist(dataset, split)
        datalist_o100 = make_datalist_o100(datalist, category_map)
        datalist_per_split[split] = datalist_o100

    d_cfg = _set_d2config(cf, cf_add_d2)
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
    d_cfg.freeze()

    num_gpus = cf['num_gpus']
    _d2_train_boring_launch(
            o100_objects, datalist_per_split,
            num_gpus, d_cfg, cf, add_args)
