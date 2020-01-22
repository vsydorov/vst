import argparse
import os
import yacs
import copy
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, cast

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
from thes.daly_d2 import simplest_daly_to_datalist
from thes.eval_tools import legacy_evaluation


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


def get_daly_odf(dataset):
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
    return odf


def get_category_map_o100(dataset):
    # o100 computations
    odf = get_daly_odf(dataset)
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


def make_datalist_objaction_similar_merged(
        d2_datalist,
        old_object_names, new_object_names,
        action_object_to_object):

    filtered_datalist = []
    for record in d2_datalist:
        filtered_annotations = []
        action_name = record['action_name']
        for obj in record['annotations']:
            old_object_name = old_object_names[obj['category_id']]
            new_object_name = action_object_to_object[
                    (action_name, old_object_name)]
            if new_object_name is None:
                continue
            new_category_id = new_object_names.index(new_object_name)
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
    o100_objects, category_map = get_category_map_o100(dataset)
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


def train_d2_dalyobj_hacky(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_cfg_defaults_d2dalyobj(cfg)
    cfg.set_deftype("""
    hacks:
        dataset: ['normal', ['normal', 'o100', 'action_object']]
        action_object:
            merge: ['sane', ['sane',]]
    """)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    o100_objects, category_map = get_category_map_o100(dataset)
    assert len(o100_objects) == 16

    datalist_per_split = {}
    for split in ['train', 'test']:
        datalist = simplest_daly_to_datalist(dataset, split)
        datalist_per_split[split] = datalist

    if cf['hacks.dataset'] == 'normal':
        num_classes = 43
        object_names = dataset.object_names
    elif cf['hacks.dataset'] == 'o100':
        o100_objects, category_map = get_category_map_o100(dataset)
        num_classes = len(o100_objects)
        assert len(o100_objects) == 16
        object_names = o100_objects
        datalist_per_split = {
            k: make_datalist_o100(datalist, category_map)
            for k, datalist in datalist_per_split.items()}
    elif cf['hacks.dataset'] == 'action_object':
        action_object_to_object = get_similar_action_objects_DALY()
        object_names = sorted([x
            for x in set(list(action_object_to_object.values())) if x])
        num_classes = len(object_names)
        datalist_per_split = {
            k: make_datalist_objaction_similar_merged(
                datalist, dataset.object_names, object_names,
                action_object_to_object)
            for k, datalist in datalist_per_split.items()}
    else:
        raise NotImplementedError()

    d_cfg = _set_d2config(cf, cf_add_d2)
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    d_cfg.freeze()

    num_gpus = cf['num_gpus']
    _d2_train_boring_launch(
            object_names, datalist_per_split,
            num_gpus, d_cfg, cf, add_args)


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
        subset: ['train', str]
    nms:
        enable: [True, bool]
        batched: [False, bool]
        thresh: [0.3, float]
    conf_thresh: [0.0, float]
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

    simple_d2_setup(d_cfg)

    # Visualizer
    predictor = DefaultPredictor(d_cfg)

    # Define subset
    subset = cf['dataset.subset']
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
    legacy_evaluation(dataset.object_names, datalist, predicted_datalist)


def get_similar_action_objects_DALY() -> Dict[Tuple[DALY_action_name, DALY_object_name], str]:
    """ Group similar looking objects, ignore other ones """
    action_object_to_object = \
        {('ApplyingMakeUpOnLips', 'balm'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'brush'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'finger'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'pencil'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'q-tip'): 'ApplyingMakeUpOnLips_stick_like',
         ('ApplyingMakeUpOnLips', 'stick'): 'ApplyingMakeUpOnLips_stick_like',
         ('BrushingTeeth', 'electricToothbrush'): 'BrushingTeeth_toothbrush_like',
         ('BrushingTeeth', 'toothbrush'): 'BrushingTeeth_toothbrush_like',
         ('CleaningFloor', 'broom'): 'CleaningFloor_mop_like',
         ('CleaningFloor', 'brush'): None,
         ('CleaningFloor', 'cloth'): None,
         ('CleaningFloor', 'mop'): 'CleaningFloor_mop_like',
         ('CleaningFloor', 'moppingMachine'): None,
         ('CleaningFloor', 'steamCleaner'): None,
         ('CleaningWindows', 'cloth'): 'CleaningWindows_squeegee_like',
         ('CleaningWindows', 'newspaper'): None,
         ('CleaningWindows', 'scrubber'): 'CleaningWindows_squeegee_like',
         ('CleaningWindows', 'soap'): None,
         ('CleaningWindows', 'sponge'): 'CleaningWindows_squeegee_like',
         ('CleaningWindows', 'squeegee'): 'CleaningWindows_squeegee_like',
         ('Drinking', 'bottle'): None,
         ('Drinking', 'bowl'): None,
         ('Drinking', 'cup'): 'Drinking_glass_like',
         ('Drinking', 'glass'): 'Drinking_glass_like',
         ('Drinking', 'glass+straw'): 'Drinking_glass_like',
         ('Drinking', 'gourd'): None,
         ('Drinking', 'hand'): None,
         ('Drinking', 'hat'): None,
         ('Drinking', 'other'): None,
         ('Drinking', 'plasticBag'): None,
         ('Drinking', 'spoon'): None,
         ('Drinking', 'vase'): None,
         ('FoldingTextile', 'bedsheet'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 'cloth'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 'shirt'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 't-shirt'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 'towel'): 'FoldingTextile_bedsheet_like',
         ('FoldingTextile', 'trousers'): 'FoldingTextile_bedsheet_like',
         ('Ironing', 'iron'): 'Ironing_iron_like',
         ('Phoning', 'mobilePhone'): 'Phoning_phone_like',
         ('Phoning', 'phone'): 'Phoning_phone_like',
         ('Phoning', 'satellitePhone'): 'Phoning_phone_like',
         ('Phoning', 'smartphone'): 'Phoning_phone_like',
         ('PlayingHarmonica', 'harmonica'): 'PlayingHarmonica_harmonica_like',
         ('TakingPhotosOrVideos', 'camera'): 'TakingPhotosOrVideos_camera_like',
         ('TakingPhotosOrVideos', 'smartphone'): 'TakingPhotosOrVideos_camera_like',
         ('TakingPhotosOrVideos', 'videocamera'): 'TakingPhotosOrVideos_camera_like'}
    return cast(Dict[Tuple[DALY_action_name, DALY_object_name], str], action_object_to_object)


def get_biggest_objects_DALY() -> Tuple[DALY_action_name, DALY_object_name]:
    """ Biggest object category per action class """
    primal_configurations = [
            ('ApplyingMakeUpOnLips', 'stick'),
            ('BrushingTeeth', 'toothbrush'),
            ('CleaningFloor', 'mop'),
            ('CleaningWindows', 'squeegee'),
            ('Drinking', 'glass'),
            ('FoldingTextile', 'bedsheet'),
            ('Ironing', 'iron'),
            ('Phoning', 'phone'),
            ('PlayingHarmonica', 'harmonica'),
            ('TakingPhotosOrVideos', 'camera')]
    return cast(Tuple[DALY_action_name, DALY_object_name], primal_configurations)


def eval_d2_dalyobj_hacky(workfolder, cfg_dict, add_args):
    """
    Evaluation code with hacks
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    what_to_eval: [~, str]
    hacks:
        model_to_eval: ['what', ['what', 'what+foldname']]
        dataset: ['normal', ['normal', 'o100', 'action_object']]
        action_object:
            merge: ['sane', ['sane',]]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    nms:
        enable: [True, bool]
        batched: [False, bool]
        thresh: [0.3, float]
    conf_thresh: [0.0, float]
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

    if cf['hacks.model_to_eval'] == 'what':
        model_to_eval = cf['model_to_eval']
    elif cf['hacks.model_to_eval'] == 'what+foldname':
        import dervo
        EPATH = dervo.experiment.EXPERIMENT_PATH
        model_name = EPATH.resolve().name
        model_to_eval = str(Path(cf['what_to_eval'])/model_name)
    else:
        raise NotImplementedError('Wrong hacks.model_to_eval')

    # Define subset
    subset = cf['dataset.subset']
    if subset == 'train':
        datalist = datalist_per_split['train']
    elif subset == 'test':
        datalist = datalist_per_split['test']
    else:
        raise RuntimeError('wrong subset')

    if cf['hacks.dataset'] == 'normal':
        num_classes = 43
        object_names = dataset.object_names
    elif cf['hacks.dataset'] == 'o100':
        o100_objects, category_map = get_category_map_o100(dataset)
        num_classes = len(o100_objects)
        assert len(o100_objects) == 16
        datalist = make_datalist_o100(datalist, category_map)
        object_names = o100_objects
    elif cf['hacks.dataset'] == 'action_object':
        odf = get_daly_odf(dataset)
        pass
    else:
        raise NotImplementedError()

    # d2_config
    d_cfg = base_d2_frcnn_config()
    set_d2_cthresh(d_cfg, cf['conf_thresh'])
    d_cfg.OUTPUT_DIR = str(small.mkdir(out/'d2_output'))
    d_cfg.DATASETS.TRAIN = ()
    d_cfg.DATASETS.TEST = ('dalyobjects_test',)
    d_cfg.SEED = cf['seed']
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    d_cfg.MODEL.WEIGHTS = model_to_eval
    d_cfg.freeze()

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
    legacy_evaluation(object_names, datalist, predicted_datalist)
