import logging
import copy
import numpy as np
import pandas as pd

import torch

from detectron2.data import detection_utils as d2_dutils
from detectron2.data import transforms as d2_transforms
from detectron2.structures import BoxMode

from thes.det2 import (
        get_frame_without_crashing)


log = logging.getLogger(__name__)


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
        gt_df.append([
            *k, min_kframe, max_kframe,
            start_frame, end_frame, v['frame_inds']])
    gt_df = pd.DataFrame(gt_df)
    gt_df.columns = ['vid', 'action', 'ins_id',
            'min_kframe', 'max_kframe',
            'start_frame', 'end_frame', 'frame_inds']
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
