"""
Basic classes that allow working with external datasets
"""
import copy
import xml.etree.ElementTree as ET
import subprocess
import csv
import hashlib
import re
import pandas as pd
import numpy as np
import cv2
import logging
import concurrent.futures
from abc import abstractmethod, ABC
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Tuple, cast, NewType, Any, TypedDict

from vsydorov_tools import small, cv as vt_cv

from thes.filesystem import get_dataset_path
from thes.tools import snippets


log = logging.getLogger(__name__)

Vid = NewType('Vid', str)  # Unique video ID
Dataset_subset = NewType('Dataset_subset', str)  # train/val/test
F0 = NewType('F0', int)  # 0-based frameNumber


Vid_daly = NewType('Vid_daly', Vid)
Action_name_daly = NewType('Action_name_daly', str)
Object_name_daly = NewType('Object_name_daly', str)
Joint_name_daly = NewType('Joint_name_daly', str)


class OCV_rstats(TypedDict):
    # OCV reachability stats
    height: int
    width: int
    frame_count: int
    fps: float
    max_pos_frames: int  # 1-based
    max_pos_msec: float


def compute_ocv_rstats(video_path, n_tries=5) -> OCV_rstats:
    with vt_cv.video_capture_open(video_path, n_tries) as vcap:
        height, width = vt_cv.video_getHW(vcap)
        frame_count = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vcap.get(cv2.CAP_PROP_FPS)
        while True:
            max_pos_frames = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
            max_pos_msec = vcap.get(cv2.CAP_PROP_POS_MSEC)
            ret = vcap.grab()
            if ret is False:
                break
    ocv_rstats: OCV_rstats = {
        'height': height,
        'width': width,
        'frame_count': frame_count,
        'fps': fps,
        'max_pos_frames': max_pos_frames,
        'max_pos_msec': max_pos_msec,
        }
    return ocv_rstats


class ProvidedMetadata_daly(TypedDict):
    duration: float
    nbframes_ffmpeg: int
    fps: float


class Instance_flags_daly(TypedDict):
    isSmall: bool
    isReflection: bool
    isShotcut: bool
    isZoom: bool
    isAmbiguous: bool
    isOccluded: bool
    isOutsideFOV: bool


class Keyframe_daly(TypedDict):
    # shape (1, 4), LTRD[xmin, ymin, xmax, ymax], relative (0..1)
    boundingBox: np.ndarray
    # [xmin, ymin, xmax, ymax, objectID, isOccluded, isHallucinate]
    objects: np.ndarray
    frameNumber: int  # 1-based
    pose: np.ndarray
    time: float  # seconds


class Instance_daly(TypedDict):
    beginTime: float
    endTime: float
    flags: Instance_flags_daly
    keyframes: List[Keyframe_daly]


class Video_daly(TypedDict):
    vid: Vid_daly
    path: Path
    suggestedClass: Action_name_daly
    instances: Dict[Action_name_daly, List[Instance_daly]]


class Dataset_daly(object):
    root_path: Path
    action_names = cast(List[Action_name_daly], [
        'ApplyingMakeUpOnLips', 'BrushingTeeth', 'CleaningFloor',
        'CleaningWindows', 'Drinking', 'FoldingTextile', 'Ironing',
        'Phoning', 'PlayingHarmonica', 'TakingPhotosOrVideos'])
    object_names = cast(List[Object_name_daly], [
        'balm', 'bedsheet', 'bottle', 'bowl', 'broom', 'brush', 'camera',
        'cloth', 'cup', 'electricToothbrush', 'finger', 'glass', 'glass+straw',
        'gourd', 'hand', 'harmonica', 'hat', 'iron', 'mobilePhone', 'mop',
        'moppingMachine', 'newspaper', 'other', 'pencil', 'phone',
        'plasticBag', 'q-tip', 'satellitePhone', 'scrubber', 'shirt',
        'smartphone', 'soap', 'sponge', 'spoon', 'squeegee', 'steamCleaner',
        'stick', 't-shirt', 'toothbrush', 'towel', 'trousers', 'vase',
        'videocamera'])
    joint_names = cast(List[Joint_name_daly], [
        'head', 'shoulderLeft', 'elbowLeft', 'wristLeft', 'shoulderRight',
        'elbowRight', 'wristRight'])
    split: Dict[Vid_daly, Dataset_subset]
    provided_metas: Dict[Vid_daly, ProvidedMetadata_daly]
    videos: Dict[Vid_daly, Video_daly]

    def __init__(self):
        super().__init__()
        self.root_path = get_dataset_path('action/DALY')
        self._load_pkl()

    def _load_pkl(self):
        pkl_path = self.root_path/'daly1.1.0.pkl'
        info = small.load_py2_pkl(pkl_path)
        assert self.action_names == info['labels']
        assert self.object_names == info['objectList']
        assert self.joint_names == info['joints']
        videos: Dict[Vid_daly, Video_daly] = {}
        provided_metas: Dict[Vid_daly, ProvidedMetadata_daly] = {}
        for video_name, v in info['annot'].items():
            vid = video_name.split('.')[0]
            video_meta = info['metadata'][video_name]
            video: Video_daly = {
                'vid': vid,
                'path': self.root_path/f'videos/{vid}.mp4',
                'suggestedClass': v['suggestedClass'],
                'instances': v['annot']}
            meta: ProvidedMetadata_daly = {
                'duration': video_meta['duration'],
                'nbframes_ffmpeg': video_meta['nbframes_ffmpeg'],
                'fps': video_meta['fps']}
            videos[vid] = video
            provided_metas[vid] = meta
        split = {k: 'train' for k in videos.keys()}
        for video_name in info['splits'][0]:
            vid = video_name.split('.')[0]
            split[vid] = 'test'
        self.videos = videos
        self.provided_metas = provided_metas
        self.split = split


class Keyframe_daly_ocv(Keyframe_daly):
    bbox_abs: np.ndarray  # shape (4), LTRD, pixels
    objects_abs: np.ndarray  # pixels
    pose_abs: np.ndarray  # pixels
    frame: F0


class Instance_daly_ocv(TypedDict):
    beginTime: float
    endTime: float
    flags: Instance_flags_daly
    keyframes: List[Keyframe_daly_ocv]
    # ocv
    start_frame: F0
    end_frame: F0


class Video_daly_ocv(TypedDict):
    vid: Vid_daly
    path: Path
    suggestedClass: Action_name_daly
    instances: Dict[Action_name_daly, List[Instance_daly_ocv]]
    # ocv
    height: int
    width: int


class Dataset_daly_ocv(Dataset_daly):
    """
    We access the videos with opencv here
    """
    rstats: Dict[Vid_daly, OCV_rstats]
    videos_ocv: Dict[Vid_daly, Video_daly_ocv]

    def __init__(self):
        super().__init__()

    def _compute_videos_ocv(self, rstats):
        videos_ocv = {}
        for vid, video in self.videos.items():
            rs = rstats[vid]
            est_fps = (rs['max_pos_frames']-1)*1000/rs['max_pos_msec']
            wh = np.array([rs['width'], rs['height']])
            whwh = np.tile(wh, 2)
            instances_ocv = {}
            for action_name, ains in video['instances'].items():
                ains_ocv = []
                for instance in ains:
                    keyframes = []
                    for kf in instance['keyframes']:
                        bbox_abs = kf['boundingBox'][0] * whwh
                        objects_abs = kf['objects']
                        objects_abs[:, :4] *= whwh
                        pose_abs = kf['pose']
                        pose_abs[:, :4] *= whwh
                        frame: F0 = np.ceil(kf['time'] * est_fps) - 1
                        kf_ocv = copy.deepcopy(kf)
                        kf_ocv.update({
                            'bbox_abs': bbox_abs,
                            'objects_abs': objects_abs,
                            'pose_abs': pose_abs,
                            'frame': frame})
                        kf_ocv = cast(Keyframe_daly_ocv, kf_ocv)
                        keyframes.append(kf_ocv)
                    instance_ocv = copy.deepcopy(instance)
                    # Edges are controlled for corner cases
                    start_frame = max(0, np.ceil(instance['beginTime'] * est_fps) - 1)
                    end_frame = np.ceil(instance['endTime'] * est_fps)
                    if end_frame > rs['max_pos_frames']:
                        end_frame = rs['max_pos_frames']
                    end_frame -= 1
                    instance_ocv.update({
                        'keyframes': keyframes,
                        'start_frame': start_frame,
                        'end_frame': end_frame})
                    instance_ocv = cast(Instance_daly_ocv, instance_ocv)
                    ains_ocv.append(instance_ocv)
                instances_ocv[action_name] = ains_ocv
            video_ocv = copy.deepcopy(video)
            video_ocv.update({
                'instances': instances_ocv,
                'height': rs['height'],
                'width': rs['width']})
            videos_ocv[vid] = video_ocv
        return videos_ocv

    def precompute_to_folder(self, fold):
        fold = Path(fold)
        vids = list(self.videos.keys())
        isaver = snippets.Threading_isaver(
            small.mkdir(fold/'isave_rstats'), vids,
            lambda vid: compute_ocv_rstats(
                self.root_path/f'videos/{vid}.mp4'), 4, 8)
        isaver_items = isaver.run()
        rstats: Dict[Vid_daly, OCV_rstats] = dict(zip(vids, isaver_items))
        videos_ocv = self._compute_videos_ocv(rstats)
        precomputed_stats = {
            'rstats': rstats,
            'videos_ocv': videos_ocv,
        }
        small.save_pkl(fold/'precomputed_stats.pkl', precomputed_stats)

    def populate_from_folder(self, fold):
        fold = Path(fold)
        precomputed_stats = small.load_pkl(fold/'precomputed_stats.pkl')
        self.rstats = precomputed_stats['rstats']
        self.videos_ocv = precomputed_stats['videos_ocv']
