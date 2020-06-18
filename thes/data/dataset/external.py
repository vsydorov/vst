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
from typing import (Dict, List, Tuple, cast, NewType,
        Any, TypedDict, Optional, Literal)

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
        'balm', 'bedsheet', 'bottle', 'bowl', 'broom', 'brush',
        'camera', 'cloth', 'cup', 'electricToothbrush', 'finger',
        'glass', 'glass+straw', 'gourd', 'hand', 'harmonica', 'hat',
        'iron', 'mobilePhone', 'mop', 'moppingMachine', 'newspaper',
        'other', 'pencil', 'phone', 'plasticBag', 'q-tip',
        'satellitePhone', 'scrubber', 'shirt', 'smartphone',
        'soap', 'sponge', 'spoon', 'squeegee', 'steamCleaner',
        'stick', 't-shirt', 'toothbrush', 'towel', 'trousers',
        'vase', 'videocamera'])
    joint_names = cast(List[Joint_name_daly], [
        'head', 'shoulderLeft', 'elbowLeft', 'wristLeft',
        'shoulderRight', 'elbowRight', 'wristRight'])
    split: Dict[Vid_daly, Dataset_subset]
    provided_metas: Dict[Vid_daly, ProvidedMetadata_daly]
    videos: Dict[Vid_daly, Video_daly]

    def __init__(self, mirror):
        super().__init__()
        self.root_path = get_dataset_path('action/daly_take2')
        self.mirror = mirror
        self._load_pkl()

    def _load_pkl(self):
        pkl_path = self.root_path/'annotations/daly1.1.0.pkl'
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
                'path': (self.root_path/
                    f'mirrors/{self.mirror}/{vid}.mp4'),
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
    nframes: int
    est_length: float
    est_fps: float


def get_daly_split_vids(
        dataset: Dataset_daly,
        split_label: Literal['train', 'test']
        ) -> List[Vid_daly]:
    split_vids = [
        vid for vid, split in dataset.split.items()
        if split == split_label]
    if split_label == 'train':
        split_size = 310
    elif split_label == 'test':
        split_size = 200
    else:
        raise RuntimeError()
    assert len(split_vids) == split_size
    return split_vids


def split_off_validation_set(dataset, fraction, n_samplings=20, seed=42):
    # // Split a validation set off (roughly equalize number of keyframes)
    # Keyframe number per vid
    keyframes_va = {}
    for vid, video in dataset.videos.items():
        counter = np.zeros(len(dataset.action_names))
        for action_name, ains in video['instances'].items():
            aid = dataset.action_names.index(action_name)
            for ins_ind, instance in enumerate(ains):
                for kf_ind, kf in enumerate(instance['keyframes']):
                    counter[aid] += 1
        keyframes_va[vid] = counter
    kdf = pd.DataFrame(keyframes_va).T
    trainval_vids = get_daly_split_vids(dataset, 'train')
    kdf_train = kdf.loc[trainval_vids]
    kf_sum = kdf_train.sum()

    random_state = np.random.RandomState(seed)

    def get_perms_and_fractions():
        perms = random_state.permutation(trainval_vids)
        selected, rest = np.split(perms, [int(len(perms) * fraction)])
        kf_val_sum = kdf.loc[selected].sum()
        frac = kf_val_sum/kf_sum
        frac_min = frac.min()
        frac_max = frac.max()
        return (selected, rest), frac_min, frac_max

    samples = [get_perms_and_fractions() for i in range(n_samplings)]
    # frac_min must be above 0
    samples = [x for x in samples if x[1] > 0]
    assert len(samples) > 0, '>= 1 action missing'
    # frac_max must be below 1
    samples = [x for x in samples if x[2] < 1]
    assert len(samples) > 0, '>=1 action fully sampled'
    # time_max must be closest to fraction
    fraction_diff = [abs(fraction-x[1]) for x in samples]
    closest_id = np.argmin(fraction_diff)
    val_vids, train_vids = samples[closest_id][0]
    return val_vids, train_vids


class Dataset_daly_ocv(Dataset_daly):
    """
    We access the videos with opencv here
    """
    rstats: Dict[Vid_daly, OCV_rstats]
    videos_ocv: Dict[Vid_daly, Video_daly_ocv]

    def __init__(self, mirror):
        super().__init__(mirror)

    def _compute_videos_ocv(self, rstats):
        videos_ocv = {}
        for vid, video in self.videos.items():
            rs = rstats[vid]
            nframes = rs['max_pos_frames']
            est_fps = (nframes-1)*1000/rs['max_pos_msec']
            est_length = np.round(nframes / est_fps, 4)
            wh = np.array([rs['width'], rs['height']])
            whwh = np.tile(wh, 2)
            instances_ocv = {}
            for action_name, ains in video['instances'].items():
                ains_ocv = []
                for instance in ains:
                    keyframes = []
                    for kf in instance['keyframes']:
                        bbox_abs = kf['boundingBox'][0] * whwh
                        objects_abs = kf['objects'].copy()
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
                        'start_frame': int(start_frame),
                        'end_frame': int(end_frame)})
                    instance_ocv = cast(Instance_daly_ocv, instance_ocv)
                    ains_ocv.append(instance_ocv)
                instances_ocv[action_name] = ains_ocv
            video_ocv = copy.deepcopy(video)
            video_ocv.update({
                'instances': instances_ocv,
                'height': rs['height'],
                'width': rs['width'],
                'nframes': nframes,
                'est_length': est_length,
                'est_fps': est_fps
                })
            videos_ocv[vid] = video_ocv
        return videos_ocv

    def precompute_to_folder(self, fold):
        fold = Path(fold)
        vids = list(self.videos.keys())
        vids_to_path = {k: v['path'] for k, v in self.videos.items()}
        isaver = snippets.Isaver_threading(
            small.mkdir(fold/'isave_rstats'), vids,
            lambda vid: compute_ocv_rstats(vids_to_path[vid]), 4, 8)
        isaver_items = isaver.run()
        rstats: Dict[Vid_daly, OCV_rstats] = dict(zip(vids, isaver_items))
        videos_ocv = self._compute_videos_ocv(rstats)
        precomputed_stats = {
            'rstats': rstats,
            'videos_ocv': videos_ocv,
            'mirror': self.mirror,
        }
        small.save_pkl(fold/'precomputed_stats.pkl', precomputed_stats)

    def populate_from_folder(self, fold):
        fold = Path(fold)
        precomputed_stats = small.load_pkl(fold/'precomputed_stats.pkl')
        self.rstats = precomputed_stats['rstats']
        self.videos_ocv = precomputed_stats['videos_ocv']
        # If mirror in precomputed stats was different - brute replace
        if self.mirror != precomputed_stats['mirror']:
            log.info(('Mirror mismatch in precomputed stats. '
                'We replace {} -> {}').format(
                    precomputed_stats['mirror'], self.mirror))
            for k, v in self.videos_ocv.items():
                v['path'] = self.videos[k]['path']


Vid_char = NewType('Vid_char', Vid)
Action_name_char = NewType('Action_name_char', str)
Object_name_char = NewType('Object_name_char', str)
Verb_name_char = NewType('Verb_name_char', str)


class Action_instance_char(TypedDict):
    name: Action_name_char
    start_time: float
    end_time: float


class Video_char(TypedDict):
    vid: Vid_char
    subject: str
    scene: str
    quality: Optional[int]
    relevance: Optional[int]
    verified: str
    script: str
    objects: List[Object_name_char]
    descriptions: str
    actions: List[Action_instance_char]
    length: float


class Dataset_charades(object):
    """
    Charades dataset, featuring trainval split that was released
    """
    root_path: Path

    action_names: List[Action_name_char]
    object_names = cast(List[Object_name_char], [
        'None', 'bag', 'bed', 'blanket', 'book', 'box', 'broom',
        'chair', 'closet/cabinet', 'clothes', 'cup/glass/bottle',
        'dish', 'door', 'doorknob', 'doorway', 'floor', 'food',
        'groceries', 'hair', 'hands', 'laptop', 'light', 'medicine',
        'mirror', 'paper/notebook', 'phone/camera', 'picture', 'pillow',
        'refrigerator', 'sandwich', 'shelf', 'shoe', 'sofa/couch',
        'table', 'television', 'towel', 'vacuum', 'window'])
    verb_names = cast(List[Verb_name_char], [
        'awaken', 'close', 'cook', 'dress', 'drink', 'eat', 'fix',
        'grasp', 'hold', 'laugh', 'lie', 'make', 'open', 'photograph',
        'play', 'pour', 'put', 'run', 'sit', 'smile', 'sneeze',
        'snuggle', 'stand', 'take', 'talk', 'throw', 'tidy', 'turn',
        'undress', 'walk', 'wash', 'watch', 'work'])
    action_mapping: Dict[Action_name_char, Tuple[
        Object_name_char, Verb_name_char]]
    split: Dict[Vid_char, Dataset_subset]
    videos: Dict[Vid_char, Video_char]

    # Utility
    _action_labels: Dict[str, Action_name_char]
    _object_labels: Dict[str, Object_name_char]
    _verb_labels: Dict[str, Verb_name_char]

    def __init__(self):
        super().__init__()
        self.root_path = get_dataset_path('action/charades_take2')
        self._anno_fold = self.root_path/'annotation/Charades'
        self._load_csv_files()

    def _get_classes_objects_verbs(self):
        def _get1(filename):
            with (filename).open('r') as f:
                lines = f.readlines()
            d = {}
            for x in lines:
                x = x.strip()
                d[x[:4]] = x[5:]
            return d

        def _get3(filename):
            with (filename).open('r') as f:
                lines = f.readlines()
            d = {}
            for x in lines:
                xs = x.strip().split()
                d[xs[0]] = (xs[1], xs[2])
            return d

        _action_labels = _get1(
                self._anno_fold/'Charades_v1_classes.txt')
        _object_labels = _get1(
                self._anno_fold/'Charades_v1_objectclasses.txt')
        _verb_labels = _get1(
                self._anno_fold/'Charades_v1_verbclasses.txt')
        _action_mapping_labels = _get3(
                self._anno_fold/'Charades_v1_mapping.txt')
        self._action_labels = _action_labels
        self._object_labels = _object_labels
        self._verb_labels = _verb_labels

        self.action_names = list(_action_labels.values())
        assert self.object_names == list(_object_labels.values())
        assert self.verb_names == list(_verb_labels.values())
        self.action_mapping = {
            _action_labels[a]: (_object_labels[o], _verb_labels[v])
            for a, (o, v) in _action_mapping_labels.items()}

    @staticmethod
    def _read_video_csv(filename, action_labels) -> List[Video_char]:
        videos = []
        with filename.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                objects = cast(List[Object_name_char],
                        row['objects'].split(';'))
                vid = Vid_char(Vid(row['id']))
                length = float(row['length'])
                actions = []
                if len(row['actions']):
                    for action_str in row['actions'].split(';'):
                        a, beg, end = action_str.split()
                        action_name = action_labels[a]
                        start_time = float(beg)
                        end_time = float(end)
                        action = Action_instance_char(
                            name=action_name, start_time=start_time,
                            end_time=end_time)
                        actions.append(action)
                relevance = int(row['relevance']) \
                        if row['relevance'] else None
                quality = int(row['quality']) \
                        if row['quality'] else None
                video: Video_char = {
                        'vid': vid,
                        'subject': row['subject'],
                        'scene': row['scene'],
                        'quality': quality,
                        'relevance': relevance,
                        'verified': row['verified'],
                        'script': row['script'],
                        'objects': objects,
                        'descriptions': row['descriptions'],
                        'actions': actions,
                        'length': length
                        }
                videos.append(video)
        return videos

    def _load_csv_files(self):
        self._get_classes_objects_verbs()

        # Load training and validation videos
        train_videos = self._read_video_csv(
            self._anno_fold/ 'Charades_v1_train.csv', self._action_labels)
        val_videos = self._read_video_csv(
            self._anno_fold/ 'Charades_v1_test.csv', self._action_labels)
        split = {}
        for v in train_videos:
            split[v['vid']] = 'train'
        for v in val_videos:
            split[v['vid']] = 'val'

        videos = []
        videos.extend(train_videos)
        videos.extend(val_videos)
        keys = [v['vid'] for v in videos]
        videos = dict(zip(keys, videos))
        self.split = split
        self.videos: Dict[Vid_daly, Video_daly] = videos


class Action_instance_char_ocv(Action_instance_char):
    start_frame: F0
    end_frame: F0


class Video_char_ocv(TypedDict):
    vid: Vid_char
    subject: str
    scene: str
    quality: Optional[int]
    relevance: Optional[int]
    verified: str
    script: str
    objects: List[Object_name_char]
    descriptions: str
    actions: List[Action_instance_char_ocv]
    length: float
    # ocv
    height: int
    width: int
    nframes: int
    est_length: float
    est_fps: float


class Dataset_charades_ocv(Dataset_charades):
    rstats: Dict[Vid_char, OCV_rstats]
    videos_ocv: Dict[Vid_char, Video_char_ocv]

    mirror: Literal['horus', 'gpuhost7', 'scratch2']
    resolution: Literal['original', '480']

    _rfold_dict = {'original': 'Charades_v1', '480': 'Charades_v1_480'}

    def __init__(self, mirror, resolution):
        super().__init__()
        self.mirror = mirror
        self.resolution = resolution

    def precompute_to_folder(self, fold):
        fold = Path(fold)
        # Construct path pattern
        _pattern = (self.root_path/'mirrors'/self.mirror/
                self._rfold_dict[self.resolution]/'{}.mp4')
        vids = list(self.videos.keys())
        isaver = snippets.Isaver_threading(
            small.mkdir(fold/'isave_rstats'), vids,
            lambda vid: compute_ocv_rstats(str(_pattern).format(vid)), 25, 8)
        isaver_items = isaver.run()
        rstats: Dict[Vid_daly, OCV_rstats] = dict(zip(vids, isaver_items))

        frames_unreachable = {}
        fps_mismatch = {}
        length_mismatch = {}
        actions_dropped = {}
        videos_ocv = {}
        for vid, rs in rstats.items():
            video = self.videos[vid]
            nframes = rs['max_pos_frames']
            est_fps = (nframes - 1) * 1000 / rs['max_pos_msec']
            est_length = np.round(nframes / est_fps, 4)
            if nframes != rs['frame_count']:
                frames_unreachable[vid] = (nframes, rs['frame_count'])
            if not np.isclose(est_fps, rs['fps']):
                fps_mismatch[vid] = (est_fps, rs['fps'])
            if not np.isclose(est_length, video['length']):
                length_mismatch[vid] = (est_length, video['length'])
            actions_ocv: List[Action_instance_char_ocv] = []
            for i, action in enumerate(video['actions']):
                action_ocv = copy.copy(action)
                start_frame = max(0, np.ceil(action['start_time'] * est_fps) - 1)
                end_frame = np.ceil(action['end_time'] * est_fps)
                if end_frame > nframes:
                    end_frame = nframes
                end_frame -= 1
                action_ocv['start_frame'] = int(start_frame)
                action_ocv['end_frame'] = int(end_frame)
                if start_frame > end_frame:
                    actions_dropped[(vid, i)] = \
                            (action_ocv, nframes, self.split[vid])
                else:
                    actions_ocv.append(action_ocv)
            video_ocv = copy.deepcopy(video)
            video_ocv = cast(Video_char_ocv, video_ocv)
            video_ocv.update({
                'actions': actions_ocv,
                'height': rs['height'],
                'width': rs['width'],
                'nframes': nframes,
                'est_length': est_length,
                'est_fps': est_fps
                })
            videos_ocv[vid] = video_ocv

        OCV_GOOD = len(frames_unreachable) == len(fps_mismatch) == 0
        MISMATCH_NUM_GOOD = len(length_mismatch) == 9844
        DROPPED_ONLY_TRAIN = all([a[2] == 'train'
            for a in actions_dropped.values()])
        DROPPED_NUM_GOOD = len(actions_dropped) == 7
        if not (OCV_GOOD and MISMATCH_NUM_GOOD and
                DROPPED_ONLY_TRAIN and DROPPED_NUM_GOOD):
            log.warn('Different error stats')
        precomputed_stats = {
            'rstats': rstats,
            'videos_ocv': videos_ocv}
        small.save_pkl(fold/'precomputed_stats.pkl', precomputed_stats)

    def populate_from_folder(self, fold):
        fold = Path(fold)
        precomputed_stats = small.load_pkl(fold/'precomputed_stats.pkl')
        self.rstats = precomputed_stats['rstats']
        self.videos_ocv = precomputed_stats['videos_ocv']
