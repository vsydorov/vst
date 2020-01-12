"""
Basic classes that allow working with external datasets
"""
import xml.etree.ElementTree as ET
import subprocess
import csv
import hashlib
import re
import pandas as pd
import numpy as np
import cv2  # type: ignore
import logging
import concurrent.futures
from abc import abstractmethod, ABC
from tqdm import tqdm
from pathlib import Path  # NOQA
from collections import OrderedDict
from mypy_extensions import TypedDict
from typing import Dict, List, Tuple, cast, NewType, Any

from vsydorov_tools import small, cv as vt_cv

from thes.filesystem import get_dataset_path
from thes.tools import snippets


log = logging.getLogger(__name__)

Vid = NewType('Vid', str)  # Unique video ID
Dataset_subset = NewType('Dataset_subset', str)  # train/val/test


def query_ocv_stats_from_video(video_path, n_tries):
    with vt_cv.video_capture_open(video_path, n_tries) as vcap:
        height, width = vt_cv.video_getHW(vcap)
        framecount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Try to iterate
        vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret = vcap.grab()
            if ret is False:
                break
        frames_reached = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
    return framecount, frames_reached, height, width


def _update_with_ocv_stats(root_path, unfinished_video_dict,
        max_workers=8, n_tries=10):
    # Check opencv stats
    def _ocv_stats_query(vid, video):
        video_path = root_path/video['rel_video_path']
        framecount, frames_reached, height, width = \
                query_ocv_stats_from_video(video_path, n_tries)
        video['framecount'] = framecount
        video['frames_reached'] = frames_reached
        video['height'] = height
        video['width'] = width
        return vid, video
    io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)
    io_futures = []
    for vid, video in unfinished_video_dict.items():
        io_futures.append(io_executor.submit(
            _ocv_stats_query, vid, video))
    video_dict_with_ocv_stats = {}
    for io_future in tqdm(concurrent.futures.as_completed(io_futures),
            total=len(io_futures), desc='check_accessible_frames'):
        vid, video = io_future.result()
        video_dict_with_ocv_stats[vid] = video
    io_executor.shutdown()
    return video_dict_with_ocv_stats


def query_ocv_stats_from_video_v2(video_path, n_tries):
    with vt_cv.video_capture_open(video_path, n_tries) as vcap:
        height, width = vt_cv.video_getHW(vcap)
        reported_framecount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        reported_fps = vcap.get(cv2.CAP_PROP_FPS)
        # Try to iterate
        vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret = vcap.grab()
            if ret is False:
                break
        frames_reached = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
        ms_reached = int(vcap.get(cv2.CAP_PROP_POS_MSEC))
    qstats = {
        'reported_framecount': reported_framecount,
        'reported_fps': reported_fps,
        'frames_reached': frames_reached,
        'ms_reached': ms_reached,
        'height': height,
        'width': width}
    return qstats


def _update_with_ocv_stats_v2(root_path, unfinished_video_dict,
        max_workers=8, n_tries=10):
    # Check opencv stats
    def _ocv_stats_query(vid, video):
        video_path = root_path/video['rel_video_path']
        qstats = query_ocv_stats_from_video_v2(video_path, n_tries)
        video['qstats'] = qstats
        return vid, video
    io_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=8)
    io_futures = []
    for vid, video in unfinished_video_dict.items():
        io_futures.append(io_executor.submit(
            _ocv_stats_query, vid, video))
    video_dict_with_ocv_stats = {}
    for io_future in tqdm(concurrent.futures.as_completed(io_futures),
            total=len(io_futures), desc='check_accessible_frames'):
        vid, video = io_future.result()
        video_dict_with_ocv_stats[vid] = video
    io_executor.shutdown()
    return video_dict_with_ocv_stats


class PrecomputableDataset(ABC):
    root_path: Path

    @abstractmethod
    def precompute_to_folder(self, folder):
        raise NotImplementedError()

    @abstractmethod
    def populate_from_folder(self, folder):
        raise NotImplementedError()


class ActionDataset(ABC):
    split: Dict[Any, Dataset_subset]
    split_id: int
    action_names: List

    @property
    @abstractmethod
    def vids(self):
        raise NotImplementedError()

    @abstractmethod
    def set_split(self, split_id):
        raise NotImplementedError()

    @staticmethod
    def list_to_hash(
            lst: List[str],
            hashlen=10) -> str:
        h = hashlib.blake2b(digest_size=hashlen)  # type: ignore
        for l in lst:
            h.update(l.encode())
        return h.hexdigest()

    def _print_split_stats(self):
        # Little bit of stats
        split = self.split
        split_id = self.split_id
        s = pd.Series(split)
        svc = s.value_counts()
        if 'train' in svc and 'val' in svc:
            svc['trainval'] = svc['train'] + svc['val']
        log.info('Split {}'.format(split_id))
        log.info('Split value counts:\n{}'.format(svc))
        if 'val' in svc:
            vids_val = s[s == 'val'].index
            val_hash = self.list_to_hash(vids_val.tolist())
            log.info(f'Validation subset hash {split_id}: {val_hash}')

    @property
    def num_classes(self):
        return len(self.action_names)

    def establish_stats(self):
        pass


RVideoMP4 = TypedDict('RVideoMP4', {
    'height': int,
    'width': int,
    'nframes': int,
    'rvideo_path': str,
    'avideo_path': Path
    })
VideoMP4 = TypedDict('VideoMP4', {
    'height': int,
    'width': int,
    'nframes': int,
    'video_path': Path
    })
Charades_vid = NewType('Charades_vid', Vid)
Charades_action_name = NewType('Charades_action_name', str)
Charades_object_name = NewType('Charades_object_name', str)
Charades_video = TypedDict('Charades_video', {
    'vid': Charades_vid,
    'subject': str,
    'scene': str,
    'quality': int,
    'relevance': int,
    'verified': str,
    'script': str,
    'objects': List[Charades_object_name],
    'descriptions': str,
    'action_names': List[Charades_action_name],
    'actions': List[Tuple[Charades_action_name, float, float]],
    'length': float,
})


class DatasetCharadesBase(
        ActionDataset,
        PrecomputableDataset):
    root_path: Path
    video_odict: "OrderedDict[Charades_vid, Charades_video]"
    action_names: List[Charades_action_name]
    object_names: List[Charades_object_name]
    verb_names: List[str]
    split: Dict[Charades_vid, Dataset_subset]

    videos_original: Dict[Charades_vid, RVideoMP4]

    mirror_str: str
    source_str: str
    source_videos: Dict[Charades_vid, RVideoMP4]

    def __init__(self):
        super().__init__()
        self.root_path = get_dataset_path('action/charades_take2')
        self.anno_trainval_fold = self.root_path/'annotation/Charades'

    def set_split(self, split_id):
        self.split_id = split_id
        assert self.split_id == 0

    @property
    def vids(self):
        return list(self.video_odict.keys())

    @abstractmethod
    def set_video_source(self, mirror, source):
        raise NotImplementedError()

    def get_source_video_for_vid(self, vid) -> VideoMP4:
        assert self.mirror_str is not None
        assert self.source_videos is not None
        v_source = self.source_videos[vid]
        if v_source.get('avideo_path') is not None:
            video_path = v_source['avideo_path']
        else:
            video_path = Path(v_source['rvideo_path'].format(
                    root=self.root_path, mirror=self.mirror_str))

        v = VideoMP4(height=v_source['height'],
                width=v_source['width'],
                nframes=v_source['nframes'],
                video_path=video_path)
        return v

    # ### PrecomputableDataset ###
    def _get_classes_objects_verbs(self):
        # Load classes
        with (self.anno_trainval_fold/
                'Charades_v1_classes.txt').open('r') as f:
            lines = f.readlines()
        action_names = [x.strip()[5:] for x in lines]
        # Load objects
        with (self.anno_trainval_fold/
                'Charades_v1_objectclasses.txt').open('r') as f:
            lines = f.readlines()
        object_names = [x.strip()[5:] for x in lines]
        # Load verbs
        with (self.anno_trainval_fold/
                'Charades_v1_verbclasses.txt').open('r') as f:
            lines = f.readlines()
        verb_names = [x.strip()[5:] for x in lines]
        return action_names, object_names, verb_names

    @staticmethod
    def _read_video_csv(csv_filename, action_names):
        videos = []
        with csv_filename.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                objects = row['objects'].split(';')
                if len(row['actions']) == 0:
                    actions = []
                    video_action_names = []
                else:
                    actions = []
                    for _action in row['actions'].split(';'):
                        vid, beg, end = _action.split()
                        action_name = action_names[int(vid[1:])]
                        actions.append((action_name, float(beg), float(end)))
                    video_action_names = [a[0] for a in actions]
                relevance = int(row['relevance']) \
                        if row['relevance'] else None
                quality = int(row['quality']) \
                        if row['quality'] else None
                video = {
                        'vid': Charades_vid(row['id']),
                        'subject': row['subject'],
                        'scene': row['scene'],
                        'quality': quality,
                        'relevance': relevance,
                        'verified': row['verified'],
                        'script': row['script'],
                        'objects': objects,
                        'descriptions': row['descriptions'],
                        'action_names': video_action_names,
                        'actions': actions,
                        'length': float(row['length'])
                        }
                videos.append(video)
            return videos


class DatasetCharades(DatasetCharadesBase):
    """
    Charades dataset in our interpretation (trainval split that was released)
    """
    videos_480: Dict[Charades_vid, RVideoMP4]

    def __init__(self):
        super().__init__()

    @staticmethod
    def _charades_cumulative_action_durations(action_names, dataframe):
        action_durations = {k: 0.0 for k in action_names}
        for actions in dataframe.actions.values:
            for action, tb, te in actions:
                duration = te-tb
                action_durations[action] += duration
        return pd.Series(action_durations)

    def set_video_source(self, mirror, source):
        self.mirror_str = mirror
        self.source_str = source
        if self.source_str == '480':
            self.source_videos = self.videos_480
        elif self.source_str == 'original':
            self.source_videos = self.videos_original
        else:
            raise NotImplementedError(
                    'Wrong source {}'.format(self.source_str))

    # ### PrecomputableDataset ###
    def _compute_light_stats(self):
        action_names, object_names, verb_names = \
                self._get_classes_objects_verbs()

        # Load training and validation videos
        train_videos = self._read_video_csv(
                self.anno_trainval_fold/
                'Charades_v1_train.csv', action_names)
        val_videos = self._read_video_csv(
                self.anno_trainval_fold/
                'Charades_v1_test.csv', action_names)
        split = {}
        for v in train_videos:
            split[v['vid']] = 'train'
        for v in val_videos:
            split[v['vid']] = 'val'

        videos = []
        videos.extend(train_videos)
        videos.extend(val_videos)
        video_odict = OrderedDict()
        for v in videos:
            video_odict[v['vid']] = v

        light_stats = {
                'video_odict': video_odict,
                'split': split,
                'action_names': action_names,
                'object_names': object_names,
                'verb_names': verb_names
                }
        return light_stats

    def precompute_to_folder(self, fold):
        light_stats = self._compute_light_stats()
        video_odict = light_stats['video_odict']

        default_mirror = 'horus'
        # videos480
        videos_480 = {}
        for vid, video in video_odict.items():
            videos_480[vid] = {
                'vid': vid,
                'rel_video_path': 'mirrors/{}/Charades_v1_480/{}.mp4'.format(
                    default_mirror, vid)}
        videos_480 = small.stash2(fold/'videos_480_ocv_stats.pkl')(
                _update_with_ocv_stats, self.root_path, videos_480, 20)

        # videos_original
        videos_original = {}
        for vid, video in video_odict.items():
            videos_original[vid] = {
                'vid': vid,
                'rel_video_path': 'mirrors/{}/Charades_v1/{}.mp4'.format(
                    default_mirror, vid)}
        videos_original = small.stash2(fold/'videos_original_ocv_stats.pkl')(
                _update_with_ocv_stats, self.root_path, videos_original, 20)

        # Make sure all frames are reachable
        df_480 = pd.DataFrame(videos_480).T
        framediff = (df_480['framecount'] -
                df_480['frames_reached']).abs().max()
        assert framediff == 0, 'all frames must be reachable'
        df_orig = pd.DataFrame(videos_original).T
        framediff = (df_orig['framecount'] -
                df_orig['frames_reached']).abs().max()
        assert framediff == 0, 'all frames must be reachable'

        # Assign video infos (relative paths)
        for vdict in [videos_480, videos_original]:
            for vid, v in list(vdict.items()):
                last = v['rel_video_path'].split('/')[-2:]
                rvideo_path = '{root}/mirrors/{mirror}/' + '/'.join(last)
                vnew = {
                    'height': v['height'],
                    'width': v['width'],
                    'nframes': v['framecount'],
                    'rvideo_path': rvideo_path,
                    'avideo_path': None}
                vdict[vid] = vnew

        small.save_pkl(fold/'light_stats.pkl', light_stats)
        small.save_pkl(fold/'videos_480.pkl', videos_480)
        small.save_pkl(fold/'videos_original.pkl', videos_original)

    def populate_from_folder(self, fold):
        fold = Path(fold)
        light_stats = small.load_pkl(fold/'light_stats.pkl')
        self.video_odict = light_stats['video_odict']
        self.split = light_stats['split']
        self.action_names = light_stats['action_names']
        self.object_names = light_stats['object_names']
        self.verb_names = light_stats['verb_names']

        self.videos_480 = small.load_pkl(fold/'videos_480.pkl')
        self.videos_original = small.load_pkl(fold/'videos_original.pkl')

    # ### ActionDataset ###

    @staticmethod
    def _balanced_subject_separation(
            seed, fraction, df_subset, action_names, n_samplings=15):
        train_subjects = df_subset.subject.unique()

        random_state = np.random.RandomState(seed)
        times_train = \
                DatasetCharades._charades_cumulative_action_durations(
                    action_names, df_subset)

        # Do several samplings over subjects
        # take one with biggest smallest fraction
        def get_subjects_and_times():
            perm_subjects = random_state.permutation(train_subjects)
            subj_fraction = perm_subjects[:int(len(perm_subjects)*fraction)]
            # Action stats
            df_fraction = df_subset[df_subset.subject.isin(subj_fraction)]
            times_fraction = \
                    DatasetCharades._charades_cumulative_action_durations(
                        action_names, df_fraction)
            times_div = (times_fraction/times_train)
            times_min = times_div.min()
            times_max = times_div.max()
            return subj_fraction, times_min, times_max

        samples = [get_subjects_and_times() for i in range(n_samplings)]
        # times_min must be above 0
        samples = [x for x in samples if x[1] > 0]
        assert len(samples) > 0, '>= 1 action missing'
        # time_max must be below 1
        samples = [x for x in samples if x[2] < 1]
        assert len(samples) > 0, '>=1 action fully sampled'
        # time_max must be closest to fraction
        fraction_diff = [abs(fraction-x[1]) for x in samples]
        closest_id = np.argmin(fraction_diff)
        sampled_subjects = samples[closest_id][0]
        df_separated = df_subset[df_subset.subject.isin(sampled_subjects)]
        return df_separated


class DatasetCharadesTest(DatasetCharadesBase):
    """
    Charades test dataset (test split available for challenge)
    """
    def __init__(self):
        super().__init__()
        self.anno_test_fold = \
                self.root_path/'annotation/vu17_charades_test'

    def set_video_source(self, mirror, source):
        self.mirror_str = mirror
        self.source_str = source
        if self.source_str == '480':
            raise NotImplementedError(
                    'source 480 not implemented for test')
        elif self.source_str == 'original':
            self.source_videos = self.videos_original
        else:
            raise NotImplementedError(
                    'Wrong source {}'.format(self.source_str))

    # ### PrecomputableDataset ###
    def _compute_light_stats(self):
        action_names, object_names, verb_names = \
                self._get_classes_objects_verbs()

        # Load training and validation videos
        test_videos = self._read_video_csv(
                self.anno_test_fold/
                'Charades_vu17_test.csv', action_names)

        split = {}
        for v in test_videos:
            split[v['vid']] = 'test'

        videos = []
        videos.extend(test_videos)
        video_odict = OrderedDict()
        for v in videos:
            video_odict[v['vid']] = v

        light_stats = {
                'video_odict': video_odict,
                'split': split,
                'action_names': action_names,
                'object_names': object_names,
                'verb_names': verb_names
                }
        return light_stats

    def precompute_to_folder(self, fold):
        light_stats = self._compute_light_stats()
        video_odict = light_stats['video_odict']

        default_mirror = 'gpuhost7'
        # videos_original
        videos_original = {}
        for vid, video in video_odict.items():
            videos_original[vid] = {
                'vid': vid,
                'rel_video_path': 'mirrors/{}/Charades_vu17_test/{}.mp4'.format(
                    default_mirror, vid)}
        videos_original = small.stash2(fold/'videos_original_ocv_stats.pkl')(
                _update_with_ocv_stats, self.root_path, videos_original, 20)

        # Some frames are unreachable - removing audio helps (for some reason)
        fold_reencoded = small.mkdir(fold/'reencoded')
        df_orig = pd.DataFrame(videos_original).T
        framediffs = (df_orig['framecount'] -
                df_orig['frames_reached']).abs()
        df_broken = df_orig.ix[framediffs.ix[framediffs!=0].index]
        for vid, line in tqdm(df_broken.iterrows()):
            in_path = self.root_path/line.rel_video_path
            out_path = fold_reencoded/'{}.mp4'.format(line.vid)
            # prepared = "ffmpeg -y -i '{}' -strict -2 -c:a copy -vcodec libx265 -x265-params crf=25 '{}'".format(in_path, out_path)
            prepared = "ffmpeg -y -i '{}' -an '{}'".format(in_path, out_path)
            result = subprocess.run(prepared, shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=20)
            if result.returncode:
                raise OSError(result.stdout)
            df_broken.ix[vid, 'new_path'] = out_path
            # Also check new framecount
            framecount, frames_reached, height, width = \
                    query_ocv_stats_from_video(out_path, 10)
            df_broken.ix[vid, 'new_framecount'] = framecount
            df_broken.ix[vid, 'new_frames_reached'] = frames_reached

        # Update videos_original with new reencoded videos
        for vid, line in df_broken.iterrows():
            videos_original[vid]['framecount'] = line['new_framecount']
            videos_original[vid]['frames_reached'] = \
                    line['new_frames_reached']
            videos_original[vid]['abs_video_path'] = line['new_path']
            del videos_original[vid]['rel_video_path']

        df_orig_ = pd.DataFrame(videos_original).T
        framediffs_ = (df_orig_['framecount'] -
                df_orig_['frames_reached']).abs()
        assert framediffs_.max() == 0, 'all frames must be reachable'

        # Assign video infos (relative paths)
        for vdict in [videos_original]:
            for vid, v in list(vdict.items()):
                vnew = {
                    'height': v['height'],
                    'width': v['width'],
                    'nframes': v['framecount']}
                if v.get('abs_video_path') is not None:
                    vnew['avideo_path'] = v['abs_video_path']
                    vnew['rvideo_path'] = None
                else:
                    last = v['rel_video_path'].split('/')[-2:]
                    rvideo_path = '{root}/mirrors/{mirror}/' + '/'.join(last)
                    vnew['avideo_path'] = None
                    vnew['rvideo_path'] = rvideo_path
                vdict[vid] = vnew

        small.save_pkl(fold/'light_stats.pkl', light_stats)
        small.save_pkl(fold/'videos_original.pkl', videos_original)

    def populate_from_folder(self, fold):
        fold = Path(fold)
        light_stats = small.load_pkl(fold/'light_stats.pkl')
        self.video_odict = light_stats['video_odict']
        self.split = light_stats['split']
        self.action_names = light_stats['action_names']
        self.object_names = light_stats['object_names']
        self.verb_names = light_stats['verb_names']

        self.videos_original = small.load_pkl(fold/'videos_original.pkl')


"""HMDB51"""


HMDB51_vid = NewType('HMDB51_vid', Vid)
HMDB51_action_name = NewType('HMDB51_action_name', str)
HMDB51_video = TypedDict('HMDB51_video', {
        'vid': HMDB51_vid,
        'action_name': HMDB51_action_name,
})
RVideoMP4_reached = TypedDict('RVideoMP4_reached', {
    'height': int,
    'width': int,
    'length_reached': float,  # seconds
    'frames_reached': int,
    'rvideo_path': str
    })
VideoMP4_reached = TypedDict('VideoMP4_reached', {
    'height': int,
    'width': int,
    'length_reached': float,  # seconds
    'frames_reached': int,
    'video_path': Path
    })


class DatasetHMDB51(
        ActionDataset,
        PrecomputableDataset):
    """
    Videos can not be read with opencv as is:
        - 6349 out of 6371 are missing last frame
        - We only try to access frames that can be reached
    """
    root_path: Path

    action_names = cast(List[HMDB51_action_name], ['brush_hair', 'cartwheel',
        'catch', 'chew', 'clap', 'climb', 'climb_stairs', 'dive', 'draw_sword',
        'dribble', 'drink', 'eat', 'fall_floor', 'fencing', 'flic_flac',
        'golf', 'handstand', 'hit', 'hug', 'jump', 'kick', 'kick_ball', 'kiss',
        'laugh', 'pick', 'pour', 'pullup', 'punch', 'push', 'pushup',
        'ride_bike', 'ride_horse', 'run', 'shake_hands', 'shoot_ball',
        'shoot_bow', 'shoot_gun', 'sit', 'situp', 'smile', 'smoke',
        'somersault', 'stand', 'swing_baseball', 'sword', 'sword_exercise',
        'talk', 'throw', 'turn', 'walk', 'wave'])

    video_odict: "OrderedDict[HMDB51_vid, HMDB51_video]"
    splits: List[Dict[HMDB51_vid, Dataset_subset]]

    videos_original: Dict[Charades_vid, RVideoMP4_reached]

    mirror_str: str
    source_str: str
    source_videos: Dict[Charades_vid, RVideoMP4_reached]

    def __init__(self):
        super().__init__()
        self.root_path = get_dataset_path('action/hmdb51_take2')
        self.anno_trainval_fold = (self.root_path/
                'annotations'/
                'test_train_splits'/
                'testTrainMulti_7030_splits')

    @property
    def vids(self):
        return list(self.video_odict.keys())

    def set_video_source(self, mirror, source):
        self.mirror_str = mirror
        self.source_str = source
        if self.source_str == 'original':
            self.source_videos = self.videos_original
        elif self.source_str == 'stabilized':
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                    'Wrong source {}'.format(self.source_str))

    def get_source_video_for_vid(self, vid) -> VideoMP4_reached:
        assert self.mirror_str is not None
        assert self.source_videos is not None
        v_source = self.source_videos[vid]
        video_path = Path(v_source['rvideo_path'].format(
                root=self.root_path, mirror=self.mirror_str))
        v = VideoMP4_reached(
                height=v_source['height'],
                width=v_source['width'],
                length_reached=v_source['length_reached'],
                frames_reached=v_source['frames_reached'],
                video_path=video_path)
        return v

    # ### PrecomputableDataset ###

    @staticmethod
    def _read_splits(
            splits_folder: Path
             ) -> List[Dict[HMDB51_vid, Dataset_subset]]:
        splitline_re = r'^(\S+)\.avi (0|1|2)\s*$'
        video_splits: List[Dict[HMDB51_vid, Dataset_subset]] = []
        for split_id in (1, 2, 3):
            splitfiles = list(splits_folder.glob(
                '*_test_split{}.txt'.format(split_id)))
            assert len(splitfiles) == 51

            video_split: Dict[HMDB51_vid, Dataset_subset] = {}
            for splitfile in splitfiles:
                split_info = {}
                with splitfile.open('r') as f:
                    splitlines = f.read().splitlines()
                    for splitline in splitlines:
                        matched = re.search(splitline_re, str(splitline))
                        if matched is None:
                            raise ValueError(
                                'File {}: Line {} does not match format {}'.
                                format(splitfile, splitline, splitline_re))
                        video_name, number = matched.groups()
                        if number == '0':
                            continue
                        elif number == '1':
                            subset = 'train'
                        elif number == '2':
                            subset = 'test'
                        else:
                            raise NotImplementedError()
                        # Assign split
                        vid = HMDB51_vid(Vid(video_name))
                        split_info[vid] = Dataset_subset(subset)
                split_counts = pd.Series(split_info).value_counts()
                # As per README
                assert split_counts['train'] == 70
                assert split_counts['test'] == 30
                video_split.update(split_info)
            video_splits.append(video_split)
        return video_splits

    @staticmethod
    def _check_folders_for_videos(
            root_mirror_path, video_folder,
            ) -> Dict[HMDB51_vid, Dict]:
        """
        - Parse folders, avoiding hidden ones
        - Remove videos that are not present in any split
        """
        unfinished_video_dict: Dict[HMDB51_vid, Dict] = {}
        action_folders = snippets.listdir_nondot(video_folder)
        for action_folder in action_folders:
            if not action_folder.is_dir():
                continue
            action_name = HMDB51_action_name(action_folder.name)
            video_files = snippets.listdir_nondot(action_folder)
            for video_file in video_files:
                video_name = video_file.name
                vid = HMDB51_vid(video_file.with_suffix('').name)
                rel_video_path = video_file.relative_to(root_mirror_path)
                video_unfinished = {
                        'vid': vid,
                        'video_name': video_name,
                        'action_name': action_name,
                        'rel_video_path': rel_video_path}
                unfinished_video_dict[vid] = video_unfinished
        return unfinished_video_dict

    def _compute_light_stats(self, root_mirror_path):
        # Load splits
        splits = self._read_splits(self.anno_trainval_fold)

        # // Load files, ensure proper number
        unfinished_video_dict = self._check_folders_for_videos(
                root_mirror_path,
                root_mirror_path/'hmdb51_org')
        # for x in `ls`; do echo `ls $x|wc -l`; done | paste -sd+ | bc
        assert len(unfinished_video_dict) == 6766

        # Filter out unused videos
        good_vids = np.unique([k for vs in splits for k in vs])
        log.debug('Remove videos not present in either split {}->{}'.format(
            len(unfinished_video_dict), len(good_vids)))
        filtered_video_dict: Dict[HMDB51_vid, Dict] = {}
        for vid in good_vids:
            filtered_video_dict[vid] = unfinished_video_dict[vid]
        assert len(filtered_video_dict) == 6371

        video_odict = OrderedDict(sorted(filtered_video_dict.items()))
        light_stats = {
                'video_odict': video_odict,
                'splits': splits,
                }
        return light_stats

    def precompute_to_folder(self, fold):
        default_mirror = 'horus'
        root_mirror_path = self.root_path/'mirrors'/default_mirror

        light_stats = self._compute_light_stats(root_mirror_path)
        video_odict = light_stats['video_odict']

        # Load opencv stats
        unfinished_vodict_with_qstats = small.stash()(
                fold/'unfinished_vodict_with_stats.pkl',
                _update_with_ocv_stats_v2,
                root_mirror_path, video_odict)

        # Finalize the odict
        videos_original = OrderedDict()
        for vid, v in unfinished_vodict_with_qstats.items():
            video = {
                'height': v['qstats']['height'],
                'width': v['qstats']['width'],
                'length_reached': v['qstats']['ms_reached']/1000,
                'frames_reached': v['qstats']['frames_reached'],
                'rvideo_path':
                    '{root}/mirrors/{mirror}/' + str(v['rel_video_path'])}
            videos_original[vid] = video

        video_odict_clean = OrderedDict()
        for k, v in video_odict.items():
            video_odict_clean[k] = {
                    'vid': v['vid'],
                    'action_name': v['action_name']}

        small.save_pkl(fold/'splits.pkl', light_stats['splits'])
        small.save_pkl(fold/'video_odict.pkl', video_odict_clean)
        small.save_pkl(fold/'videos_original.pkl', videos_original)

    def populate_from_folder(self, fold):
        fold = Path(fold)

        self.video_odict = small.load_pkl(fold/'video_odict.pkl')
        self.splits = small.load_pkl(fold/'splits.pkl')
        self.videos_original = small.load_pkl(fold/'videos_original.pkl')

    # ### ActionDataset ###
    def _establish_validation_set(self, split, fraction, seed):
        """
        Separate 'validation' subset from the training set "fraction" - rough
        fraction of training set to be assigned to validation. Per each action
        we sample "fraction" of videos to training set

        Note:
            - fraction == 0: do nothing
            - 0 < fraction < eps: at least one meta_video per each category
        """
        if np.isclose(fraction, 0):
            return split
        random_state = np.random.RandomState(seed)
        tail_pattern = (r'(.*)_([fuhl])_(cm|nm)_(np\d+)_'
                r'(fr|le|ri|ba)_(bad|med|goo)_(\d+)$')

        # Names of videos from which clips are sampled
        meta_videos = []
        for x in self.video_odict.keys():
            matched = re.search(tail_pattern, x)
            meta_videos.append(matched.group(1))
        assert len(np.unique(meta_videos)) == 2251

        df = pd.DataFrame(self.video_odict).T
        df['meta_video'] = meta_videos

        splits_w_validation: Dict[HMDB51_vid, Dataset_subset]
        s_split = pd.Series(split)
        s_split_w_validation = s_split.copy()

        vids_train = s_split[s_split == 'train'].index
        df_train = df.ix[vids_train]
        for action_name in self.action_names:
            df_train_action = df_train.query(
                    'action_name == @action_name')

            meta_video_names_train = df_train_action.meta_video.unique()

            # Add metavideos to new split until capacity reached
            N_validation_videos = int(len(df_train_action) * fraction)
            val_vids = []
            for meta_video_name in random_state.permutation(
                    meta_video_names_train):
                if len(val_vids) >= N_validation_videos:
                    break
                meta_video_vids = df_train_action.query(
                        'meta_video == @meta_video_name').index
                val_vids.extend(meta_video_vids)
            s_split_w_validation.ix[val_vids] = 'val'
        split_w_validation = s_split_w_validation.to_dict()
        return split_w_validation

    def set_split(self, split_id):
        self.split = self.splits[split_id]
        self.split_id = split_id

    def establish_validation_set(self, fraction, seed, cachefold):
        """
        Caching for validation sets
        """
        FS = 'split_{:d}_frac{:.2f}_seed{:03d}.pkl'
        split_filename = cachefold/FS.format(self.split_id, fraction, seed)
        split_w_validation = small.stash2(split_filename)(
                self._establish_validation_set,
                self.split, fraction, seed)
        self.split = split_w_validation
        self._print_split_stats()


DALY_vid = NewType('DALY_vid', Vid)
DALY_action_name = NewType('DALY_action_name', str)
DALY_object_name = NewType('DALY_object_name', str)
DALY_joint_name = NewType('DALY_joint_name', str)
DALY_instance_flags = TypedDict('DALY_instance_flags', {
    'isSmall': bool,
    'isReflection': bool,
    'isShotcut': bool,
    'isZoom': bool,
    'isAmbiguous': bool,
    'isOccluded': bool,
    'isOutsideFOV': bool
})
DALY_keyframe = TypedDict('DALY_keyframe', {
    'boundingBox': np.ndarray,  # xmin, ymin, xmax, ymax, ltrd
    'objects': np.ndarray,
    'frameNumber': int,
    'pose': np.ndarray,
    'time': float
})
DALY_instance = TypedDict('DALY_instance', {
    'beginTime': float,
    'endTime': float,
    'flags': DALY_instance_flags,
    'keyframes': List[DALY_keyframe]
})
DALY_video = TypedDict('DALY_video', {
    'vid': DALY_vid,
    'suggestedClass': str,
    'instances': Dict[DALY_action_name, List[DALY_instance]],
    # Meta
    'duration': float,
    'nbframes_ffmpeg': int,
    'fps': float
})


class DatasetDALY(object):
    root_path: Path
    action_names: List[DALY_action_name]
    object_names: List[DALY_object_name]
    joint_names: List[DALY_joint_name]
    split: Dict[DALY_vid, Dataset_subset]
    video_odict: "OrderedDict[DALY_vid, DALY_video]"

    source_videos = Dict[DALY_vid, RVideoMP4_reached]

    def __init__(self):
        super().__init__()
        self.root_path = get_dataset_path('action/DALY')
        self.videos_fold = (self.root_path/'videos')

    def _compute_light_stats(self):
        pkl_path = self.root_path/'daly1.1.0.pkl'
        info = small.load_py2_pkl(pkl_path)

        action_names = info['labels']
        object_names = info['objectList']
        joint_names = info['joints']

        video_odict = OrderedDict()
        for video_name, v in info['annot'].items():
            vid = video_name.split('.')[0]
            meta = info['metadata'][video_name]
            video = {
                'vid': vid,
                'suggestedClass': v['suggestedClass'],
                'instances': v['annot'],
                'duration': meta['duration'],
                'nbframes_ffmpeg': meta['nbframes_ffmpeg'],
                'fps': meta['fps']
            }
            video_odict[vid] = video

        split = {k: 'train' for k in video_odict.keys()}
        for video_name in info['splits'][0]:
            vid = video_name.split('.')[0]
            split[vid] = 'test'

        light_stats = {
            'video_odict': video_odict,
            'split': split,
            'action_names': action_names,
            'object_names': object_names,
            'joint_names': joint_names
        }
        return light_stats

    def populate_from_folder(self, fold):
        fold = Path(fold)
        light_stats = small.load_pkl(fold/'light_stats.pkl')
        source_videos: Dict[DALY_vid, RVideoMP4_reached] = \
                small.load_pkl(fold/'source_videos.pkl')

        self.source_videos = source_videos
        self.video_odict = light_stats['video_odict']
        self.split = light_stats['split']
        self.action_names = light_stats['action_names']
        self.object_names = light_stats['object_names']
        self.joint_names = light_stats['joint_names']

    def precompute_to_folder(self, fold):
        fold = Path(fold)
        light_stats = self._compute_light_stats()
        video_odict = light_stats['video_odict']

        # Get ocv stats
        videos_w_ocv = {}
        for vid, video in video_odict.items():
            videos_w_ocv[vid] = {
                'vid': vid,
                'rel_video_path': f'videos/{vid}.mp4'}
        videos_w_ocv = small.stash2(fold/'videos_ocv_stats.pkl')(
                _update_with_ocv_stats_v2, self.root_path, videos_w_ocv, 20)

        # Confirm reachable frames
        for vid, video in videos_w_ocv.items():
            qstats = video['qstats']
            assert qstats['reported_framecount'] == \
                    qstats['frames_reached']

        # Record duration/frames mismatches
        duration_mismatches = {}
        frames_mismatches = {}
        for vid, video in videos_w_ocv.items():
            ovideo = video_odict[vid]
            qstats = video['qstats']
            ocv_reached = qstats['ms_reached']
            meta_duration = ovideo['duration']*1000
            frames_reached = qstats['frames_reached']
            meta_frames = ovideo['nbframes_ffmpeg']
            if ocv_reached != meta_duration:
                duration_mismatches[vid] = [ocv_reached, meta_duration]
            if frames_reached != meta_frames:
                frames_mismatches[vid] = [frames_reached, meta_frames]

        d_mi = pd.DataFrame(duration_mismatches).T
        d_mi.columns = ['ocv', 'meta']
        d_mi['diff'] = d_mi['meta'] - d_mi['ocv']

        f_mi = pd.DataFrame(frames_mismatches).T
        f_mi.columns = ['ocv', 'meta']
        f_mi['diff'] = f_mi['meta'] - f_mi['ocv']

        """ We prioritize OCV stats everywhere """
        relpath_source_videos: Dict[DALY_vid, RVideoMP4_reached] = {}
        for vid, v in videos_w_ocv.items():
            video = {
                'height': v['qstats']['height'],
                'width': v['qstats']['width'],
                'length_reached': v['qstats']['ms_reached']/1000,
                'frames_reached': v['qstats']['frames_reached'],
                'rvideo_path': str(v['rel_video_path'])}
            relpath_source_videos[vid] = video

        # replace rpaths -> paths
        source_videos = {}
        for vid, v in relpath_source_videos.items():
            vnew = v.copy()
            vnew['video_path'] = self.root_path/v['rvideo_path']
            del vnew['rvideo_path']
            source_videos[vid] = vnew

        small.save_pkl(fold/'source_videos.pkl', source_videos)
        small.save_pkl(fold/'light_stats.pkl', light_stats)


# Detection datasets
VOC_object = TypedDict('VOC_object', {
    'name': str,
    'pose': str,
    'difficult': bool,
    'truncated': bool,
    'box': np.ndarray
})
VOC_image_annotation = TypedDict('VOC_image_annotation', {
    'filename': str,
    'filepath': Path,
    'size_WHD': Tuple[int, int, int],
    'objects': List[VOC_object]
})


class DatasetVOC(object):
    root_path = None  # type: Path
    object_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
            'tvmonitor']  # type: List[str]
    annotations_per_split: "Dict[str, OrderedDict[str, VOC_image_annotation]]"

    def _get_image_index_for_subset(self, prefix, subset: str) -> List[str]:
        # Load "image index" for a subset (which indices belong to it)
        filename = self.root_path/prefix/'ImageSets/Main'/f'{subset}.txt'
        with filename.open('r') as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_annotations_from_index(
            self, prefix, image_index
            ) -> "OrderedDict[str, VOC_image_annotation]":
        annotations: "OrderedDict[str, VOC_image_annotation]" = OrderedDict()
        # Load region annotation
        for index in tqdm(image_index):
            xml_filename = self.root_path/prefix/'Annotations'/f'{index}.xml'
            xml_tree = ET.parse(xml_filename)
            filename = str(xml_tree.find('filename').text)  # type: ignore
            filepath = self.root_path/prefix/'JPEGImages'/filename
            assert xml_tree.find('folder').text == prefix  # type: ignore

            xml_size = xml_tree.find('size')
            width, height, depth = [
                    int(xml_size.find(x).text)  # type: ignore
                    for x in ('width', 'height', 'depth')]  # type: ignore
            size_WHD = (width, height, depth)

            # Load objects
            objects: List[VOC_object] = []
            xml_objs = xml_tree.findall('object')
            for xml_obj in xml_objs:
                # // Getting the box
                bndbox = xml_obj.find('bndbox')
                # Make pixel indexes 0-based
                l = float(bndbox.find('xmin').text) - 1  # type: ignore
                t_ = float(bndbox.find('ymin').text) - 1  # type: ignore
                r = float(bndbox.find('xmax').text) - 1  # type: ignore
                d = float(bndbox.find('ymax').text) - 1  # type: ignore
                box = np.r_[l, t_, r, d]

                # cls = self._class_to_ind[obj.find('name').text.lower().strip()]
                objects.append(VOC_object(
                    name=xml_obj.find('name').text,  # type: ignore
                    pose=xml_obj.find('pose').text,  # type: ignore
                    difficult=xml_obj.find('difficult').text == '1',  # type: ignore
                    truncated=xml_obj.find('truncated').text == '1',  # type: ignore
                    box=box))

            annotations[index] = VOC_image_annotation(
                    filename=filename,
                    filepath=filepath,
                    size_WHD=size_WHD,
                    objects=objects)
        return annotations

    def precompute_to_folder(self, fold):
        annotations_per_split = small.stash2(
                fold/'get_annotations_per_split.pkl')(
                        self.get_annotations_per_split)
        light_stats = {
                'annotations_per_split': annotations_per_split}
        small.save_pkl(fold/'light_stats.pkl', light_stats)

    def populate_from_folder(self, fold):
        fold = Path(fold)
        light_stats = small.load_pkl(fold/'light_stats.pkl')
        self.annotations_per_split = light_stats['annotations_per_split']


class DatasetVOC2007(DatasetVOC):
    def __init__(self):
        self.root_path = get_dataset_path('detection/VOC2007/VOCdevkit')

    def get_annotations_per_split(self):
        annotations_per_split = {}
        # Reading XML
        for split in ['train', 'trainval', 'val', 'test']:
            log.info('Loading splits {}'.format(split))
            image_index = self._get_image_index_for_subset('VOC2007', split)
            annotations: "OrderedDict[str, VOC_image_annotation]" = \
                    self._get_annotations_from_index('VOC2007', image_index)
            annotations_per_split[split] = annotations
        return annotations_per_split


class DatasetVOC2012(DatasetVOC):
    def __init__(self):
        self.root_path = get_dataset_path('detection/VOC2012')

    def get_annotations_per_split(self):
        annotations_per_split = {}
        # Reading XML
        for split in ['train', 'trainval', 'val']:
            log.info('Loading splits {}'.format(split))
            image_index = self._get_image_index_for_subset('VOC2012', split)
            annotations: "OrderedDict[str, VOC_image_annotation]" = \
                    self._get_annotations_from_index('VOC2012', image_index)
            annotations_per_split[split] = annotations
        return annotations_per_split
