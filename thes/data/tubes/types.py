import logging
import numpy as np
from typing import (  # NOQA
        Tuple, TypedDict, NewType, Dict, List, TypeVar, Union, Any)

from vsydorov_tools import small

from thes.data.dataset.external import (
        Dataset_daly_ocv, Vid_daly, Action_name_daly,
        F0, Instance_flags_daly)

log = logging.getLogger(__name__)

# Index of DALY weinzaepful tube (as it was in py2 .pkl)
I_dwein = Tuple[Vid_daly, int, int]

# Index of DALY ground truth tube
I_dgt = Tuple[Vid_daly, Action_name_daly, int]

I_tube = Union[I_dwein, I_dgt]


class Base_frametube(TypedDict):
    frame_inds: np.ndarray  # 0-based ocv indices
    boxes: np.ndarray  # LTRD, absolute scale


class Mixin_score(TypedDict):
    score: float


class Tube_daly_wein_as_provided(Base_frametube):
    """DALY weinzaepfel tube as provided in py2 .pkl"""
    hscores: np.ndarray  # human detector
    iscores: np.ndarray  # instance detector


class Frametube(Base_frametube):
    # frame_inds not guaranteed to include
    # all frames in [start_frame, end_frame]
    start_frame: F0
    end_frame: F0


class Frametube_scored(Frametube):
    score: float


class T_dgt(Frametube):
    """DALY ground truth tube, adapted for our needs"""
    index: I_dgt
    flags: Instance_flags_daly
    times: np.ndarray
    start_time: float
    end_time: float

class T_dwein(Frametube):
    """DALY weinzaepfel tube, after conversion"""
    index: I_dwein


class T_dwein_scored(Frametube_scored):
    index: I_dwein


TV = TypeVar('TV')
TV_I = TypeVar('TV_I', I_dwein, I_dgt)


# Shorthand for tube groupins
V_dict = Dict[Vid_daly, List[TV]]
AV_dict = Dict[Action_name_daly, Dict[Vid_daly, List[TV]]]


Objaction_dets = TypedDict('Objaction_dets', {
    'pred_boxes': np.ndarray,
    'scores': np.ndarray,
    'pred_classes': np.ndarray  # strings
})


def _reconvert_dwein_tube(
        index: I_dwein,
        tube: Tube_daly_wein_as_provided) -> T_dwein:
    frame_inds = tube['frame_inds']
    tdwein: T_dwein = {
        'index': index,
        'frame_inds': frame_inds,
        'boxes': tube['boxes'],
        'start_frame': frame_inds.min(),
        'end_frame': frame_inds.max()}
    return tdwein


def loadconvert_tubes_dwein(path_dwein) -> Dict[I_dwein, T_dwein]:
    # Tubes faithfully loaded in experiments.data.load_wein_tubes
    tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided] = \
            small.load_pkl(path_dwein)
    # Convenience reconversion
    tubes_dwein = {k: _reconvert_dwein_tube(k, t)
            for k, t in tubes_dwein_prov.items()}
    return tubes_dwein


def get_daly_gt_tubes(
        dataset: Dataset_daly_ocv
        ) -> Dict[I_dgt, T_dgt]:
    dgt_tubes: Dict[I_dgt, T_dgt] = {}
    for vid, v in dataset.videos_ocv.items():
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                fl = instance['flags']
                frame_inds = []
                times = []
                boxes = []
                for keyframe in instance['keyframes']:
                    frame_inds.append(keyframe['frame'])
                    times.append(keyframe['time'])
                    boxes.append(keyframe['bbox_abs'])
                index: I_dgt = (vid, action_name, ins_ind)
                tube: T_dgt = {
                    'index': index,
                    'flags': fl,
                    'start_time': instance['beginTime'],
                    'end_time': instance['endTime'],
                    'start_frame': instance['start_frame'],
                    'end_frame': instance['end_frame'],
                    'frame_inds': np.array(frame_inds),
                    'times': np.array(times),
                    'boxes': np.array(boxes)}
                dgt_tubes[index] = tube
    return dgt_tubes


def push_into_avdict(dgt_tubes: Dict[I_dgt, TV]) -> AV_dict[TV]:
    avdict: AV_dict[TV] = {}
    for dgt_index, tube in dgt_tubes.items():
        vid, action_name, ins_ind = dgt_index
        (avdict.setdefault(action_name, {})
            .setdefault(vid, []).append(tube))
    return avdict


def av_filter_split(
        avx: AV_dict[TV],
        split_vids: List[Vid_daly]) -> AV_dict[TV]:
    favx = {}
    for a, vx in avx.items():
        favx[a] = {v: x for v, x in vx.items() if v in split_vids}
    return favx


def dtindex_filter_split(
        dtindex_dict: Dict[TV_I, TV],
        split_vids: List[Vid_daly]
        ) -> Dict[TV_I, TV]:
    dwt_index_fdict = {}
    for dwt_index, v in dtindex_dict.items():
        (vid, bunch_id, tube_id) = dwt_index
        if vid in split_vids:
            dwt_index_fdict[dwt_index] = v
    return dwt_index_fdict


def av_stubes_above_score(av_stubes, score: float):
    av_stubes_good: AV_dict[Any] = {}
    for action_name, v_stubes in av_stubes.items():
        for vid, tubes in v_stubes.items():
            tubes_good = [t for t in tubes if t['score'] > score]
            (av_stubes_good
                    .setdefault(action_name, {})[vid]) = tubes_good
    return av_stubes_good
