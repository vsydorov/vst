import logging
import math
import numpy as np
from typing import (  # NOQA
        Tuple, TypedDict, NewType, Dict, List, TypeVar)

from thes.data.dataset.external import (
        Dataset_daly_ocv, Vid_daly, Action_name_daly, F0)

log = logging.getLogger(__name__)

T = TypeVar('T')
DALY_wein_tube_index = Tuple[Vid_daly, int, int]
DALY_gt_tube_index = Tuple[Vid_daly, Action_name_daly, int]


class Base_frametube(TypedDict):
    frame_inds: np.ndarray  # 0-based ocv indices
    boxes: np.ndarray  # LTRD, absolute scale


class DALY_wein_tube(Base_frametube):
    hscores: np.ndarray  # human detector
    iscores: np.ndarray  # instance detector


class DALY_gt_tube(Base_frametube):
    times: np.ndarray
    start_time: float
    end_time: float
    start_frame: F0
    end_frame: F0


class Frametube(Base_frametube):
    # frame_inds not guaranteed to include
    # all frames in [start_frame, end_frame]
    start_frame: F0
    end_frame: F0


class Sframetube(Frametube):
    score: float


T_tube = TypeVar('T_tube', Frametube, Sframetube)
V_dict = Dict[Vid_daly, List[T_tube]]
AV_dict = Dict[Action_name_daly, Dict[Vid_daly, List[T_tube]]]


Objaction_dets = TypedDict('Objaction_dets', {
    'pred_boxes': np.ndarray,
    'scores': np.ndarray,
    'pred_classes': np.ndarray  # strings
})


def convert_dwein_tube(tube: DALY_wein_tube) -> Frametube:
    frame_inds = tube['frame_inds']
    ftube: Frametube = {
        'frame_inds': frame_inds,
        'boxes': tube['boxes'],
        'start_frame': frame_inds.min(),
        'end_frame': frame_inds.max()}
    return ftube


def get_daly_gt_tubes(
        dataset: Dataset_daly_ocv
            ) -> Dict[DALY_gt_tube_index, DALY_gt_tube]:
    dgt_tubes: Dict[DALY_gt_tube_index, DALY_gt_tube] = {}
    for vid, v in dataset.videos_ocv.items():
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                frame_inds = []
                times = []
                boxes = []
                for keyframe in instance['keyframes']:
                    frame_inds.append(keyframe['frame'])
                    times.append(keyframe['time'])
                    boxes.append(keyframe['bbox_abs'])
                tube: DALY_gt_tube = {
                    'start_time': instance['beginTime'],
                    'end_time': instance['endTime'],
                    'start_frame': instance['start_frame'],
                    'end_frame': instance['end_frame'],
                    'frame_inds': np.array(frame_inds),
                    'times': np.array(times),
                    'boxes': np.array(boxes)}
                dgt_tubes[(vid, action_name, ins_ind)] = tube
    return dgt_tubes


def convert_dgt_tubes(
        dgt_tubes: Dict[DALY_gt_tube_index, DALY_gt_tube]
            ) -> AV_dict[Frametube]:
    """
    - Accesses opencv inferred fps to estimate frame limits
    """
    av_gt_tubes: AV_dict[Frametube] = {}
    for dgt_index, tube in dgt_tubes.items():
        vid, action_name, ins_ind = dgt_index
        ftube: Frametube = {
            'frame_inds': tube['frame_inds'],
            'boxes': tube['boxes'],
            'start_frame': tube['start_frame'],
            'end_frame': tube['end_frame']}
        (av_gt_tubes.setdefault(action_name, {})
                .setdefault(vid, []).append(ftube))
    return av_gt_tubes


def av_filter_split(
        avx: AV_dict[T],
        split_vids: List[Vid_daly]) -> AV_dict[T]:
    favx = {}
    for a, vx in avx.items():
        favx[a] = {v: x for v, x in vx.items() if v in split_vids}
    return favx


def dtindex_filter_split(
        dtindex_dict: Dict[DALY_wein_tube_index, T],
        split_vids: List[Vid_daly]
        ) -> Dict[DALY_wein_tube_index, T]:
    dwt_index_fdict = {}
    for dwt_index, v in dtindex_dict.items():
        (vid, bunch_id, tube_id) = dwt_index
        if vid in split_vids:
            dwt_index_fdict[dwt_index] = v
    return dwt_index_fdict


def av_stubes_above_score(
        av_stubes: AV_dict[Sframetube],
        score: float
        ) -> AV_dict[Sframetube]:
    av_stubes_good: AV_dict[Sframetube] = {}
    for action_name, v_stubes in av_stubes.items():
        for vid, tubes in v_stubes.items():
            tubes_good = [t for t in tubes if t['score'] > score]
            (av_stubes_good
                    .setdefault(action_name, {})[vid]) = tubes_good
    return av_stubes_good
