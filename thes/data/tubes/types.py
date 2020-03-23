import logging
import numpy as np
from typing import (
        Tuple, TypedDict, NewType, Dict, List, TypeVar)

from thes.data.dataset.external import (
        DatasetDALY, DALY_vid, DALY_action_name)

log = logging.getLogger(__name__)

T = TypeVar('T')
DALY_wein_tube_index = Tuple[DALY_vid, int, int]
DALY_gt_tube_index = Tuple[DALY_vid, DALY_action_name, int]
FrameNumber0 = NewType('FrameNumber0', int)  # 0-based frame number
Scores_per_frame = Dict[FrameNumber0, np.ndarray]


class Base_frametube(TypedDict):
    frame_inds: np.ndarray
    boxes: np.ndarray  # LTRD


class DALY_wein_tube(Base_frametube):
    hscores: np.ndarray  # human detector
    iscores: np.ndarray  # instance detector


class DALY_gt_tube(Base_frametube):
    times: np.ndarray
    start_time: float
    end_time: float


class Frametube(Base_frametube):
    # frame_inds not guaranteed to include
    # all frames in [start_frame, end_frame]
    start_frame: int
    end_frame: int


class Sframetube(Frametube):
    score: float


T_tube = TypeVar('T_tube', Frametube, Sframetube)
V_dict = Dict[DALY_vid, List[T_tube]]
AV_dict = Dict[DALY_action_name, Dict[DALY_vid, List[T_tube]]]


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
        dataset: DatasetDALY
            ) -> Dict[DALY_gt_tube_index, DALY_gt_tube]:
    # Daly GT tubes
    gt_tubes: Dict[DALY_gt_tube_index, DALY_gt_tube] = {}
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
                tube: DALY_gt_tube = {
                    'start_time': instance['beginTime'],
                    'end_time': instance['endTime'],
                    'frame_inds': np.array(frame_inds),
                    'times': np.array(times),
                    'boxes': np.array(boxes)}
                gt_tubes[(vid, action_name, ins_ind)] = tube
    return gt_tubes


def convert_dgt_tube(
        tube: DALY_gt_tube,
        ocv_video_fps: float) -> Frametube:
    start_frame = int(tube['start_time'] * ocv_video_fps)
    end_frame = int(tube['end_time'] * ocv_video_fps)
    ftube: Frametube = {
        'frame_inds': tube['frame_inds'],
        'boxes': tube['boxes'],
        'start_frame': start_frame,
        'end_frame': end_frame}
    return ftube


def convert_dgt_tubes_to_framebased(
        dataset: DatasetDALY,
        dgt_tubes: Dict[DALY_gt_tube_index, DALY_gt_tube]
            ) -> Dict[DALY_gt_tube_index, Frametube]:
    """
    - Accesses opencv inferred fps to estimate frame limits
    """
    ftubes: Dict[DALY_gt_tube_index, Frametube] = {}
    for dgt_index, dgt_tube in dgt_tubes.items():
        vid, action_name, ins_ind = dgt_index
        vmp4 = dataset.source_videos[vid]
        ocv_video_fps = vmp4['frames_reached']/vmp4['length_reached']
        ftubes[dgt_index] = convert_dgt_tube(dgt_tube, ocv_video_fps)
    return ftubes


def dtindex_filter_split(
        dtindex_dict: Dict[DALY_wein_tube_index, T],
        split_vids: List[DALY_vid]
        ) -> Dict[DALY_wein_tube_index, T]:
    dwt_index_fdict = {}
    for dwt_index, v in dtindex_dict.items():
        (vid, bunch_id, tube_id) = dwt_index
        if vid in split_vids:
            dwt_index_fdict[dwt_index] = v
    return dwt_index_fdict
