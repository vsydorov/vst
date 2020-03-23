from typing import (
    Dict, List, Tuple, TypeVar, Literal,
    Callable, TypedDict, NewType, NamedTuple)
from thes.data.tubes.types import (Frametube, Sframetube, V_dict, AV_dict)
import numpy as np

from thes.data.dataset.external import (
        DALY_vid, DALY_action_name)


class AP_fgt_framebox(TypedDict):
    ind: Tuple[DALY_vid, int, int]  # vid, frame, anno_id
    obj: np.ndarray  # LTRD box
    diff: bool


class AP_fdet_framebox(TypedDict):
    ind: Tuple[DALY_vid, int, int]  # vid, frame, det_id
    obj: np.ndarray  # LTRD box
    score: float


class Recall_coverage(TypedDict):
    max_spatial: float
    max_spatiotemp: float


class Options_tube_ap(TypedDict):
    iou_thresh: float
    spatiotemporal: bool
    use_07_metric: bool
    use_diff: bool


class AP_fgt_tube(TypedDict):
    ind: Tuple[str, int]
    obj: Frametube
    diff: bool


class AP_fdet_tube(TypedDict):
    ind: Tuple[str, int]
    obj: Frametube
    score: float


Stats_daly_ap = TypedDict('Stats_daly_ap', {
    'flat_annotations': List[AP_fgt_tube],
    'flat_detections': List[AP_fdet_tube],
    'detection_matched': np.ndarray,
    'gt_already_matched': np.ndarray,
    'possible_matches': List[Dict[int, float]],
    'iou_coverages_per_detection_ind': Dict[int, List[float]],
    'detection_matched_to_which_gt': np.ndarray,
    'sorted_inds': np.ndarray,
    'fp': np.ndarray,
    'tp': np.ndarray,
    'npos': int,
    'rec': np.ndarray,
    'prec': np.ndarray,
    'ap': float
})
