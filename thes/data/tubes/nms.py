import logging
import numpy as np
from tqdm import tqdm
from typing import (  # NOQA
    Dict, List, Tuple, TypeVar, Set, Optional, Callable,
    TypedDict, NewType, NamedTuple, Sequence, Literal, cast)

from vsydorov_tools import small

from thes.data.tubes.types import (
        Frametube_scored, T_dwein_scored, Vid_daly, V_dict, AV_dict)
from thes.data.dataset.external import (
        Action_name_daly,)
from thes.data.tubes.routines import (
        temporal_ious_where_positive, spatial_tube_iou_v3)


log = logging.getLogger(__name__)


TV = TypeVar('TV')
TV_frametube_scored_co = \
    TypeVar('TV_frametube_scored_co', bound=Frametube_scored)


def nms_over_custom_elements(
        element_list: List[TV],
        overlaps_func: Callable[[TV, Sequence[TV]], List[float]],
        score_func: Callable[[TV], float],
        thresh: float,
        ) -> List[TV]:
    scores = [score_func(e) for e in element_list]
    sorted_ids = np.argsort(scores)[::-1]  # In decreasing order
    sorted_candidates = [element_list[i] for i in sorted_ids]
    results = []
    while len(sorted_candidates):
        taken = sorted_candidates.pop(0)
        results.append(taken)
        overlaps = overlaps_func(taken, sorted_candidates)
        sorted_candidates = [
                c for c, o in zip(sorted_candidates, overlaps) if o < thresh]
    return results


def spatiotemp_tube_iou_1N(
        x: Frametube_scored,
        ys: Sequence[Frametube_scored]
        ) -> np.ndarray:
    """
    Spatiotemporal IOUs: x tube with every y tube
    """
    y_frange = np.array([(y['start_frame'], y['end_frame']) for y in ys])
    ptious, pids = temporal_ious_where_positive(
            x['start_frame'], x['end_frame'], y_frange)
    st_overlaps = np.zeros(len(ys))
    if len(pids):
        pys = [ys[pid] for pid in pids]
        pmious = [spatial_tube_iou_v3(y, x) for y in pys]
        st_overlaps[pids] = ptious * pmious
    return st_overlaps


def compute_nms_for_stubes(
        stubes: List[TV_frametube_scored_co], thresh: float):
    return nms_over_custom_elements(
            stubes, spatiotemp_tube_iou_1N, lambda x: x['score'], thresh)


def compute_nms_for_v_stubes(
        v_stubes: V_dict[TV_frametube_scored_co],
        thresh: float,
        verbose_nms: bool) -> V_dict[TV_frametube_scored_co]:
    v_stubes_nms = {}
    for vid, stubes in tqdm(v_stubes.items(),
            desc='nms', disable=not verbose_nms):
        nmsed_stubes = compute_nms_for_stubes(stubes, thresh)
        v_stubes_nms[vid] = nmsed_stubes
    return v_stubes_nms


def computecache_nms_for_av_stubes(
        av_stubes: AV_dict[TV_frametube_scored_co],
        thresh: float,
        nms_folder) -> AV_dict[TV_frametube_scored_co]:
    av_stubes_nms = {}
    for a, v_stubes in av_stubes.items():
        nmsed_stubes_v = small.stash2(
            nms_folder/f'scored_tubes_nms_{thresh:.2f}_at_{a}_v2.pkl')(
            compute_nms_for_v_stubes,
            v_stubes, thresh, True)
        av_stubes_nms[a] = nmsed_stubes_v
    return av_stubes_nms


def compute_nms_for_av_stubes(
        av_stubes: AV_dict[TV_frametube_scored_co],
        thresh: float,
        verbose_nms: bool = False,
        ) -> AV_dict[TV_frametube_scored_co]:
    av_stubes_nms = {}
    for a, v_stubes in av_stubes.items():
        av_stubes_nms[a] = compute_nms_for_v_stubes(
                v_stubes, thresh, verbose_nms)
    return av_stubes_nms
