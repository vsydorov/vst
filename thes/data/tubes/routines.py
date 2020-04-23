import logging
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
from typing import (  # NOQA
    Dict, List, Tuple, TypeVar, Set, Optional, Callable,
    TypedDict, NewType, NamedTuple, Sequence, Literal, cast)
from thes.tools import snippets
from thes.data.dataset.external import (
        Dataset_daly_ocv, Vid_daly,
        Action_name_daly)

from thes.data.tubes.types import (
    Sframetube, Frametube, Base_frametube,
    DALY_wein_tube_index, Objaction_dets,
    V_dict, AV_dict, DALY_wein_tube, DALY_gt_tube,
    DALY_gt_tube_index, I_weingroup)

from vsydorov_tools import small

log = logging.getLogger(__name__)


T = TypeVar('T')


def filter_tube_keyframes_only_gt_v2(
        tubes: Dict[DALY_wein_tube_index, Frametube],
        av_gttubes: AV_dict[Frametube],
        keep_temporal: bool,
            ) -> Dict[DALY_wein_tube_index, Frametube]:
    """
    Filter "tubes" to contain only those frames,
    which are present in the DALY GT annotations
    """

    # Query good inds per vid
    gtinds: Dict[Vid_daly, Set[int]] = {}
    for action_name, v_gttubes in av_gttubes.items():
        for vid, gttubes in v_gttubes.items():
            for t in gttubes:
                gtinds[vid] = gtinds.get(vid, set()) | set(t['frame_inds'])

    # Filter tubes to only gt keyframes
    ftubes: Dict[DALY_wein_tube_index, Frametube] = {}
    for dwt_index, v in tqdm(tubes.items(), 'filter_tubes'):
        (vid, bunch_id, tube_id) = dwt_index
        good_inds: List[int] = list(gtinds.get(vid, set()))
        intersecting_inds, comm1, comm2 = \
            np.intersect1d(v['frame_inds'], good_inds, return_indices=True)
        if len(intersecting_inds):
            frame_inds = v['frame_inds'][comm1]
            boxes = v['boxes'][comm1]
            if keep_temporal:
                start_frame = v['start_frame']
                end_frame = v['end_frame']
            else:
                start_frame = np.min(frame_inds)
                end_frame = np.max(frame_inds)
            v_intersect: Frametube = {
                'index': v['index'],
                'frame_inds': frame_inds,
                'boxes': boxes,
                'start_frame': start_frame,
                'end_frame': end_frame}
            ftubes[dwt_index] = v_intersect
    return ftubes


def _barea(box):
    return np.prod(box[2:] - box[:2])


def _bareas(boxes):
    return np.prod(boxes[..., 2:] - boxes[..., :2], axis=1)


def _inter_areas(boxes1, boxes2):
    inter = np.c_[
        np.maximum(boxes1[..., :2], boxes2[..., :2]),
        np.minimum(boxes1[..., 2:], boxes2[..., 2:])]
    inter_subs = inter[..., 2:] - inter[..., :2]
    inter_areas = np.prod(inter_subs, axis=1)
    inter_areas[(inter_subs < 0).any(axis=1)] = 0.0
    return inter_areas


def numpy_iou_11(box1, box2):
    assert box1.shape == (4,)
    assert box2.shape == (4,)
    # Computing IOU
    inter = np.r_[
        np.maximum(box1[:2], box2[:2]),
        np.minimum(box1[2:], box2[2:])]
    if np.any(inter[:2] >= inter[2:]):
        iou = 0.0
    else:
        inter_area = _barea(inter)
        box1_area = _barea(box1)
        box2_area = _barea(box2)
        union_area = box1_area + box2_area - inter_area
        iou = inter_area/union_area
    return iou


def numpy_inner_overlap_N1(boxes1, box2):
    assert len(boxes1.shape) == 2
    assert boxes1.shape[-1] == 4
    assert box2.shape == (4,)
    inter_areas = _inter_areas(boxes1, box2)
    boxes1_areas = _bareas(boxes1)
    ioverlaps = inter_areas / boxes1_areas
    return ioverlaps


def numpy_iou_N1(boxes1, box2):
    assert len(boxes1.shape) == 2
    assert boxes1.shape[-1] == 4
    assert box2.shape == (4,)
    inter_areas = _inter_areas(boxes1, box2)
    boxes1_areas = _bareas(boxes1)
    box2_area = _barea(box2)
    union_areas = boxes1_areas + box2_area - inter_areas
    ious = inter_areas / union_areas
    return ious


def numpy_iou_NN(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    assert len(boxes1.shape) == 2
    assert boxes1.shape[-1] == 4
    inter_areas = _inter_areas(boxes1, boxes2)
    boxes1_areas = _bareas(boxes1)
    boxes2_areas = _bareas(boxes2)
    union_areas = boxes1_areas + boxes2_areas - inter_areas
    ious = inter_areas / union_areas
    return ious


def nms_over_custom_elements(
        element_list: List[T],
        overlaps_func: Callable[[T, Sequence[T]], List[float]],
        score_func: Callable[[T], float],
        thresh: float,
        ) -> List[T]:
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


def temporal_IOU(
        b1, e1, b2, e2):
    begin = max(b1, b2)
    end = min(e1, e2)
    inter = end-begin+1
    if inter <= 0:
        return 0.0
    else:
        union = (e1 - b1 + 1) + (e2 - b2 + 1) - inter
        return inter/union


def spatial_tube_iou_v3(
        tube1: Base_frametube,
        tube2: Base_frametube,
        ) -> float:
    """
    Compute avg iou over matching keyframes
    """
    ii, c1, c2 = np.intersect1d(
            tube1['frame_inds'], tube2['frame_inds'],
            assume_unique=True, return_indices=True)
    if len(ii):
        c1_boxes = tube1['boxes'][c1]
        c2_boxes = tube2['boxes'][c2]
        ious = numpy_iou_NN(c1_boxes, c2_boxes)
        miou = np.mean(ious)
    else:
        miou = np.nan
    return miou


def temporal_ious_where_positive(x_bf, x_ef, y_frange):
    """
    Temporal ious between inter X and multiple Y inters
    Inputs:
        x_bg, x_ef - temporal range of input
    Returns 2 np.ndarrays:
        pids: indices of ytubes with >0 temporal iou
        ptious: >0 temporal ious
    """
    if len(y_frange) == 0:
        pids = np.array([], dtype=np.int)
        ptious = np.array([])
        return ptious, pids
    ibegin = np.maximum(y_frange[:, 0], x_bf)
    iend = np.minimum(y_frange[:, 1], x_ef)
    temporal_intersections = iend-ibegin+1
    pids = np.where(temporal_intersections > 0)[0]
    if len(pids) == 0:
        ptious = np.array([])
    else:
        ptemp_inters = temporal_intersections[pids]
        p_bfs, p_efs = y_frange[pids].T
        ptemp_unions = (x_ef - x_bf + 1) + (p_efs - p_bfs + 1) - ptemp_inters
        ptious = ptemp_inters/ptemp_unions
    return ptious, pids


def temporal_ious_NN(x_frange, y_frange):
    begin = np.maximum(x_frange[..., 0], y_frange[..., 0])
    end = np.minimum(x_frange[..., 1], y_frange[..., 1])
    inter = end - begin + 1
    inter[inter<0] = 0
    union = ((x_frange[..., 1] - x_frange[..., 0] + 1)
        + (y_frange[..., 1] - y_frange[..., 0] + 1)
        - inter)
    return inter/union


def spatiotemp_tube_iou_1N(
        x: Sframetube, ys: Sequence[Sframetube]) -> np.ndarray:
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


def compute_nms_for_stubes(stubes: List[Sframetube], thresh: float):
    return nms_over_custom_elements(
            stubes, spatiotemp_tube_iou_1N, lambda x: x['score'], thresh)


def compute_nms_for_v_stubes(
        v_stubes: V_dict[Sframetube],
        thresh: float,
        verbose_nms: bool) -> V_dict[Sframetube]:
    v_stubes_nms = {}
    for vid, stubes in tqdm(v_stubes.items(),
            desc='nms', disable=not verbose_nms):
        nmsed_stubes = compute_nms_for_stubes(stubes, thresh)
        v_stubes_nms[vid] = nmsed_stubes
    return v_stubes_nms


def computecache_nms_for_av_stubes(
        av_stubes: AV_dict[Sframetube],
        thresh: float,
        nms_folder) -> AV_dict[Sframetube]:
    av_stubes_nms = {}
    for a, v_stubes in av_stubes.items():
        nmsed_stubes_v = small.stash2(
            nms_folder/f'scored_tubes_nms_{thresh:.2f}_at_{a}_v2.pkl')(
            compute_nms_for_v_stubes,
            v_stubes, thresh, True)
        av_stubes_nms[a] = nmsed_stubes_v
    return av_stubes_nms


def compute_nms_for_av_stubes(
        av_stubes: AV_dict[Sframetube],
        thresh: float,
        verbose_nms: bool = False,
        ) -> AV_dict[Sframetube]:
    av_stubes_nms = {}
    for a, v_stubes in av_stubes.items():
        av_stubes_nms[a] = compute_nms_for_v_stubes(
                v_stubes, thresh, verbose_nms)
    return av_stubes_nms


def score_ftubes_via_objaction_overlap_aggregation(
        dataset: Dataset_daly_ocv,
        objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]],
        ftubes: Dict[DALY_wein_tube_index, Frametube],
        overlap_type: Literal['inner_overlap', 'iou'],
        overlap_cutoff: float,
        score_cutoff: float
        ) -> AV_dict[Sframetube]:
    """
    """
    # To every tube, find matching keyframes
    dwti_ascore: Dict[DALY_wein_tube_index, Dict[Action_name_daly, float]] = {}
    for dwt_index, tube in tqdm(ftubes.items(), 'match_keyframes'):
        (vid, bunch_id, tube_id) = dwt_index
        cls_scores: Dict[Action_name_daly, float] = {}
        for frame_ind, tube_box in zip(
                tube['frame_inds'], tube['boxes']):
            # In frame, match box to all objections
            odets: Optional[Objaction_dets] = \
                    objactions_vf.get(vid, {}).get(frame_ind)
            if odets is None:
                continue
            # Check score
            score_above = odets['scores'] > score_cutoff
            sa_boxes = odets['pred_boxes'][score_above]
            # Check overlap
            if overlap_type == 'iou':
                sa_overlaps = numpy_iou_N1(sa_boxes, tube_box)
            elif overlap_type == 'inner_overlap':
                sa_overlaps = numpy_inner_overlap_N1(sa_boxes, tube_box)
            else:
                raise RuntimeError()
            sa_overlap_above = sa_overlaps > overlap_cutoff
            sa_oa_scores = odets['scores'][score_above][sa_overlap_above]
            sa_oa_classes = odets['pred_classes'][score_above][sa_overlap_above]
            for score, cls in zip(sa_oa_scores, sa_oa_classes):
                cls_scores[cls] = cls_scores.get(cls, 0.0) + score
        dwti_ascore[dwt_index] = cls_scores

    # Score the ftubes, convert to av_dict
    av_stubes: AV_dict[Sframetube] = {}
    for dwt_index, tube in ftubes.items():
        (vid, bunch_id, tube_id) = dwt_index
        scores: Dict[Action_name_daly, float] = dwti_ascore[dwt_index]
        # Sum the perframe scores
        for action_name in dataset.action_names:
            score = scores.get(action_name, 0.0)
            stube = tube.copy()
            stube['score'] = score
            stube = cast(Sframetube, stube)
            (av_stubes
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes

def _get_df_daly_groundtruth(gt_tubes):
    dgt_frange_ = np.array([
        (gt_tube['start_frame'], gt_tube['end_frame'])
        for gt_tube in gt_tubes.values()])
    df_dgt = pd.DataFrame(dgt_frange_,
            pd.MultiIndex.from_tuples(
                list(gt_tubes.keys()), names=['vid', 'act', 'id']),
            columns=['start', 'end'],)
    return df_dgt

def _get_df_weingroup_range(dwein_tubes):
    dwt_frange_ = [
        (tube['frame_inds'].min(), tube['frame_inds'].max())
        for tube in dwein_tubes.values()]
    dwt_df = pd.DataFrame(dwt_frange_, pd.MultiIndex.from_tuples(
        dwein_tubes.keys(), names=['vid', 'gid', 'tid']),
        columns=['start', 'end'])

    dwt_df_grouped_ = dwt_df.groupby(level=[0, 1]).agg(lambda x: set(x))
    assert dwt_df_grouped_.applymap(
            lambda x: len(x) == 1).all().all(), \
            'All groups must have equal size'
    weingroup_range = dwt_df_grouped_.applymap(lambda x: list(x)[0])
    return weingroup_range


def get_weingroup_assignment(
        gt_tubes: Dict[DALY_gt_tube_index, Frametube],
        dwein_tubes: Dict[DALY_wein_tube_index, DALY_wein_tube],
        ) -> Tuple[
                List[Tuple[Vid_daly, int]],
                List[DALY_gt_tube_index]]:

    df_gt = _get_df_daly_groundtruth(gt_tubes)
    df_weingroup_range = _get_df_weingroup_range(dwein_tubes)

    wgi = []
    gti = []
    all_vids = df_gt.index.levels[0]
    for vid in all_vids:
        wg = df_weingroup_range.loc[vid].sort_values('start')
        gt = df_gt.loc[vid].sort_values('start')
        assert len(wg) == len(gt)
        ious = temporal_ious_NN(wg.to_numpy(), gt.to_numpy())
        assert (ious >= 0.75).all()
        for gid in wg.index:
            wgi.append((vid, gid))
        for (act, ind) in gt.index:
            gti.append((vid, act, ind))
    assert len(set(wgi)) == len(wgi) == len(set(gti)) == len(gti)
    return wgi, gti

def wein_coverage_stats(
        gt_tubes: Dict[DALY_gt_tube_index, DALY_gt_tube],
        dwein_tubes: Dict[DALY_wein_tube_index, DALY_wein_tube],
        ) -> Dict[Tuple[Vid_daly, int], DALY_gt_tube_index]:
    raise NotImplementedError()
    pass

# wgi_to_dgti = {}
# for (vid, gid), row in weingroup_range.iterrows():
#     dgt_vmask = dgt_indexes[:, 0] == vid
#     dgt_indexes_vmasked = dgt_indexes[dgt_vmask]
#     dgt_frange_vmasked = dgt_frange[dgt_vmask]
#     ptious, pids = temporal_ious_where_positive(
#             row['start'], row['end'], dgt_frange_vmasked)
#     assert len(ptious), 'must match something'
#     best_match_i_vmasked = pids[np.argmax(ptious)]
#     I = dgt_indexes_vmasked[best_match_i_vmasked].tolist()
#     dgt_index = (I[0], I[1], int(I[2]))
#     wgi_to_dgti[(vid, gid)] = dgt_index
# dgti_to_wgi: Dict[DALY_gt_tube_index, I_weingroup] = \
#         dict(zip(wgi_to_dgti.values(), wgi_to_dgti.keys()))

# # Experiment, of whether any windows intersect
# frame_counter = {}
# max_per_vid = df_dgt.groupby('vid').end.max()
# for vid, row in max_per_vid.iteritems():
#     frame_counter[vid] = np.zeros(int(row))
# for idx, row in df_dgt.iterrows():
#     b = int(row.start)
#     e = int(row.end)
#     frame_counter[idx[0]][b: e] += 1
# intersecting = {k: v.max() for k, v in frame_counter.items() if v.max() > 1}
