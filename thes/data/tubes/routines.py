"""
General tube routines that can function without a tight dataset connections
"""
import logging
import numpy as np
from tqdm import tqdm
from typing import (  # NOQA
    Dict, List, Tuple, TypeVar, Set, Optional, Callable,
    TypedDict, NewType, NamedTuple, Sequence, Literal, cast, Any)
from sklearn.metrics import (accuracy_score)

from thes.tools import snippets
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly, Action_name_daly)
from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt, AV_dict,
    Base_frametube, Objaction_dets, Box_connections_dwti)

log = logging.getLogger(__name__)


T = TypeVar('T')


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


def numpy_inner_overlap_NN(boxes1, boxes2):
    assert boxes1.shape == boxes2.shape
    assert len(boxes1.shape) == 2
    assert boxes1.shape[-1] == 4
    inter_areas = _inter_areas(boxes1, boxes2)
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


"""
OVERLAP SCORING

I use these guys to score tubes based on intersections with box detections
- Helps with transfering detection information to tubes
- Usecases:
  - Score DWEIN tubes via RCNN objdect detection overlap
  - Cheating evaluation of DWEIN tubes via keyframe-only predictions
"""


def create_kinda_objaction_struct(dataset, test_kfs, Y_conf_scores_sm):
    # // Creating kinda objaction structure
    # Group vid -> frame
    grouped_kfscores_vf: Dict[Vid_daly, Dict[int, Any]] = {}
    for kf, scores in zip(test_kfs, Y_conf_scores_sm):
        vid = kf['vid']
        frame0 = kf['frame0']
        pred_box = kf['bbox']
        (grouped_kfscores_vf
                .setdefault(vid, {})
                .setdefault(frame0, [])
                .append([pred_box, scores]))
    # fake objactions
    objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]] = {}
    for vid, grouped_kfscores_f in grouped_kfscores_vf.items():
        for frame_ind, gkfscores in grouped_kfscores_f.items():
            all_scores, all_boxes, all_classes = [], [], []
            for (box, scores) in gkfscores:
                all_boxes.append(np.tile(box, (len(scores), 1)))
                all_classes.append(np.array(dataset.action_names))
                all_scores.append(scores)
            all_scores_ = np.hstack(all_scores)
            all_classes_ = np.hstack(all_classes)
            all_boxes_ = np.vstack(all_boxes)
            detections = {
                    'pred_boxes': all_boxes_,
                    'scores': all_scores_,
                    'pred_classes': all_classes_}
            objactions_vf.setdefault(vid, {})[frame_ind] = detections
    return objactions_vf


def score_ftubes_via_objaction_overlap_aggregation(
        dataset: Dataset_daly_ocv,
        objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]],
        ftubes: Dict[I_dwein, T_dwein],
        overlap_type: Literal['inner_overlap', 'iou'],
        overlap_cutoff: float,
        score_cutoff: float,
        enable_tqdm: bool = True,
        ) -> AV_dict[T_dwein_scored]:
    """
    """
    # To every tube, find matching keyframes
    dwti_ascore: Dict[I_dwein, Dict[Action_name_daly, float]] = {}
    if enable_tqdm:
        pbar = tqdm(ftubes.items(), 'match_keyframes')
    else:
        pbar = ftubes.items()
    for dwt_index, tube in pbar:
        (vid, bunch_id, tube_id) = dwt_index
        cls_scores: Dict[Action_name_daly, float] = {}
        for frame_ind, tube_box in zip(
                tube['frame_inds'], tube['boxes']):
            # In frame, match box to all objections
            odets: Optional[Objaction_dets] = \
                    objactions_vf.get(vid, {}).get(frame_ind)
            if odets is None:
                continue

            # from vsydorov_tools import cv as vt_cv
            # import cv2
            # video_path = dataset.videos_ocv[vid]['path']
            # with vt_cv.video_capture_open(video_path) as vcap:
            #     fl_u8_bgr = vt_cv.video_sample(vcap, [frame_ind])[0]
            # Y = np.ascontiguousarray(fl_u8_bgr)
            # snippets.misc.cv_put_box_with_text(
            #         Y, tube_box, text='DWT_box')
            # # Highest scored odet
            # argmax = odets['scores'].argmax()
            # best_box = odets['pred_boxes'][argmax]
            # best_score = odets['scores'][argmax]
            # best_label = odets['pred_classes'][argmax]
            # label = f'{best_label} = {best_score:.2f}'
            # snippets.misc.cv_put_box_with_text(
            #         Y, best_box, text=label)
            # cv2.imshow("test", Y)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

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
            sa_oa_classes = \
                    odets['pred_classes'][score_above][sa_overlap_above]
            for score, cls in zip(sa_oa_scores, sa_oa_classes):
                cls_scores[cls] = cls_scores.get(cls, 0.0) + score
        dwti_ascore[dwt_index] = cls_scores

    # Score the ftubes, convert to av_dict
    av_stubes: AV_dict[T_dwein_scored] = {}
    for dwt_index, tube in ftubes.items():
        (vid, bunch_id, tube_id) = dwt_index
        scores: Dict[Action_name_daly, float] = dwti_ascore[dwt_index]
        # Sum the perframe scores
        for action_name in dataset.action_names:
            score = scores.get(action_name, 0.0)
            stube = tube.copy()
            stube['score'] = score
            stube = cast(T_dwein_scored, stube)
            (av_stubes
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes


"""
DGT, DWEIN rubbish

These functions are connected to DALY dataset and should probably be in a
separate module, but we assumed that connection is not inherent and can be
removed in the future

"""


# Box_connections code


def group_tubes_on_frame_level(
        tubes_dwein: Dict[I_dwein, T_dwein],
        frames_to_cover: Optional[Dict[Vid_daly, np.ndarray]]
        ) -> Dict[Tuple[Vid_daly, int], Box_connections_dwti]:
    """
    Given dictionary of tubes, group them on frame level
    - If "frames_to_cover" is passed - group only in those frames

    This is a simplified _prepare_ftube_box_computations
    """
    connections: Dict[Tuple[Vid_daly, int], Box_connections_dwti] = {}
    for dwt_index, tube in tubes_dwein.items():
        (vid, _, _) = dwt_index
        tube_finds = tube['frame_inds']
        if frames_to_cover is None:
            common_finds = tube_finds
            good_tube_boxes = tube['boxes']
        else:
            good_finds = frames_to_cover[vid]
            common_finds, comm1, comm2 = np.intersect1d(
                tube_finds, good_finds,
                assume_unique=True, return_indices=True)
            if len(common_finds) == 0:
                continue
            good_tube_boxes = tube['boxes'][comm1]
        for frame_ind, box in zip(common_finds, good_tube_boxes):
            frame_ind = int(frame_ind)
            c = connections.setdefault((vid, frame_ind),
                Box_connections_dwti(
                    vid=vid, frame_ind=frame_ind,
                    dwti_sources=[], boxes=[]))
            c['dwti_sources'].append(dwt_index)
            c['boxes'].append(box)
    return connections


def perform_connections_split(connections_f, cc, ct, fancy=True):
    ckeys = list(connections_f.keys())
    if fancy:
        weights_dict = {k: len(v['boxes'])
                for k, v in connections_f.items()}
        weights = np.array(list(weights_dict.values()))
    else:
        weights = np.ones(len(connections_f))
    ii_ckeys_split = snippets.weighted_array_split(
            np.arange(len(ckeys)), weights, ct)
    ckeys_split = [[ckeys[i] for i in ii] for ii in ii_ckeys_split]
    ktw = dict(zip(ckeys, weights))
    weights_split = []
    for ckeys_ in ckeys_split:
        weight = np.sum([ktw[ckey] for ckey in ckeys_])
        weights_split.append(weight)
    chunk_ckeys = ckeys_split[cc]
    log.info(f'Quick split stats [{cc,ct=}]: '
        'Frames(boxes): {}({}) -> {}({})'.format(
            len(ckeys), np.sum(weights),
            len(chunk_ckeys), weights_split[cc]))
    chunk_connections_f = {k: connections_f[k] for k in chunk_ckeys}
    return chunk_connections_f


# Overlap matching code


def record_overlaps(
        tubes_dgt: Dict[I_dgt, T_dgt],
        tubes_dwein: Dict[I_dwein, T_dwein]):
    overlap_hits: Dict[I_dwein, List[
        Tuple[Action_name_daly, Tuple[float, float, float]]]] = {}
    for dgt_index, gt_tube in tubes_dgt.items():
        vid, action_name, ins_id = dgt_index
        matched_dwt_vids = {k: v for k, v in tubes_dwein.items()
                if k[0] == vid}
        dwt_vid_keys = list(matched_dwt_vids.keys())
        dwt_vid_values = list(matched_dwt_vids.values())
        dwt_frange = np.array([
            (x['start_frame'], x['end_frame']) for x in dwt_vid_values])
        # Temporal
        t_ious, pids = temporal_ious_where_positive(
            gt_tube['start_frame'], gt_tube['end_frame'], dwt_frange)
        # Spatial (where temporal >0)
        dwt_intersect = [dwt_vid_values[pid] for pid in pids]
        sp_mious = [spatial_tube_iou_v3(p, gt_tube)
                for p in dwt_intersect]
        for p, t_iou, sp_miou in zip(pids, t_ious, sp_mious):
            st_iou = t_iou * sp_miou
            if st_iou > 0:
                dwt_vid = dwt_vid_keys[p]
                overlap_hits.setdefault(dwt_vid, []).append(
                        (action_name, (t_iou, sp_miou, st_iou)))
    best_ious = {}
    for k, v in overlap_hits.items():
        vsorted_last = sorted(v, key=lambda x: x[1][0])[-1]
        action_name = vsorted_last[0]
        st_miou = vsorted_last[1][2]
        best_ious[k] = (action_name, st_miou)
    return best_ious


def get_best_overlaping_dgt_per_dwein(
        tubes_dgt: Dict[I_dgt, T_dgt],
        tubes_dwein: Dict[I_dwein, T_dwein]
        ) -> Dict[I_dwein, Tuple[I_dgt, float]]:
    """
    - For each DWEIN, find matching DGT tubes
    - Select the DGT with best avg spatial overlap
    """
    # Record overlap hits
    overlap_hits: Dict[I_dwein, Dict[I_dgt, float]] = {}
    for dgt_index, gt_tube in tubes_dgt.items():
        vid, action_name, ins_id = dgt_index
        matched_dwt_vids = {k: v for k, v in tubes_dwein.items()
                if k[0] == vid}
        dwt_vid_keys = list(matched_dwt_vids.keys())
        dwt_vid_values = list(matched_dwt_vids.values())
        dwt_frange = np.array([
            (x['start_frame'], x['end_frame']) for x in dwt_vid_values])
        # Temporal
        t_ious, pids = temporal_ious_where_positive(
            gt_tube['start_frame'], gt_tube['end_frame'], dwt_frange)
        # Spatial (where temporal >0)
        dwt_intersect = [dwt_vid_values[pid] for pid in pids]
        sp_mious = [spatial_tube_iou_v3(p, gt_tube)
                for p in dwt_intersect]
        for p, t_iou, sp_miou in zip(pids, t_ious, sp_mious):
            st_iou = t_iou * sp_miou
            if st_iou > 0:
                dwt_vid = dwt_vid_keys[p]
                overlap_hits.setdefault(dwt_vid, {})[dgt_index] = sp_miou
    # Return only best hit
    best_hits: Dict[I_dwein, Tuple[I_dgt, float]] = {}
    for dwt_index, hits in overlap_hits.items():
        hits_list = list(hits.items())
        hits_list_sorted = sorted(hits_list, key=lambda x: x[1])
        dgt_index, overlap = hits_list_sorted[-1]
        best_hits[dwt_index] = (dgt_index, overlap)
    return best_hits

def get_dwein_overlaps_per_dgt(
        tubes_dgt: Dict[I_dgt, T_dgt],
        tubes_dwein: Dict[I_dwein, T_dwein]
        ) -> Dict[I_dgt, Dict[I_dwein, float]]:
    """
    For each DGT, match DWEINs, sort by spatial overlaps
    """
    matched_dwts: Dict[I_dgt, Dict[I_dwein, float]] = {}  # spatial mious
    for dgt_index, gt_tube in tubes_dgt.items():
        vid, action_name, ins_id = dgt_index
        matched_dwt_vids = {k: v for k, v in tubes_dwein.items()
                if k[0] == vid}
        dwt_vid_keys = list(matched_dwt_vids.keys())
        dwt_vid_values = list(matched_dwt_vids.values())
        dwt_frange = np.array([
            (x['start_frame'], x['end_frame']) for x in dwt_vid_values])
        # Temporal
        t_ious, pids = temporal_ious_where_positive(
                gt_tube['start_frame'], gt_tube['end_frame'], dwt_frange)
        # Spatial (where temporal >0)
        dwt_intersect = [dwt_vid_values[pid] for pid in pids]
        sp_mious = [spatial_tube_iou_v3(p, gt_tube)
                for p in dwt_intersect]
        matched: Dict[I_dwein, float] = {}
        for p, t_iou, sp_miou in zip(pids, t_ious, sp_mious):
            st_iou = t_iou * sp_miou
            if st_iou > 0:
                dwt_vid = dwt_vid_keys[p]
                matched[dwt_vid] = sp_miou
        # Sort by overlap
        matched = dict(sorted(matched.items(),
            key=lambda x: x[1], reverse=True))
        matched_dwts[dgt_index] = matched
    return matched_dwts


# Trying to create synthetic labels based on overlaps


def select_fg_bg_tubes(
        matched_dwts: Dict[I_dgt, Dict[I_dwein, float]],
        top_n=None):
    # / Create FG and BG tubes
    fg_meta: Dict[I_dwein, Dict] = {}
    for dgti, matched in matched_dwts.items():
        overlaps = np.array([m for m in matched.values()])
        # take "top_n" matches with overlap > 0.5
        ii = np.flatnonzero(overlaps > 0.5)
        if top_n is not None:
            ii = ii[:top_n]
        for i in ii:
            dwti, overlap = list(matched.items())[i]
            fg_meta[dwti] = {'kind': 'fg', 'dgti': dgti, 'overlap': overlap}
    bg_meta: Dict[I_dwein, Dict] = {}
    for dgti, matched in matched_dwts.items():
        overlaps = np.array([m for m in matched.values()])
        # take "top_n" matches with 0 < overlap <= 0.3
        ii = np.flatnonzero((overlaps <= 0.3) & (overlaps > 0))
        if top_n is not None:
            ii = ii[:top_n]
        for i in ii:
            dwti, overlap = list(matched.items())[i]
            bg_meta[dwti] = {'kind': 'bg', 'dgti': dgti, 'overlap': overlap}
    return fg_meta, bg_meta


def create_synthetic_tube_labels(
        tubes_dwein: Dict[I_dwein, T_dwein],
        best_ious) -> Dict[I_dwein, str]:
    # Assign to classes
    POS_THRESH = 0.5
    HN_THRESH = 0.3
    labels: Dict[I_dwein, str] = {}
    for dwt_index in tubes_dwein.keys():
        label = 'background'
        if dwt_index in best_ious:
            best_cls, best_iou = best_ious[dwt_index]
            if best_iou > POS_THRESH:
                label = best_cls
            elif 0 < best_iou < HN_THRESH:
                label = 'background_hard'
            else:
                label = 'none'
        labels[dwt_index] = label
    return labels


def qload_synthetic_tube_labels(
        tubes_dgt: Dict[I_dgt, T_dgt],
        tubes_dwein: Dict[I_dwein, T_dwein],
        dataset: Dataset_daly_ocv
        ) -> Tuple[List[str], Dict[I_dwein, int]]:
    # / Divide trainval tubes into classes (intersection with GT tubes)
    best_ious = record_overlaps(tubes_dgt, tubes_dwein)
    labels_train: Dict[I_dwein, str] = create_synthetic_tube_labels(
            tubes_dwein, best_ious)
    # / Create classification dataset
    cls_labels = cast(List[str], dataset.action_names) + ['background']
    dwti_to_label: Dict[I_dwein, int] = {}
    for dwti, label in labels_train.items():
        if label == 'none':
            continue
        elif label in ('background', 'background_hard'):
            ilabel = len(dataset.action_names)
        elif label in dataset.action_names:
            alabel = cast(Action_name_daly, label)
            ilabel = dataset.action_names.index(alabel)
        else:
            raise RuntimeError()
        dwti_to_label[dwti] = ilabel
    return cls_labels, dwti_to_label


def quick_assign_scores_to_dwein_tubes(
        tubes_dwein: Dict[I_dwein, T_dwein],
        tube_softmaxes: Dict[I_dwein, np.ndarray],
        dataset: Dataset_daly_ocv
        ) -> AV_dict[T_dwein_scored]:
    """
    Softmaxes should correspond to dataset.action_names
    """
    # Assert absence of background cls
    x = next(iter(tube_softmaxes.values()))
    assert x.shape[-1] == 10

    av_stubes: AV_dict[T_dwein_scored] = {}
    for dwt_index, tube in tubes_dwein.items():
        softmaxes = tube_softmaxes[dwt_index]
        scores = softmaxes.mean(axis=0)
        (vid, bunch_id, tube_id) = dwt_index
        for action_name, score in zip(dataset.action_names, scores):
            stube = cast(T_dwein_scored, tube.copy())
            stube['score'] = score
            stube = cast(T_dwein_scored, stube)
            (av_stubes
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes


def compute_flattube_syntlabel_acc(
        tube_softmaxes: Dict[I_dwein, np.ndarray],
        dwti_to_label: Dict[I_dwein, int]) -> float:
    """
    Compute synthetic per-frame accuracy over dwein tubes
    """
    # Assert presence of background cls
    x = next(iter(tube_softmaxes.values()))
    assert x.shape[-1] == 11

    flat_sm_ = []
    flat_label_ = []
    for dwti, label in dwti_to_label.items():
        softmaxes = tube_softmaxes[dwti]
        flat_sm_.append(softmaxes)
        flat_label_.append(np.repeat(label, len(softmaxes)))
    flat_sm = np.vstack(flat_sm_)
    flat_label = np.hstack(flat_label_)
    return accuracy_score(flat_label, flat_sm.argmax(axis=1))
