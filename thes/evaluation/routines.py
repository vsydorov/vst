import warnings
import logging
import pandas as pd
import numpy as np
from typing import (
    Dict, List, Tuple, TypeVar, Literal,
    Callable, TypedDict, NewType, NamedTuple,
    Any)
from collections import namedtuple
from pathlib import Path
from abc import abstractmethod, ABC

from thes.tools import snippets
from thes.data.dataset.external import (
        DALY_vid, DALY_action_name)
from thes.data.tubes.routines import (
        numpy_iou, temporal_IOU,
        spatial_tube_iou_v2,
)
from thes.data.tubes.types import (Frametube, Sframetube, V_dict, AV_dict)
from thes.evaluation.types import (
        Recall_coverage,)
from thes.evaluation.ap import (
        AP_fgt_framebox, AP_fdet_framebox,
        AP_fgt_tube, AP_fdet_tube,
        AP_tube_computer, AP_framebox_computer)


log = logging.getLogger(__name__)


def _compute_daly_recall_coverage(
        gt_tube: Frametube,
        proposals: List[Sframetube],
        proposals_frange: np.ndarray,
            ) -> Recall_coverage:
    if len(proposals) == 0:
        return {'max_spatial': np.nan,
                'max_spatiotemp': np.nan}
    assert len(proposals_frange.shape) == 2
    assert proposals_frange.shape[1] == 2
    # Extract min/max frames
    gt_bf = gt_tube['start_frame']
    gt_ef = gt_tube['end_frame']
    # Computing temporal intersection
    ibegin = np.maximum(proposals_frange[:, 0], gt_bf)
    iend = np.minimum(proposals_frange[:, 1], gt_ef)
    temporal_intersections = iend-ibegin+1
    # Prepare iou values
    spatial_mious = np.zeros(len(proposals))
    temp_ious = np.zeros(len(proposals))
    # Loop over proposal tubes that have at least some temporal
    # intersection
    for pid in np.where(temporal_intersections > 0)[0]:
        proposal_tube: Sframetube = proposals[pid]
        spatial_miou, spatial_ious = \
                spatial_tube_iou_v2(proposal_tube, gt_tube)
        # Temporal IOU
        temp_inter = temporal_intersections[pid]
        p_bf, p_ef = proposals_frange[pid]
        temp_union = (gt_ef - gt_bf + 1) + (p_ef - p_bf + 1) - temp_inter
        temp_iou = temp_inter/temp_union
        # Report upstairs
        spatial_mious[pid] = spatial_miou
        temp_ious[pid] = temp_iou
    st_mious = spatial_mious * temp_ious
    # N_viable = len(np.where(possible_spatial_temp_ious > 0)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        recall_coverage: Recall_coverage = {
            'max_spatial': np.nanmax(spatial_mious),
            'max_spatiotemp': np.nanmax(st_mious)}
    return recall_coverage


def _compute_daly_recall_coverage_v(
        v_gt_tubes: V_dict[Frametube],
        v_stubes: V_dict[Sframetube]
            ) -> V_dict[Recall_coverage]:
    v_rcovs: V_dict[Recall_coverage] = {}
    for vid, gt_tubes in v_gt_tubes.items():
        proposals: List[Sframetube] = v_stubes.get(vid, [])
        proposals_frange = np.array([
            (x['start_frame'], x['end_frame']) for x in proposals])
        rcovs = []
        for gt_tube in gt_tubes:
            rcoverage = _compute_daly_recall_coverage(
                    gt_tube, proposals, proposals_frange)
            rcovs.append(rcoverage)
        v_rcovs[vid] = rcovs
    return v_rcovs


def compute_daly_recall_coverage_av(
        av_gt_tubes: AV_dict[Frametube],
        av_stubes: AV_dict[Sframetube],
            ) -> AV_dict[Recall_coverage]:
    """
    For each GT tube, compute best possible IOU over proposals
    """
    av_rcovs: AV_dict[Recall_coverage] = {}
    for action_cls in av_gt_tubes.keys():
        av_rcovs[action_cls] = _compute_daly_recall_coverage_v(
                    av_gt_tubes[action_cls], av_stubes[action_cls])
    return av_rcovs


def tube_daly_recall_as_df(
        df_rcovs: pd.DataFrame,
        iou_thresholds: List[float],
        column: Literal['max_spatial', 'max_spatiotemp']
            ) -> pd.DataFrame:
    # Compute s/st recall for thresh
    lst = []
    for thresh in iou_thresholds:
        good = df_rcovs[column] > thresh
        recall = good.groupby(level=0).mean()
        recall['all'] = good.mean()
        recall.name = f'{thresh:.2f}'
        lst.append(recall)
    df = pd.concat(lst, axis=1)
    return df


def _tube_daly_ap_v(
        v_gt_tubes: V_dict[Frametube],
        v_stubes: V_dict[Sframetube],
        iou_thresholds: List[float],
        spatiotemporal: bool,
            ) -> Dict[float, float]:
    # Convert to flat ap-able representation
    use_diff = True  # no difference, since no diff flags exist
    use_07_metric = False  # no reason to use this metric
    fgts: List[AP_fgt_tube] = []
    fdets: List[AP_fdet_tube] = []
    for vid, gt_tubes in v_gt_tubes.items():
        for i, gt_tube in enumerate(gt_tubes):
            fgt: AP_fgt_tube = {
                'ind': (vid, i),
                'obj': gt_tube,
                'diff': False
            }
            fgts.append(fgt)
    for vid, stubes in v_stubes.items():
        for i, stube in enumerate(stubes):
            fdet: AP_fdet_tube = {
                'ind': (vid, i),
                'obj': stube,
                'score': stube['score']}
            fdets.append(fdet)
    # compute via cls
    thresh_ap: Dict[float, float] = {}
    ap_computer = AP_tube_computer(fgts, fdets)
    for iou_thresh in iou_thresholds:
        thresh_ap[iou_thresh] = ap_computer.compute_ap(
                iou_thresh, spatiotemporal, use_diff, use_07_metric)
    return thresh_ap


def _tube_daly_ap_av(
    av_gt_tubes: AV_dict[Frametube],
    av_stubes: AV_dict[Sframetube],
    iou_thresholds: List[float],
    spatiotemporal: bool,
        ) -> Dict[DALY_action_name, Dict[float, float]]:
    cls_thresh_ap = {}
    for action_cls in av_gt_tubes.keys():
        thresh_ap = _tube_daly_ap_v(
            av_gt_tubes[action_cls], av_stubes[action_cls],
            iou_thresholds, spatiotemporal)
        cls_thresh_ap[action_cls] = thresh_ap
    return cls_thresh_ap


def compute_recall_for_avtubes(
        av_gt_tubes: AV_dict[Frametube],
        av_stubes: AV_dict[Sframetube],
        iou_thresholds: List[float],
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute recall tables
    """
    av_rcovs: AV_dict[Recall_coverage] = \
            compute_daly_recall_coverage_av(av_gt_tubes, av_stubes)
    df_rcovs_ = {}
    for action_cls, v_rcovs in av_rcovs.items():
        for vid, rcovs in v_rcovs.items():
            for i, rcov in enumerate(rcovs):
                df_rcovs_[(action_cls, vid, i)] = rcov
    df_rcovs = pd.DataFrame(df_rcovs_).T
    dft_recall_s = tube_daly_recall_as_df(
            df_rcovs, iou_thresholds, 'max_spatial')
    dft_recall_st = tube_daly_recall_as_df(
            df_rcovs, iou_thresholds, 'max_spatiotemp')
    table_recall_s = snippets.df_to_table_v2((dft_recall_s*100).round(2))
    table_recall_st = snippets.df_to_table_v2((dft_recall_st*100).round(2))
    return table_recall_s, table_recall_st


def compute_ap_for_avtubes(
    av_gt_tubes: AV_dict[Frametube],
    av_stubes: AV_dict[Sframetube],
    iou_thresholds: List[float],
    spatiotemporal: bool,
        ) -> pd.DataFrame:
    """
    Compute ap table
    """
    cls_thresh_ap: \
        Dict[DALY_action_name, Dict[float, float]] = _tube_daly_ap_av(
            av_gt_tubes, av_stubes, iou_thresholds, spatiotemporal)

    dft_ap = pd.DataFrame(cls_thresh_ap).T
    dft_ap = dft_ap.sort_index()
    dft_ap.loc['all'] = dft_ap.mean()
    table_ap = snippets.df_to_table_v2((dft_ap*100).round(2))
    return table_ap


def compute_ap_for_video_datalist(
        datalist, predicted_datalist,
        object_names, iou_thresh: float,
        ) -> Dict[str, float]:
    o_fgts: Dict[str, List[AP_fgt_framebox]] = \
            {on: [] for on in object_names}
    for record in datalist:
        vid = record['vid']
        iframe = record['video_frame_number']
        for anno_id, anno in enumerate(record['annotations']):
            object_name = object_names[anno['category_id']]
            fgt: AP_fgt_framebox = {
                'ind': (vid, iframe, anno_id),
                'obj': anno['bbox'],
                'diff': False
            }
            o_fgts[object_name].append(fgt)
    o_fdets: Dict[str, List[AP_fdet_framebox]] = \
            {on: [] for on in object_names}
    for record, pred_item in zip(datalist, predicted_datalist):
        vid = record['vid']
        iframe = record['video_frame_number']
        pred_boxes = pred_item.pred_boxes.tensor.numpy()
        scores = pred_item.scores.numpy()
        pred_classes = pred_item.pred_classes.numpy()
        for det_id, (bbox, score, cls_ind) in enumerate(
                zip(pred_boxes, scores, pred_classes)):
            object_name = object_names[cls_ind]
            fdet: AP_fdet_framebox = {
                'ind': (vid, iframe, anno_id),
                'obj': bbox,
                'score': score
            }
            o_fdets[object_name].append(fdet)
    # Params
    use_07_metric = False
    use_diff = False
    iou_thresh = 0.5
    object_classes = object_names
    ap_per_cls: Dict[str, float] = {}
    for obj_cls in object_classes:
        fgts = o_fgts[obj_cls]
        fdets = o_fdets[obj_cls]
        ap_computer = AP_framebox_computer(fgts, fdets)
        ap = ap_computer.compute_ap(
                use_diff, iou_thresh, use_07_metric)
        ap_per_cls[obj_cls] = ap
    return ap_per_cls
