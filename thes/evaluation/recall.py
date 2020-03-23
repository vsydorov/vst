import logging
import warnings
import numpy as np
from typing import (TypedDict, List,)
from thes.data.tubes.types import (Frametube, Sframetube, V_dict, AV_dict)
from thes.data.tubes.routines import (
        spatial_tube_iou_v2,)


log = logging.getLogger(__name__)


class Recall_coverage(TypedDict):
    max_spatial: float
    max_spatiotemp: float


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
