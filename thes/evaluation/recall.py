import logging
import warnings
import numpy as np
from typing import (TypedDict, List,)
from thes.data.tubes.types import (Frametube, Sframetube, V_dict, AV_dict)
from thes.data.tubes.routines import (
        spatial_tube_iou_v2,
        temporal_ious_where_positive)


log = logging.getLogger(__name__)


class Recall_coverage(TypedDict):
    max_spatial: float
    max_spatiotemp: float


def _compute_daly_recall_coverage(
        gt_tube: Frametube,
        proposals: List[Sframetube],
        proposals_frange: np.ndarray,) -> Recall_coverage:
    if len(proposals) == 0:
        return {'max_spatial': np.nan,
                'max_spatiotemp': np.nan}
    spatial_mious = np.ones(len(proposals)) * np.nan
    temp_ious = np.zeros(len(proposals))
    # Temporal
    ptious, pids = temporal_ious_where_positive(
            gt_tube['start_frame'], gt_tube['end_frame'], proposals_frange)
    temp_ious[pids] = ptious
    # Spatial (where temporal >0)
    pproposals = [proposals[pid] for pid in pids]
    pmious = [spatial_tube_iou_v2(p, gt_tube)[0] for p in pproposals]
    spatial_mious[pids] = pmious
    # Spatio-temporal
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
