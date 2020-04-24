import logging
import warnings
import numpy as np
import pandas as pd
from typing import (  # NOQA
    TypedDict, List, Literal, Tuple,
    Dict, cast, Set, Sequence, Optional,
    TypeVar)

from thes.data.tubes.types import (
        Frametube, Frametube_scored,
        V_dict, AV_dict, T_dgt)
from thes.data.tubes.routines import (
        spatial_tube_iou_v3,
        temporal_ious_where_positive)


log = logging.getLogger(__name__)


TV_frametube_co = TypeVar('TV_frametube_co', bound=Frametube)
TV_frametube_scored_co = \
    TypeVar('TV_frametube_scored_co', bound=Frametube_scored)


class Recall_coverage(TypedDict):
    max_spatial: float
    max_spatiotemp: float


def _compute_daly_recall_coverage(
        gt_tube: Frametube,
        proposals: List[TV_frametube_scored_co],
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
    pmious = [spatial_tube_iou_v3(p, gt_tube) for p in pproposals]
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
        v_gt_tubes: V_dict[T_dgt],
        v_stubes: V_dict[TV_frametube_scored_co],
        use_diff: bool,
            ) -> V_dict[Recall_coverage]:
    v_rcovs: V_dict[Recall_coverage] = {}
    for vid, gt_tubes in v_gt_tubes.items():
        proposals: List[TV_frametube_scored_co] = v_stubes.get(vid, [])
        proposals_frange = np.array([
            (x['start_frame'], x['end_frame']) for x in proposals])
        rcovs = []
        for gt_tube in gt_tubes:
            fl = gt_tube['flags']
            diff = fl['isReflection'] or fl['isAmbiguous']
            if diff and not use_diff:
                continue
            rcoverage = _compute_daly_recall_coverage(
                    gt_tube, proposals, proposals_frange)
            rcovs.append(rcoverage)
        v_rcovs[vid] = rcovs
    return v_rcovs


def compute_daly_recall_coverage_av(
        av_gt_tubes: AV_dict[T_dgt],
        av_stubes: AV_dict[TV_frametube_scored_co],
        use_diff: bool,
            ) -> AV_dict[Recall_coverage]:
    """
    For each GT tube, compute best possible IOU over proposals
    """
    av_rcovs: AV_dict[Recall_coverage] = {}
    for action_cls in av_gt_tubes.keys():
        av_rcovs[action_cls] = _compute_daly_recall_coverage_v(
            av_gt_tubes[action_cls], av_stubes[action_cls], use_diff)
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
        # recall.name = f'{thresh:.2f}'
        recall.name = thresh
        lst.append(recall)
    df = pd.concat(lst, axis=1)
    return df


def compute_recall_for_avtubes_as_dfs(
        av_gt_tubes: AV_dict[T_dgt],
        av_stubes: AV_dict[TV_frametube_scored_co],
        iou_thresholds: List[float],
        use_diff: bool,
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute recall tables
    """
    av_rcovs: AV_dict[Recall_coverage] = \
            compute_daly_recall_coverage_av(
                    av_gt_tubes, av_stubes, use_diff)
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
    return dft_recall_s, dft_recall_st
