import logging
import pandas as pd
from typing import (Dict, List, Literal, Tuple)

from thes.tools import snippets
from thes.data.dataset.external import (Action_name_daly,)
from thes.data.tubes.types import (Frametube, Sframetube, V_dict, AV_dict)
from thes.detectron.daly import (Datalist, Dl_record)
from thes.evaluation.ap import (
        AP_fgt_framebox, AP_fdet_framebox,
        AP_fgt_tube, AP_fdet_tube,
        AP_tube_computer, AP_framebox_computer)
from thes.evaluation.recall import (
        compute_daly_recall_coverage_av, Recall_coverage,)


log = logging.getLogger(__name__)


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
        ) -> Dict[Action_name_daly, Dict[float, float]]:
    cls_thresh_ap = {}
    for action_cls in av_gt_tubes.keys():
        thresh_ap = _tube_daly_ap_v(
            av_gt_tubes[action_cls], av_stubes[action_cls],
            iou_thresholds, spatiotemporal)
        cls_thresh_ap[action_cls] = thresh_ap
    return cls_thresh_ap


def compute_recall_for_avtubes_as_dfs(
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
    return dft_recall_s, dft_recall_st


def compute_ap_for_avtubes_as_df(
    av_gt_tubes: AV_dict[Frametube],
    av_stubes: AV_dict[Sframetube],
    iou_thresholds: List[float],
    spatiotemporal: bool,
        ) -> pd.DataFrame:
    """
    Compute ap table
    """
    cls_thresh_ap: \
        Dict[Action_name_daly, Dict[float, float]] = _tube_daly_ap_av(
            av_gt_tubes, av_stubes, iou_thresholds, spatiotemporal)
    dft_ap = pd.DataFrame(cls_thresh_ap).T
    dft_ap = dft_ap.sort_index()
    dft_ap.loc['all'] = dft_ap.mean()
    return dft_ap


def compute_ap_for_video_datalist(
        datalist: Datalist,
        predicted_datalist,
        object_names,
        iou_thresh: float,
            ) -> Dict[str, float]:
    o_fgts: Dict[str, List[AP_fgt_framebox]] = \
            {on: [] for on in object_names}
    record: Dl_record
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
    object_classes = object_names
    ap_per_cls: Dict[str, float] = {}
    for obj_cls in object_classes:
        fgts = o_fgts[obj_cls]
        fdets = o_fdets[obj_cls]
        ap_computer = AP_framebox_computer(fgts, fdets)
        ap = ap_computer.compute_ap(
                iou_thresh, use_diff, use_07_metric)
        ap_per_cls[obj_cls] = ap
    return ap_per_cls
