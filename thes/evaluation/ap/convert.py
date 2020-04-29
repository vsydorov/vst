import logging
import numpy as np
import pandas as pd
from typing import (Dict, List, Literal, Tuple, cast, TypeVar, NewType)

from thes.data.dataset.external import (
        Action_name_daly, Vid_daly, Vid)
from thes.data.tubes.types import (
        Frametube, Frametube_scored, V_dict, AV_dict,
        I_dgt, T_dgt)
from thes.data.tubes.routines import (
        temporal_ious_where_positive, temporal_IOU)
from thes.detectron.daly import (Datalist, Dl_record)
from thes.evaluation.ap.core import (
        AP_fgt_framebox, AP_fdet_framebox,
        AP_fgt_tube, AP_fdet_tube,
        AP_tube_computer, AP_framebox_computer,
        IFDET, IFGT)


log = logging.getLogger(__name__)


TV_frametube_scored_co = \
    TypeVar('TV_frametube_scored_co', bound=Frametube_scored)


"""Detection of video level tubes"""


def _convert_to_flat_representation(
        v_gt_tubes: V_dict[T_dgt],
        v_stubes: V_dict[TV_frametube_scored_co]):
    fgts: List[AP_fgt_tube] = []
    fdets: List[AP_fdet_tube] = []
    for vid, gt_tubes in v_gt_tubes.items():
        for i, gt_tube in enumerate(gt_tubes):
            fl = gt_tube['flags']
            diff = fl['isReflection'] or fl['isAmbiguous']
            fgt: AP_fgt_tube = {
                'ind': (vid, i),
                'obj': gt_tube,
                'diff': diff
            }
            fgts.append(fgt)
    for vid, stubes in v_stubes.items():
        for i, stube in enumerate(stubes):
            score = cast(float, stube['score'])
            fdet: AP_fdet_tube = {
                'ind': (vid, i),
                'obj': stube,
                'score': score}
            fdets.append(fdet)
    return fgts, fdets


def _precompute_temporal_overlaps(
        fgts: List[AP_fgt_tube],
        fdets: List[AP_fdet_tube]
            ) -> Dict[IFDET, Dict[IFGT, float]]:
    if len(fdets) == 0:
        return {}
    # Group FDETS belonging to the same vid
    ifdet_vid_groups: Dict[Vid, List[IFDET]] = {}
    for ifdet, fdet in enumerate(fdets):
        vid = fdet['ind'][0]
        ifdet = cast(IFDET, ifdet)
        ifdet_vid_groups.setdefault(vid, []).append(ifdet)

    # For every FGT, find corresponding FDETS (common VID)
    ifgt_to_ifdets_vid_groups: Dict[IFGT, List[IFDET]] = {}
    for ifgt, fgt in enumerate(fgts):
        vid = fgt['ind'][0]
        ifgt = cast(IFGT, ifgt)
        ifgt_to_ifdets_vid_groups[ifgt] = ifdet_vid_groups.get(vid, [])

    # For every FGT, record Temp_IOU(FGT, FDET), if >0
    proposals_frange = np.array([(
        f['obj']['start_frame'], f['obj']['end_frame'])
        for f in fdets])
    ifgt_to_ifdets_tious: Dict[IFGT, Dict[IFDET, float]] = {}
    for ifgt, ifdets in ifgt_to_ifdets_vid_groups.items():
        fgt = fgts[ifgt]
        ptious, pids = temporal_ious_where_positive(
            fgt['obj']['start_frame'], fgt['obj']['end_frame'],
            proposals_frange[ifdets, :])
        for pid, ptiou in zip(pids, ptious):
            ifdet = ifdets[pid]
            ifdet = cast(IFDET, ifdet)
            ifgt_to_ifdets_tious.setdefault(ifgt, {})[ifdet] = ptiou

    # Flip it - for every FDET, record positive Temp_IOU
    ifdet_to_ifgt_tious: Dict[IFDET, Dict[IFGT, float]] = {}
    for ifgt, ifdet_to_tiou in ifgt_to_ifdets_tious.items():
        for ifdet, tiou in ifdet_to_tiou.items():
            ifdet_to_ifgt_tious.setdefault(
                    ifdet, {})[ifgt] = tiou
    return ifdet_to_ifgt_tious


def _tube_daly_ap_v(
        v_gt_tubes: V_dict[T_dgt],
        v_stubes: V_dict[TV_frametube_scored_co],
        iou_thresholds: List[float],
        spatiotemporal: bool,
        use_diff: bool,
            ) -> Dict[float, float]:
    use_07_metric = False  # no reason to use this metric
    # Convert to flat representation
    fgts, fdets = _convert_to_flat_representation(v_gt_tubes, v_stubes)
    # Prepare eligible computations
    det_to_eligible_gt = _precompute_temporal_overlaps(fgts, fdets)

    thresh_ap: Dict[float, float] = {}
    ap_computer = AP_tube_computer(fgts, fdets, det_to_eligible_gt)
    for iou_thresh in iou_thresholds:
        thresh_ap[iou_thresh] = ap_computer.compute_ap(
                iou_thresh, spatiotemporal, use_diff, use_07_metric)
    return thresh_ap


def _tube_daly_ap_av(
    av_gt_tubes: AV_dict[T_dgt],
    av_stubes: AV_dict[TV_frametube_scored_co],
    iou_thresholds: List[float],
    spatiotemporal: bool,
    use_diff: bool,
        ) -> Dict[Action_name_daly, Dict[float, float]]:
    cls_thresh_ap = {}
    for action_cls in av_gt_tubes.keys():
        thresh_ap = _tube_daly_ap_v(
            av_gt_tubes[action_cls], av_stubes[action_cls],
            iou_thresholds, spatiotemporal, use_diff)
        cls_thresh_ap[action_cls] = thresh_ap
    return cls_thresh_ap


def _apdict_to_df(cls_thresh_ap):
    dft_ap = pd.DataFrame(cls_thresh_ap).T
    dft_ap = dft_ap.sort_index()
    dft_ap.loc['all'] = dft_ap.mean()
    return dft_ap


def compute_ap_for_avtubes_as_df(
    av_gt_tubes: AV_dict[T_dgt],
    av_stubes: AV_dict[TV_frametube_scored_co],
    iou_thresholds: List[float],
    spatiotemporal: bool,
    use_diff: bool,
        ) -> pd.DataFrame:
    """
    Compute ap table
    """
    cls_thresh_ap: Dict[Action_name_daly, Dict[float, float]]
    cls_thresh_ap = _tube_daly_ap_av(
        av_gt_tubes, av_stubes, iou_thresholds, spatiotemporal, use_diff)
    return _apdict_to_df(cls_thresh_ap)


# weingroup twist


I_weingroup = Tuple[Vid_daly, int]


def _compute_eligible_tubes_for_eval_weingroup(
        fgts: List[AP_fgt_tube],
        fdets: List[AP_fdet_tube],
        wgi_to_gti: Dict[I_weingroup, I_dgt],
            ) -> Dict[IFDET, Dict[IFGT, float]]:
    if len(fdets) == 0:
        return {}
    gti_to_ifgt: Dict[I_dgt, IFGT] = {
            fgt['obj']['index']: ifgt  # type: ignore
            for ifgt, fgt in enumerate(fgts)}
    # Break detections into weingroups
    wgi_to_ifdets: Dict[I_weingroup, List[IFDET]] = {}
    for ifdet, fdet in enumerate(fdets):
        (vid, bunch_id, tube_id) = fdet['obj']['index']  # type: ignore
        ifdet = cast(IFDET, ifdet)
        (wgi_to_ifdets
                .setdefault((vid, bunch_id), [])  # type: ignore
                .append(ifdet))

    ifdet_to_ifgt_tious: Dict[IFDET, Dict[IFGT, float]] = {}
    for wgi, ifdets in wgi_to_ifdets.items():
        gti = wgi_to_gti.get(wgi)
        if gti is None:
            continue
        ifgt = gti_to_ifgt.get(gti)
        if ifgt is None:
            continue
        fgt = fgts[ifgt]
        for ifdet in ifdets:
            fdet = fdets[ifdet]
            temp_iou = temporal_IOU(
                    fdet['obj']['start_frame'], fdet['obj']['end_frame'],
                    fgt['obj']['start_frame'], fgt['obj']['end_frame'])
            ifdet_to_ifgt_tious[ifdet] = {ifgt: temp_iou}
    return ifdet_to_ifgt_tious


def _tube_daly_ap_v_weingroup(
        wgi_to_gti: Dict[I_weingroup, I_dgt],
        v_gt_tubes: V_dict[T_dgt],
        v_stubes: V_dict[TV_frametube_scored_co],
        iou_thresholds: List[float],
        spatiotemporal: bool,
        use_diff: bool,
            ) -> Dict[float, float]:
    use_diff = True  # no difference, since no diff flags exist
    use_07_metric = False  # no reason to use this metric
    # Convert to flat ap-able representation
    fgts, fdets = _convert_to_flat_representation(v_gt_tubes, v_stubes)
    # Eligible computations are weingroup dependent
    det_to_eligible_gt = _compute_eligible_tubes_for_eval_weingroup(
            fgts, fdets, wgi_to_gti)
    thresh_ap: Dict[float, float] = {}
    ap_computer = AP_tube_computer(fgts, fdets, det_to_eligible_gt)
    for iou_thresh in iou_thresholds:
        thresh_ap[iou_thresh] = ap_computer.compute_ap(
                iou_thresh, spatiotemporal, use_diff, use_07_metric)
    return thresh_ap


def compute_ap_for_avtubes_WG_as_df(
        wgi_to_gti: Dict[I_weingroup, I_dgt],
        av_gt_tubes: AV_dict[T_dgt],
        av_stubes: AV_dict[TV_frametube_scored_co],
        iou_thresholds: List[float],
        spatiotemporal: bool,
        use_diff: bool
        ) -> pd.DataFrame:
    cls_thresh_ap = {}
    for action_cls in av_gt_tubes.keys():
        thresh_ap = _tube_daly_ap_v_weingroup(wgi_to_gti,
            av_gt_tubes[action_cls], av_stubes[action_cls],
            iou_thresholds, spatiotemporal, use_diff)
        cls_thresh_ap[action_cls] = thresh_ap
    dft_ap = pd.DataFrame(cls_thresh_ap).T
    dft_ap = dft_ap.sort_index()
    dft_ap.loc['all'] = dft_ap.mean()
    return dft_ap


"""Detection of image level boxes"""


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
