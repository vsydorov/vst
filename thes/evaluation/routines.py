import logging
import numpy as np
import pandas as pd
from typing import (Dict, List, Literal, Tuple, cast)

from thes.data.dataset.external import (
        Action_name_daly, Vid_daly, Vid)
from thes.data.tubes.types import (
        Frametube, Sframetube, V_dict, AV_dict,
        DALY_gt_tube_index, I_weingroup)
from thes.data.tubes.routines import (
        temporal_ious_where_positive, temporal_IOU)
from thes.detectron.daly import (Datalist, Dl_record)
from thes.evaluation.ap import (
        AP_fgt_framebox, AP_fdet_framebox,
        AP_fgt_tube, AP_fdet_tube,
        AP_tube_computer, AP_framebox_computer)


log = logging.getLogger(__name__)


def _convert_to_flat_representation(v_gt_tubes, v_stubes):
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
    return fgts, fdets


def _compute_eligible_tubes_for_eval(
        fgts: List[AP_fgt_tube],
        fdets: List[AP_fdet_tube]
            ) -> Dict[int, Dict[int, float]]:
    if len(fdets) == 0:
        return {}
    # Group fdets belonging to same vid
    ifdet_vid_groups: Dict[Vid, List[int]] = {}
    for ifdet, fdet in enumerate(fdets):
        vid = fdet['ind'][0]
        ifdet_vid_groups.setdefault(vid, []).append(ifdet)
    # ifgts to ifdets (belonging to same vid)
    ifgt_to_ifdets_vid_groups: Dict[int, List[int]] = {}
    for ifgt, fgt in enumerate(fgts):
        vid = fgt['ind'][0]
        ifgt_to_ifdets_vid_groups[ifgt] = ifdet_vid_groups.get(vid, [])
    proposals_frange = np.array([(
        f['obj']['start_frame'], f['obj']['end_frame'])
        for f in fdets])
    ifgt_to_ifdets_tious: Dict[int, Dict[int, float]] = {}
    for ifgt, ifdets in ifgt_to_ifdets_vid_groups.items():
        fgt = fgts[ifgt]
        ptious, pids = temporal_ious_where_positive(
            fgt['obj']['start_frame'], fgt['obj']['end_frame'],
            proposals_frange[ifdets, :])
        for pid, ptiou in zip(pids, ptious):
            ifdet = ifdets[pid]
            ifgt_to_ifdets_tious.setdefault(ifgt, {})[ifdet] = ptiou
    ifdet_to_ifgt_tious: Dict[int, Dict[int, float]] = {}
    for ifgt, ifdet_to_tiou in ifgt_to_ifdets_tious.items():
        for ifdet, tiou in ifdet_to_tiou.items():
            ifdet_to_ifgt_tious.setdefault(
                    ifdet, {})[ifgt] = tiou
    return ifdet_to_ifgt_tious


def _compute_eligible_tubes_for_eval_weingroup(
        fgts: List[AP_fgt_tube],
        fdets: List[AP_fdet_tube],
        wgi_to_gti: Dict[I_weingroup, DALY_gt_tube_index],
            ) -> Dict[int, Dict[int, float]]:
    if len(fdets) == 0:
        return {}
    gti_to_ifgt = {fgt['obj']['index']: ifgt
            for ifgt, fgt in enumerate(fgts)}
    # Break detections into weingroups
    wgi_to_ifdets: Dict[I_weingroup, List[int]] = {}
    for ifdet, fdet in enumerate(fdets):
        (vid, bunch_id, tube_id) = fdet['obj']['index']
        (wgi_to_ifdets
                .setdefault((vid, bunch_id), [])  # type: ignore
                .append(ifdet))

    ifdet_to_ifgt_tious: Dict[int, Dict[int, float]] = {}
    for wgi, ifdets in wgi_to_ifdets.items():
        gti = wgi_to_gti.get(wgi)
        ifgt = gti_to_ifgt.get(gti)  # type: ignore
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


def _tube_daly_ap_v(
        v_gt_tubes: V_dict[Frametube],
        v_stubes: V_dict[Sframetube],
        iou_thresholds: List[float],
        spatiotemporal: bool,
            ) -> Dict[float, float]:
    use_diff = True  # no difference, since no diff flags exist
    use_07_metric = False  # no reason to use this metric
    # Convert to flat ap-able representation
    fgts, fdets = _convert_to_flat_representation(v_gt_tubes, v_stubes)
    # Prepare eligible computations
    det_to_eligible_gt = _compute_eligible_tubes_for_eval(fgts, fdets)

    thresh_ap: Dict[float, float] = {}
    ap_computer = AP_tube_computer(fgts, fdets, det_to_eligible_gt)
    for iou_thresh in iou_thresholds:
        thresh_ap[iou_thresh] = ap_computer.compute_ap(
                iou_thresh, spatiotemporal, use_diff, use_07_metric)
    return thresh_ap


def _tube_daly_ap_v_weingroup(
        v_gt_tubes: V_dict[Frametube],
        v_stubes: V_dict[Sframetube],
        iou_thresholds: List[float],
        spatiotemporal: bool,
        wgi_to_gti: Dict[Tuple[Vid_daly, int], DALY_gt_tube_index],
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


def temporal_coverage_stats(ex_df, gt_df):
    # Let's compute temporal coverage stats
    coverage_dict = {}
    for key, line in tqdm(gt_df.iterrows(), total=len(gt_df),
            desc='total_coverage_stats'):
        vid_tubes = ex_df.query('vid=="{}"'.format(line['vid']))
        s, e = line['start_frame'], line['end_frame']
        frange = vid_tubes[['min_frame', 'max_frame']].to_numpy()
        # Interesting tubes: one of the limits lies inside
        limits_inside = (s <= frange) & (frange <= e)
        either_limit_inside = limits_inside.any(1)
        interesting_frange = frange[either_limit_inside]
        if len(interesting_frange):
            # Clip within range, compute "intersect" part
            clipped_frange = np.clip(interesting_frange, s, e)
            total_gt = e - s
            total_intersect = clipped_frange[:, 1] - clipped_frange[:, 0]
            # Compute union
            union_frange = np.empty_like(interesting_frange)
            union_frange[:, 0] = np.minimum(interesting_frange[:, 0], s)
            union_frange[:, 1] = np.maximum(interesting_frange[:, 1], e)
            total_union = union_frange[:, 1] - union_frange[:, 0]
            # Compute fraction
            fraction_intersect = total_intersect/total_gt
            fraction_iou = total_intersect/total_union
            max_intersect = np.max(fraction_intersect)
            max_iou = np.max(fraction_iou)
        else:
            max_intersect = 0.0
            max_iou = 0.0
        coverage_dict[key] = [max_intersect, max_iou]

    coverage_df = pd.DataFrame(coverage_dict).T
    coverage_df.columns = ['minter', 'miou']

    # Compute stats
    cdf = gt_df.copy()
    cdf[coverage_df.columns] = coverage_df[coverage_df.columns]
    stats = {}
    stats['mean_iou'] = cdf.miou.mean() * 100
    stats['mean_inter'] = cdf.minter.mean() * 100
    N = len(cdf)
    N_tubes_above_iou05 = (cdf.miou >= 0.5).sum()
    stats['tubes_above_iou_0.5'] = '{}/{} = {}'.format(
            N_tubes_above_iou05, N, N_tubes_above_iou05/N * 100)
    N_tubes_above_minter05 = (cdf.minter >= 0.5).sum()
    stats['tubes_above_inter_0.5'] = '{}/{} = {}'.format(
            N_tubes_above_minter05, N, N_tubes_above_minter05/N * 100)
    return coverage_df, stats


def spatial_coverage_stats(ex_df, gt_df, dataset, extracted_tubes):
    # Let's compute spatial coverage stats
    coverage_dict = {}
    for key, line in tqdm(gt_df.iterrows(), total=len(gt_df),
            desc='total_coverage_stats'):
        vid_tubes = ex_df.query('vid=="{}"'.format(line['vid']))
        s, e = line['start_frame'], line['end_frame']
        frange = vid_tubes[['min_frame', 'max_frame']].to_numpy()
        # Interesting tubes: one of the limits lies inside
        limits_inside = (s <= frange) & (frange <= e)
        either_limit_inside = limits_inside.any(1)
        interesting_frange = frange[either_limit_inside]
        tubes_inside = vid_tubes.iloc[either_limit_inside]
        if len(interesting_frange):
            # // Compute keyframe intersections
            # Retrieve GT keyframes
            gt_instance = (dataset.video_odict[line.vid]
                    ['instances'][line.action][line.ins_id])
            vmp4 = dataset.source_videos[line.vid]
            gt_frames = []
            gt_boxes_unscaled = []
            for kf in gt_instance['keyframes']:
                gt_frames.append(kf['frameNumber'])
                gt_boxes_unscaled.append(kf['boundingBox'])
            gt_frames = np.array(gt_frames)
            gt_boxes_unscaled = np.vstack(gt_boxes_unscaled)
            gt_boxes = gt_boxes_unscaled * np.tile(
                    [vmp4['width'], vmp4['height']], 2)
            # Retrieve those keyframes from proposals that match gt_frames
            retrieved = []
            for i, tube_row in tubes_inside.iterrows():
                ext_tube = extracted_tubes[
                        tube_row.vid, tube_row.bunch_id, tube_row.tube_id]
                found = np.isin(gt_frames, ext_tube['frame_inds'])
                found_ind = np.searchsorted(ext_tube['frame_inds'], gt_frames)
                found_ind[~found] = 0
                found_boxes = ext_tube['boxes'][found_ind]
                retrieved.append({'boxes': found_boxes, 'found': found})
            # Compute pairwise box IOUs
            pairwise_box_ious = []
            for i, x in enumerate(retrieved):
                ious = []
                for gt_box, p_box, found in zip(
                        gt_boxes, x['boxes'], x['found']):
                    if not found:
                        iou = 0.0
                    else:
                        # Computing IOU
                        inter = np.r_[
                            np.maximum(gt_box[:2], p_box[:2]),
                            np.minimum(gt_box[2:], p_box[2:])]
                        if np.any(inter[:2] > inter[2:]):
                            iou = 0.0
                        else:
                            inter_area = np.prod(inter[2:] - inter[:2])
                            union_area = (
                                np.prod(gt_box[2:] - gt_box[:2]) +
                                np.prod(p_box[2:] - p_box[:2]) - inter_area)
                            iou = inter_area/union_area
                    ious.append(iou)
                pairwise_box_ious.append(ious)
            pairwise_box_ious = np.array(pairwise_box_ious)
            # Mean per GT frame
            mean_box_ious = np.mean(pairwise_box_ious, axis=1)
            # Maximum iou
            max_iou = np.max(mean_box_ious)
        else:
            max_iou = 0.0
        coverage_dict[key] = max_iou

    coverage_df = pd.Series(coverage_dict).to_frame()
    coverage_df.columns = ['max_iou']

    cdf = gt_df.copy()
    cdf[coverage_df.columns] = coverage_df[coverage_df.columns]
    stats = {}
    stats['mean_iou'] = cdf.max_iou.mean()*100
    N = len(cdf)
    N_tubes_above_iou05 = (cdf.max_iou > 0.5).sum()
    stats['N_tubes_above_iou05'] = '{}/{} = {}'.format(
            N_tubes_above_iou05, N, N_tubes_above_iou05/N * 100)
    N_tubes_above_iou03 = (cdf.max_iou > 0.3).sum()
    stats['N_tubes_above_iou03'] = '{}/{} = {}'.format(
            N_tubes_above_iou03, N, N_tubes_above_iou03/N * 100)

    # Stats per action
    acdf = cdf[['action', 'max_iou']].copy()
    acdf['iou_above_05'] = acdf['max_iou'] > 0.5
    acdf['iou_above_03'] = acdf['max_iou'] > 0.3
    sum_per_action = acdf.groupby('action').sum()
    count_per_action = acdf.groupby('action').count()
    stats_df_peraction = sum_per_action/count_per_action*100
    return coverage_df, stats, stats_df_peraction
