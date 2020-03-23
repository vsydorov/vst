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
        spatial_tube_iou, spatial_tube_iou_v2,
)
from thes.data.tubes.types import (Frametube, Sframetube, V_dict, AV_dict)
from thes.evaluation.types import (
        AP_fgt_framebox, AP_fdet_framebox,
        AP_fgt_tube, AP_fdet_tube,
        Recall_coverage, Options_tube_ap, Stats_daly_ap)
from thes.evaluation.ap import voc_ap


log = logging.getLogger(__name__)


class AP_fgt(TypedDict):
    ind: Any
    obj: Any
    diff: bool


class AP_fdet(TypedDict):
    ind: Any
    obj: Any
    score: float


class AP_computer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_matchable_ifgts(ifdet: int) -> List[int]:
        pass

    @abstractmethod
    def prepare_computation(
            fgts: List[AP_fgt], fdets: List[AP_fdet]):
        pass

    @abstractmethod
    def compute_iou_coverages(self,
            matchable_ifgts: List[int],
            fdets, ifdet, fgts) -> List[float]:
        pass

    def compute_ap(
            self,
            fgts: List[AP_fgt],
            fdets: List[AP_fdet],
            use_diff: bool,
            iou_thresh: float,
            use_07_metric: bool
            ) -> float:
        gt_already_matched = np.zeros(len(fgts), dtype=bool)
        nd = len(fdets)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        if use_diff:
            npos = len(fgts)
        else:
            npos = len([x for x in fgts if not x['diff']])
        # Go through ordered detections
        detection_scores = np.array([x['score'] for x in fdets])
        detection_scores = detection_scores.round(3)
        sorted_inds = np.argsort(-detection_scores)

        for d, ifdet in enumerate(sorted_inds):
            matchable_ifgts: List[int] = self.get_matchable_ifgts(ifdet)
            if not len(matchable_ifgts):
                fp[d] = 1
                continue
            iou_coverages: List[float] = \
                self.compute_iou_coverages(matchable_ifgts, fdets, ifdet, fgts)
            max_coverage_local_id: int = np.argmax(iou_coverages)
            max_coverage: float = iou_coverages[max_coverage_local_id]
            max_coverage_ifgt: int = matchable_ifgts[max_coverage_local_id]
            if max_coverage > iou_thresh:
                if (not use_diff) and fgts[max_coverage_ifgt]['diff']:
                    continue
                if not gt_already_matched[max_coverage_ifgt]:
                    tp[d] = 1
                    gt_already_matched[max_coverage_ifgt] = True
                else:
                    fp[d] = 1
            else:
                fp[d] = 1
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
        return ap


class AP_framebox_computer(AP_computer):
    # ifgts that are in the same frame as ifdets
    ifdet_to_ifgts: Dict[int, List[int]]

    def __init__(self):
        pass

    def prepare_computation(
            self, fgts: List[AP_fgt], fdets: List[AP_fdet]):
        # Group fdets belonging to same frame, assign to fgts
        ifdet_groups: Dict[Tuple[DALY_vid, int], List[int]] = {}
        for ifdet, fdet in enumerate(fdets):
            vf_id = (fdet['ind'][0], fdet['ind'][1])
            ifdet_groups.setdefault(vf_id, []).append(ifdet)
        ifgt_to_ifdets: Dict[int, List[int]] = {}
        for ifgt, fgt in enumerate(fgts):
            vf_id = (fgt['ind'][0], fgt['ind'][1])
            ifgt_to_ifdets[ifgt] = ifdet_groups.get(vf_id, [])
        ifdet_to_ifgts: Dict[int, List[int]] = {}
        for ifgt, ifdets in ifgt_to_ifdets.items():
            for ifdet in ifdets:
                ifdet_to_ifgts.setdefault(ifdet, []).append(ifgt)
        self.ifdet_to_ifgts = ifdet_to_ifgts

    def get_matchable_ifgts(self, ifdet: int) -> List[int]:
        # Check available GTs
        share_image_ifgts: List[int] = self.ifdet_to_ifgts.get(ifdet, [])
        return share_image_ifgts

    def compute_iou_coverages(self,
            matchable_ifgts: List[int],
            fdets, ifdet, fgts) -> List[float]:
        fdet = fdets[ifdet]
        # Compute IOUs
        iou_coverages: List[float] = []
        for ifgt in matchable_ifgts:
            fgt: AP_fgt_framebox = fgts[ifgt]
            iou = numpy_iou(fgt['obj'], fdet['obj'])
            iou_coverages.append(iou)
        return iou_coverages


class AP_tube_computer(AP_computer):
    possible_matches_per_detection: List[Dict[int, float]]
    spatiotemporal: bool

    def __init__(self, spatiotemporal: bool):
        self.spatiotemporal = spatiotemporal

    def prepare_computation(
            self, fgts: List[AP_fgt], fdets: List[AP_fdet]):
        # Precompute 'temporal iou' and indices of tubes
        possible_matches_per_detection: List[Dict[int, float]] = []
        for fdet in fdets:
            ind_to_iou: Dict[int, float] = {}
            det_bf = fdet['obj']['start_frame']
            det_ef = fdet['obj']['end_frame']
            for i_fgt, fgt in enumerate(fgts):
                if fgt['ind'][0] == fdet['ind'][0]:
                    gt_bf = fgt['obj']['start_frame']
                    gt_ef = fgt['obj']['end_frame']
                    temp_iou = temporal_IOU(
                            gt_bf, gt_ef, det_bf, det_ef)
                    if temp_iou > 0.0:
                        ind_to_iou[i_fgt] = temp_iou
            possible_matches_per_detection.append(ind_to_iou)
        self.possible_matches_per_detection = possible_matches_per_detection

    def get_matchable_ifgts(self, ifdet: int) -> List[int]:
        # Check available GTs
        gt_ids_that_overlap: Dict[int, float] = \
                self.possible_matches_per_detection[ifdet]
        return list(gt_ids_that_overlap.keys())

    def compute_iou_coverages(self,
            matchable_ifgts: List[int],
            fdets, ifdet, fgts) -> List[float]:
        fdet = fdets[ifdet]
        gt_ids_that_overlap: Dict[int, float] = \
                self.possible_matches_per_detection[ifdet]
        # Compute IOUs
        iou_coverages: List[float] = []
        for gt_id, temp_iou in gt_ids_that_overlap.items():
            fgt: AP_fgt_tube = fgts[gt_id]
            spatial_iou = spatial_tube_iou(fgt['obj'], fdet['obj'])
            if self.spatiotemporal:
                iou = temp_iou * spatial_iou
            else:
                iou = spatial_iou
            iou_coverages.append(iou)
        return iou_coverages


def compute_framebox_ap(
        fgts: List[AP_fgt_framebox],
        fdets: List[AP_fdet_framebox],
        iou_thresh: float,
        use_07_metric: bool,
        use_diff: bool,
            ) -> float:
    """
    Params:
    use_07_metric: If True, will evaluate AP over 11 points, like in
        VOC2007 evaluation code
    use_diff: If True, will eval difficult proposals in the same way as
        real ones
    """
    # Group fdets belonging to same frame, assign to fgts
    ifdet_groups: Dict[Tuple[DALY_vid, int], List[int]] = {}
    for ifdet, fdet in enumerate(fdets):
        vf_id = (fdet['ind'][0], fdet['ind'][1])
        ifdet_groups.setdefault(vf_id, []).append(ifdet)
    ifgt_to_ifdets: Dict[int, List[int]] = {}
    for ifgt, fgt in enumerate(fgts):
        vf_id = (fgt['ind'][0], fgt['ind'][1])
        ifgt_to_ifdets[ifgt] = ifdet_groups.get(vf_id, [])
    ifdet_to_ifgts: Dict[int, List[int]] = {}
    for ifgt, ifdets in ifgt_to_ifdets.items():
        for ifdet in ifdets:
            ifdet_to_ifgts.setdefault(ifdet, []).append(ifgt)
    # Preparation
    detection_matched = np.zeros(len(fdets), dtype=bool)
    gt_already_matched = np.zeros(len(fgts), dtype=bool)
    # Provenance
    detection_matched_to_which_gt = np.ones(len(fdets), dtype=int)*-1
    iou_coverages_per_detection_ind: Dict[int, List[float]] = {}

    # VOC2007 preparation
    nd = len(fdets)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    if use_diff:
        npos = len(fgts)
    else:
        npos = len([x for x in fgts if not x['diff']])

    # Go through ordered detections
    detection_scores = np.array([x['score'] for x in fdets])
    detection_scores = detection_scores.round(3)
    sorted_inds = np.argsort(-detection_scores)
    for d, ifdet in enumerate(sorted_inds):
        # Check available GTs
        share_image_ifgts: List[int] = ifdet_to_ifgts.get(ifdet, [])
        if not len(share_image_ifgts):
            fp[d] = 1
            continue

        detection: AP_fdet_framebox = fdets[ifdet]
        detection_box = detection['obj']

        # Compute IOUs
        iou_coverages: List[float] = []
        for ifgt in share_image_ifgts:
            gt_box_anno: AP_fgt_framebox = fgts[ifgt]
            gt_box = gt_box_anno['obj']
            iou = numpy_iou(gt_box, detection_box)
            iou_coverages.append(iou)
        # Provenance
        iou_coverages_per_detection_ind[ifdet] = iou_coverages

        max_coverage_local_id = np.argmax(iou_coverages)
        max_coverage = iou_coverages[max_coverage_local_id]
        max_coverage_ifgt = share_image_ifgts[max_coverage_local_id]

        # Mirroring voc_eval
        if max_coverage > iou_thresh:
            if (not use_diff) and fgts[max_coverage_ifgt]['diff']:
                continue
            if not gt_already_matched[max_coverage_ifgt]:
                tp[d] = 1
                detection_matched[ifdet] = True
                gt_already_matched[max_coverage_ifgt] = True
                detection_matched_to_which_gt[ifdet] = max_coverage_ifgt
            else:
                fp[d] = 1
        else:
            fp[d] = 1
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return ap


def compute_tube_ap(
        fgts: List[AP_fgt_tube],
        fdets: List[AP_fdet_tube],
        iou_thresh: float,
        use_07_metric: bool,
        use_diff: bool,
        spatiotemporal: bool,
            ) -> float:
    # Precompute 'temporal iou' and indices of tubes
    possible_matches_per_detection: List[Dict[int, float]] = []
    for fdet in fdets:
        ind_to_iou: Dict[int, float] = {}
        det_bf = fdet['obj']['start_frame']
        det_ef = fdet['obj']['end_frame']
        for i_fgt, fgt in enumerate(fgts):
            if fgt['ind'][0] == fdet['ind'][0]:
                gt_bf = fgt['obj']['start_frame']
                gt_ef = fgt['obj']['end_frame']
                temp_iou = temporal_IOU(
                        gt_bf, gt_ef, det_bf, det_ef)
                if temp_iou > 0.0:
                    ind_to_iou[i_fgt] = temp_iou
        possible_matches_per_detection.append(ind_to_iou)
    # Preparation
    detection_matched = np.zeros(len(fdets), dtype=bool)
    gt_already_matched = np.zeros(len(fgts), dtype=bool)
    # Provenance
    detection_matched_to_which_gt = np.ones(len(fdets), dtype=int)*-1
    iou_coverages_per_detection_ind: Dict[int, List[float]] = {}

    # VOC2007 preparation
    nd = len(fdets)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    if use_diff:
        npos = len(fgts)
    else:
        npos = len([x for x in fgts if not x['diff']])

    # Go through ordered detections
    detection_scores = np.array([x['score'] for x in fdets])
    detection_scores = detection_scores.round(3)
    sorted_inds = np.argsort(-detection_scores)
    for d, detection_ind in enumerate(sorted_inds):
        # Check available GTs
        gt_ids_that_overlap = possible_matches_per_detection[detection_ind]
        if len(gt_ids_that_overlap) == 0:
            fp[d] = 1
            continue

        detection: AP_fdet_tube = fdets[detection_ind]
        detection_tube: Frametube = detection['obj']

        # Compute IOUs
        iou_coverages: List[float] = []
        for gt_id, temp_iou in gt_ids_that_overlap.items():
            gt_tube_anno: AP_fgt_tube = fgts[gt_id]
            gt_tube = gt_tube_anno['obj']
            spatial_iou = spatial_tube_iou(gt_tube, detection_tube)
            if spatiotemporal:
                iou = temp_iou * spatial_iou
            else:
                iou = spatial_iou
            iou_coverages.append(iou)
        # Provenance
        iou_coverages_per_detection_ind[detection_ind] = iou_coverages

        max_coverage_id = np.argmax(iou_coverages)
        max_coverage = iou_coverages[max_coverage_id]
        max_coverage_gt_id = list(gt_ids_that_overlap.keys())[max_coverage_id]

        # Mirror VOC eval
        if max_coverage > iou_thresh:
            if (not use_diff) and fgts[max_coverage_gt_id]['diff']:
                continue
            if not gt_already_matched[max_coverage_gt_id]:
                tp[d] = 1
                detection_matched[detection_ind] = True
                gt_already_matched[max_coverage_gt_id] = True
                detection_matched_to_which_gt[detection_ind] = max_coverage_gt_id
            else:
                fp[d] = 1
        else:
            fp[d] = 1
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    # All kinds of stats gathered together
    stats = Stats_daly_ap(flat_annotations=fgts,
            flat_detections=fdets,
            detection_matched=detection_matched,
            gt_already_matched=gt_already_matched,
            possible_matches=possible_matches_per_detection,
            iou_coverages_per_detection_ind=iou_coverages_per_detection_ind,
            detection_matched_to_which_gt=detection_matched_to_which_gt,
            sorted_inds=sorted_inds, fp=fp, tp=tp, npos=npos, rec=rec,
            prec=prec, ap=ap)
    return stats


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
        # spatial_miou_ = spatial_tube_iou(proposal_tube, gt_tube)
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
        options_tube_ap: Options_tube_ap,
            ) -> Stats_daly_ap:
    """
    We always compute stats
    """
    (iou_thresh, spatiotemporal, use_07_metric, use_diff) = \
            options_tube_ap.values()
    # Convert to flat ap-able representation
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

    stats = compute_tube_ap(
            fgts, fdets, iou_thresh,
            use_07_metric, use_diff, spatiotemporal)

    # compute via cls
    ap_computer = AP_tube_computer(spatiotemporal)
    ap_computer.prepare_computation(fgts, fdets)
    ap = ap_computer.compute_ap(
            fgts, fdets, use_diff, iou_thresh, use_07_metric)
    assert stats['ap'] == ap
    return stats


def tube_daly_ap_av(
    av_gt_tubes: AV_dict[Frametube],
    av_stubes: AV_dict[Sframetube],
    options_tube_ap: Options_tube_ap
        ):
    ap_per_cls = {}
    for action_cls in av_gt_tubes.keys():
        stats = _tube_daly_ap_v(
                av_gt_tubes[action_cls],
                av_stubes[action_cls],
                options_tube_ap)
        ap = stats['ap']
        ap_per_cls[action_cls] = ap
    return ap_per_cls


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
    options_tube_ap: Options_tube_ap,
        ) -> pd.DataFrame:
    """
    Compute ap table
    """
    ap_per_thresh = {}
    for thresh in iou_thresholds:
        options_tube_ap['iou_thresh'] = thresh
        ap_per_cls = tube_daly_ap_av(
            av_gt_tubes, av_stubes, options_tube_ap)
        ap_per_thresh[thresh] = ap_per_cls
    lst = []
    for thresh, ap_per_cls in ap_per_thresh.items():
        ap = pd.Series(ap_per_cls, name=f'{thresh:.2f}')
        ap['all'] = ap.mean()
        lst.append(ap)
    dft_ap = pd.concat(lst, axis=1)
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
        ap = compute_framebox_ap(
            o_fgts[obj_cls],
            o_fdets[obj_cls],
            iou_thresh, use_07_metric, use_diff)

        fgts = o_fgts[obj_cls]
        fdets = o_fdets[obj_cls]
        ap_computer = AP_framebox_computer()
        ap_computer.prepare_computation(fgts, fdets)
        ap_ = ap_computer.compute_ap(
                fgts, fdets, use_diff, iou_thresh, use_07_metric)

        assert ap == ap_

        ap_per_cls[obj_cls] = ap
    return ap_per_cls
