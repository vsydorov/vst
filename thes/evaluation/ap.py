import logging
import numpy as np
from abc import abstractmethod, ABC
from thes.data.dataset.external import (
        DALY_vid)
from typing import (
    Any, Dict, List, Tuple, TypedDict)
from thes.data.tubes.routines import (
        numpy_iou, temporal_IOU,
        spatial_tube_iou_v3,
        temporal_ious_where_positive)
from thes.data.tubes.types import (Frametube,)


log = logging.getLogger(__name__)


class AP_fgt(TypedDict):
    ind: Any
    obj: Any
    diff: bool


class AP_fdet(TypedDict):
    ind: Any
    obj: Any
    score: float


class AP_fgt_tube(TypedDict):
    ind: Tuple[str, int]
    obj: Frametube
    diff: bool


class AP_fdet_tube(TypedDict):
    ind: Tuple[str, int]
    obj: Frametube
    score: float


class AP_fgt_framebox(TypedDict):
    ind: Tuple[DALY_vid, int, int]  # vid, frame, anno_id
    obj: np.ndarray  # LTRD box
    diff: bool


class AP_fdet_framebox(TypedDict):
    ind: Tuple[DALY_vid, int, int]  # vid, frame, det_id
    obj: np.ndarray  # LTRD box
    score: float


Stats_daly_ap = TypedDict('Stats_daly_ap', {
    'flat_annotations': List[AP_fgt_tube],
    'flat_detections': List[AP_fdet_tube],
    'detection_matched': np.ndarray,
    'gt_already_matched': np.ndarray,
    'possible_matches': List[Dict[int, float]],
    'iou_coverages_per_detection_ind': Dict[int, List[float]],
    'detection_matched_to_which_gt': np.ndarray,
    'sorted_inds': np.ndarray,
    'fp': np.ndarray,
    'tp': np.ndarray,
    'npos': int,
    'rec': np.ndarray,
    'prec': np.ndarray,
    'ap': float
})


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t_ in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t_) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t_])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _compute_framebox_ap_old(
        fgts: List[AP_fgt_framebox],
        fdets: List[AP_fdet_framebox],
        iou_thresh: float,
        use_07_metric: bool,
        use_diff: bool,
            ) -> float:
    raise Warning('To be removed')
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


def _compute_tube_ap_old(
        fgts: List[AP_fgt_tube],
        fdets: List[AP_fdet_tube],
        iou_thresh: float,
        use_07_metric: bool,
        use_diff: bool,
        spatiotemporal: bool,
            ) -> float:
    raise Warning('To be removed')
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
            spatial_miou = \
                spatial_tube_iou_v3(gt_tube, detection_tube)
            if spatiotemporal:
                iou = temp_iou * spatial_miou
            else:
                iou = spatial_miou
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


class AP_computer(ABC):
    """
    use_07_metric: If True, will evaluate AP over 11 points, like in
        VOC2007 evaluation code
    use_diff: If True, will eval difficult proposals in the same way as
        real ones
    """
    fgts: List[AP_fgt]
    fdets: List[AP_fdet]

    def __init__(self):
        pass

    @abstractmethod
    def _get_matchable_ifgts(self, ifdet: int) -> List[int]:
        pass

    @abstractmethod
    def _compute_iou_coverages(self,
            matchable_ifgts: List[int],
            ifdet: int) -> List[float]:
        pass

    def _compute_ap(
            self,
            iou_thresh: float,
            use_diff: bool,
            use_07_metric: bool
                ) -> float:
        fgts = self.fgts
        fdets = self.fdets
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
            matchable_ifgts: List[int] = self._get_matchable_ifgts(ifdet)
            if not len(matchable_ifgts):
                fp[d] = 1
                continue
            iou_coverages: List[float] = \
                self._compute_iou_coverages(matchable_ifgts, ifdet)
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
    fgts: List[AP_fgt_framebox]
    fdets: List[AP_fdet_framebox]
    ifdet_to_ifgts: Dict[int, List[int]]

    def __init__(self,
            fgts: List[AP_fgt_framebox],
            fdets: List[AP_fdet_framebox]):
        self.fgts = fgts
        self.fdets = fdets
        self._prepare_computation()

    def _prepare_computation(self):
        fgts = self.fgts
        fdets = self.fdets
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

    def _get_matchable_ifgts(self, ifdet: int) -> List[int]:
        # Check available GTs
        share_image_ifgts: List[int] = self.ifdet_to_ifgts.get(ifdet, [])
        return share_image_ifgts

    def _compute_iou_coverages(self,
            matchable_ifgts: List[int],
            ifdet) -> List[float]:
        fgts = self.fgts
        fdets = self.fdets
        fdet = fdets[ifdet]
        # Compute IOUs
        iou_coverages: List[float] = []
        for ifgt in matchable_ifgts:
            fgt: AP_fgt_framebox = fgts[ifgt]
            iou = numpy_iou(fgt['obj'], fdet['obj'])
            iou_coverages.append(iou)
        return iou_coverages

    def compute_ap(
            self,
            iou_thresh: float,
            use_diff: bool,
            use_07_metric: bool) -> float:
        return self._compute_ap(iou_thresh, use_diff, use_07_metric)


class AP_tube_computer(AP_computer):
    fgts: List[AP_fgt_tube]
    fdets: List[AP_fdet_tube]
    _spatiotemporal: bool
    _possible_matches_per_detection: Dict[int, Dict[int, float]]

    def __init__(self,
            fgts: List[AP_fgt_tube], fdets: List[AP_fdet_tube]):
        self.fgts = fgts
        self.fdets = fdets
        self._prepare_computation()

    def _prepare_computation(self):
        fgts = self.fgts
        fdets = self.fdets
        # Group fdets belonging to same vid
        ifdet_vid_groups: Dict[DALY_vid, List[int]] = {}
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
        self._possible_matches_per_detection = ifdet_to_ifgt_tious

    def _get_matchable_ifgts(self, ifdet: int) -> List[int]:
        # Check available GTs
        gt_ids_that_overlap: Dict[int, float] = \
                self._possible_matches_per_detection.get(ifdet, {})
        return list(gt_ids_that_overlap.keys())

    def _compute_iou_coverages(self,
            matchable_ifgts: List[int],
            ifdet: int) -> List[float]:
        fgts = self.fgts
        fdets = self.fdets
        fdet = fdets[ifdet]
        gt_ids_that_overlap: Dict[int, float] = \
                self._possible_matches_per_detection.get(ifdet, {})
        # Compute IOUs
        iou_coverages: List[float] = []
        for gt_id, temp_iou in gt_ids_that_overlap.items():
            fgt: AP_fgt_tube = fgts[gt_id]
            spatial_miou = \
                    spatial_tube_iou_v3(fdet['obj'], fgt['obj'])
            if self._spatiotemporal:
                iou = temp_iou * spatial_miou
            else:
                iou = spatial_miou
            iou_coverages.append(iou)
        return iou_coverages

    def compute_ap(
            self,
            iou_thresh: float,
            spatiotemporal: bool,
            use_diff: bool,
            use_07_metric: bool) -> float:
        self._spatiotemporal = spatiotemporal
        return self._compute_ap(iou_thresh, use_diff, use_07_metric)
