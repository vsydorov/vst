import logging
import warnings
import numpy as np
from abc import abstractmethod, ABC
from thes.data.dataset.external import (Vid)
from typing import (
    Any, Dict, List, Tuple, TypedDict)
from thes.data.tubes.routines import (
        numpy_iou_11, spatial_tube_iou_v3,
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
    ind: Tuple[Vid, int]
    obj: Frametube
    diff: bool


class AP_fdet_tube(TypedDict):
    ind: Tuple[Vid, int]
    obj: Frametube
    score: float


class AP_fgt_framebox(TypedDict):
    ind: Tuple[Vid, int, int]  # vid, frame, anno_id
    obj: np.ndarray  # LTRD box
    diff: bool


class AP_fdet_framebox(TypedDict):
    ind: Tuple[Vid, int, int]  # vid, frame, det_id
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
        if len(fgts) == 0:
            raise RuntimeError("Can't have 0 groundtruth")
        if len(fdets) == 0:
            warnings.warn('No detections when computing ap')
            return np.nan
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
        ifdet_groups: Dict[Tuple[Vid, int], List[int]] = {}
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
            ifdet: int) -> List[float]:
        fgts = self.fgts
        fdets = self.fdets
        fdet = fdets[ifdet]
        # Compute IOUs
        iou_coverages: List[float] = []
        for ifgt in matchable_ifgts:
            fgt: AP_fgt_framebox = fgts[ifgt]
            iou = numpy_iou_11(fgt['obj'], fdet['obj'])
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
    _det_to_eligible_gt: Dict[int, Dict[int, float]]

    def __init__(self,
            fgts: List[AP_fgt_tube],
            fdets: List[AP_fdet_tube],
            det_to_eligible_gt: Dict[int, Dict[int, float]]):
        self.fgts = fgts
        self.fdets = fdets
        self._det_to_eligible_gt = det_to_eligible_gt

    def _get_matchable_ifgts(self, ifdet: int) -> List[int]:
        # Check available GTs
        gt_ids_that_overlap: Dict[int, float] = \
                self._det_to_eligible_gt.get(ifdet, {})
        return list(gt_ids_that_overlap.keys())

    def _compute_iou_coverages(self,
            matchable_ifgts: List[int],
            ifdet: int) -> List[float]:
        fgts = self.fgts
        fdets = self.fdets
        fdet = fdets[ifdet]
        gt_ids_that_overlap: Dict[int, float] = \
                self._det_to_eligible_gt.get(ifdet, {})
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
