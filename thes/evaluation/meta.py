import copy
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import (
    accuracy_score, roc_auc_score)
from typing import (  # NOQA
    Dict, Any, List, Optional, Tuple, TypedDict, Set, cast)

from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.tubes.types import (
    Box_connections_dwti,
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    Frametube_scored,
    Tube_daly_wein_as_provided,
    get_daly_gt_tubes, push_into_avdict,
    AV_dict, loadconvert_tubes_dwein,
    av_stubes_above_score)
from thes.data.tubes.routines import (
    group_tubes_on_frame_level,
    score_ftubes_via_objaction_overlap_aggregation,
    create_kinda_objaction_struct,
    qload_synthetic_tube_labels
    )
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)

from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)

log = logging.getLogger(__name__)


def keyframe_cls_scores(
        softmaxes: np.ndarray,
        gt_ids: np.ndarray):
    preds = np.argmax(softmaxes, axis=1)
    kf_acc = accuracy_score(gt_ids, preds)
    try:
        kf_roc_auc = roc_auc_score(gt_ids,
                softmaxes, multi_class='ovr')
    except ValueError as err:
        log.warning(f'auc could not be computed, Caught "{err}"')
        kf_roc_auc = np.NaN
    return kf_acc, kf_roc_auc


def cheating_tube_scoring(
        softmaxes: np.ndarray,
        keyframes,
        tubes_dwein: Dict[I_dwein, T_dwein],
        dataset: Dataset_daly_ocv,
        ) -> AV_dict[T_dwein_scored]:
    """
    Will record overlaps of keyframes with philtubes and assign scores
    based on this. Cheating because we use knowledge of GT keyframes.
    """
    objactions_vf = create_kinda_objaction_struct(
            dataset, keyframes, softmaxes)
    # Assigning scores based on intersections
    av_stubes: AV_dict[T_dwein_scored] = \
        score_ftubes_via_objaction_overlap_aggregation(
            dataset, objactions_vf, tubes_dwein, 'iou',
            0.1, 0.0, enable_tqdm=False)
    return av_stubes


def quick_tube_eval(
        av_stubes: AV_dict[T_dwein_scored],
        tubes_dgt: Dict[I_dgt, T_dgt],
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick evaluation of daly tubes with default parameters
    """
    av_stubes_ = av_stubes_above_score(av_stubes, 0.0)
    av_stubes_ = compute_nms_for_av_stubes(av_stubes_, 0.3)
    iou_thresholds = [.3, .5, .7]
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(tubes_dgt)
    df_recall = compute_recall_for_avtubes_as_dfs(
        av_gt_tubes, av_stubes_, iou_thresholds, False)[0]
    df_ap = compute_ap_for_avtubes_as_df(
        av_gt_tubes, av_stubes_, iou_thresholds, False, False)
    return df_ap, df_recall


def assign_scorefield(av_stubes, score_field):
    av_stubes = copy.deepcopy(av_stubes)
    for a, v_stubes in av_stubes.items():
        for v, stubes in v_stubes.items():
            for stube in stubes:
                stube['score'] = stube[score_field]
    return cast(AV_dict[Frametube_scored], av_stubes)


def assign_scores_to_dwt_roipooled(
        tubes_dwein: Dict[I_dwein, T_dwein],
        tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided],
        tube_softmaxes_eval_nobg: Dict[I_dwein, np.ndarray],
        dataset: Dataset_daly_ocv) -> AV_dict[Dict]:
    """
    Given per-tube scores obtained with roipooled detector,
        assign scores to the DWEIN tubes
    """
    # Universal detector experiments
    av_stubes_with_scores: AV_dict[Dict] = {}
    for dwt_index, tube in tubes_dwein.items():
        softmaxes = tube_softmaxes_eval_nobg[dwt_index]
        scores = softmaxes.mean(axis=0)
        hscores = tubes_dwein_prov[dwt_index]['hscores']
        iscores = tubes_dwein_prov[dwt_index]['iscores']
        (vid, bunch_id, tube_id) = dwt_index
        for action_name, score in zip(dataset.action_names, scores):
            stube = cast(Dict, tube.copy())
            stube['hscore'] = hscores.mean()
            stube['iscore'] = np.nanmean(iscores)
            stube['box_det_score'] = score
            stube['box_nonbg_score'] = scores.sum()
            (av_stubes_with_scores
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes_with_scores


def assign_scores_to_dwt_fullframe(
        tubes_dwein: Dict[I_dwein, T_dwein],
        tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided],
        frame_scores: Dict[Tuple[Vid_daly, int], np.ndarray],
        dataset: Dataset_daly_ocv) -> AV_dict[Dict]:
    """
    Given per-frame scores obtaiend with fullframe classifier,
        assign scores to the DWEIN tubes
    """
    av_stubes_with_scores: AV_dict[Dict] = {}
    for dwt_index, tube in tubes_dwein.items():
        (vid, bunch_id, tube_id) = dwt_index
        # Human score from dwein tubes
        hscores = tubes_dwein_prov[dwt_index]['hscores']
        iscores = tubes_dwein_prov[dwt_index]['iscores']
        # Aggregated frame score
        fscores_for_tube_ = []
        for frame_ind in tube['frame_inds']:
            fscore = frame_scores.get((vid, frame_ind))
            if fscore is not None:
                fscores_for_tube_.append(fscore)
        fscores_for_tube = np.vstack(fscores_for_tube_)
        for ia, action_name in enumerate(dataset.action_names):
            stube = cast(Dict, tube.copy())
            stube['hscore'] = hscores.mean()
            stube['iscore'] = np.nanmean(iscores)
            stube['frame_cls_score'] = fscores_for_tube.mean(0)[ia]
            stube['hscore*frame_cls_score'] = \
                    stube['hscore'] * stube['frame_cls_score']
            (av_stubes_with_scores
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes_with_scores
