import copy
import pprint
import itertools
import pandas as pd
import logging
import numpy as np
from pathlib import Path
from typing import (  # NOQA
    List, Tuple, Dict, cast, TypedDict, Set, Sequence, Optional)

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly, get_daly_split_vids, )
from thes.caffe import Nicolas_net_helper
from thes.detectron.rcnn import D2_rcnn_helper
from thes.generic_rcnn import (Ncfg_generic_rcnn_eval)
from thes.detectron.daly import (
    simplest_daly_to_datalist_v2,
    get_datalist_action_object_converter,)
from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    loadconvert_tubes_dwein, get_daly_gt_tubes,
    push_into_avdict, dtindex_filter_split,
    Objaction_dets, Frametube,
    av_stubes_above_score, AV_dict,)
from thes.data.tubes.routines import (
    score_ftubes_via_objaction_overlap_aggregation,)
from thes.data.tubes.routines import temporal_ious_NN
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df,
    compute_ap_for_avtubes_WG_as_df)
from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.tools import snippets
from thes.evaluation.ap.convert import (
    _convert_to_flat_representation,
    _compute_eligible_tubes_for_eval_weingroup
    )
from thes.evaluation.ap.core import (voc_ap)
from thes.data.tubes.routines import (spatial_tube_iou_v3)
from thes.data.dataset.external import (Action_name_daly)
from thes.data.tubes.nms import compute_nms_for_stubes
I_weingroup = Tuple[Vid_daly, int]


log = logging.getLogger(__name__)


def _compute_quick_stats(
        av_gt_tubes: AV_dict[T_dgt],
        av_stubes: AV_dict[T_dwein_scored],
        iou_thresholds: List[float]
        ) -> Dict[str, pd.DataFrame]:
    df_recall_s_nodiff = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, av_stubes, iou_thresholds, False)[0]
    df_ap_s_nodiff = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes, iou_thresholds, False, False)
    return {'recall': df_recall_s_nodiff, 'ap': df_ap_s_nodiff}


def _compute_exhaustive_evaluation_stats(
        av_gt_tubes: AV_dict[T_dgt],
        av_stubes: AV_dict[T_dwein_scored],
        iou_thresholds: List[float]
        ) -> Dict[str, pd.DataFrame]:
    df_recall_s_diff, df_recall_st_diff = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, av_stubes, iou_thresholds, True)
    df_recall_s_nodiff, df_recall_st_nodiff = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, av_stubes, iou_thresholds, False)
    df_ap_s_diff = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes, iou_thresholds, False, True)
    df_ap_s_nodiff = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes, iou_thresholds, False, False)
    df_ap_st_diff = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes, iou_thresholds, True, True)
    df_ap_st_nodiff = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes, iou_thresholds, True, False)
    dfdict = {
            'recall_s_diff': df_recall_s_diff,
            'recall_st_diff': df_recall_st_diff,
            'recall_s_nodiff': df_recall_s_nodiff,
            'recall_st_nodiff': df_recall_st_nodiff,
            'ap_s_diff': df_ap_s_diff,
            'ap_s_nodiff': df_ap_s_nodiff,
            'ap_st_diff': df_ap_st_diff,
            'ap_st_nodiff': df_ap_st_nodiff,
            }
    return dfdict


def _print_quick_evaluation_stats(dfdict):
    # Convert to str_tables
    tables = {k: snippets.df_to_table_v2((v*100).round(2))
            for k, v in dfdict.items()}
    # Print
    log.info('Recall:\n{}'.format(tables['recall']))
    log.info('Video AP:\n{}'.format(tables['ap']))


def _print_exhaustive_evaluation_stats(dfdict):
    # Convert to str_tables
    tables = {k: snippets.df_to_table_v2((v*100).round(2))
            for k, v in dfdict.items()}
    # Print
    log.info('Spatial Recall (diff):\n{}'.format(tables['recall_s_diff']))
    log.info('Spatial Recall (nodiff):\n{}'.format(tables['recall_s_nodiff']))
    log.info('Spatiotemp Recall (diff):\n{}'.format(tables['recall_st_diff']))
    log.info('Spatiotemp Recall (nodiff):\n{}'.format(tables['recall_st_nodiff']))
    log.info('Spatial AP (diff):\n{}'.format(tables['ap_s_diff']))
    log.info('Spatial AP (nodiff):\n{}'.format(tables['ap_s_nodiff']))
    log.info('Spatiotemp AP (diff):\n{}'.format(tables['ap_st_diff']))
    log.info('Spatiotemp AP (nodiff):\n{}'.format(tables['ap_s_nodiff']))


class Ncfg_dataset:
    @staticmethod
    def set_dataset_seed(cfg):
        cfg.set_deftype("""
        dataset:
            name: [~, ['daly']]
            cache_folder: [~, str]
            subset: ['test', str]
            mirror: ['uname', ~]
        seed: [42, int]
        """)

    @staticmethod
    def resolve_dataset_tubes(cf):
        dataset = Dataset_daly_ocv(cf['dataset.mirror'])
        dataset.populate_from_folder(cf['dataset.cache_folder'])
        split_label = cf['dataset.subset']
        split_vids: List[Vid_daly] = \
                get_daly_split_vids(dataset, split_label)
        dgt_tubes: Dict[I_dgt, T_dgt] = \
                get_daly_gt_tubes(dataset)
        dgt_tubes = dtindex_filter_split(dgt_tubes, split_vids)
        av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(dgt_tubes)
        return dataset, split_vids, av_gt_tubes


class Ncfg_tubes:
    @staticmethod
    def set_defcfg(cfg):
        """
        wein.leave_only_gt_keyframes:
            only keyframes that overlap with gt keyframes are left
        """
        cfg.set_deftype("""
        tubes:
            source: ['wein', ['wein', 'gt']]
            wein:
                path: [~, ~]
        """)

    @staticmethod
    def resolve_tubes(
            cf,
            av_gt_tubes: AV_dict[Frametube],
            split_vids: List[Vid_daly]
            ) -> Dict[I_dwein, T_dwein]:
        if cf['tubes.source'] == 'wein':
            raise NotImplementedError()
        elif cf['tubes.source'] == 'gt':
            raise NotImplementedError()
        else:
            raise NotImplementedError()

    @staticmethod
    def resolve_tubes_dwein(
            cf, split_vids: List[Vid_daly]
            ) -> Dict[I_dwein, T_dwein]:
        assert cf['tubes.source'] == 'wein'
        tubes_dwein: Dict[I_dwein, T_dwein] = \
                loadconvert_tubes_dwein(cf['tubes.wein.path'])
        tubes_dwein = dtindex_filter_split(tubes_dwein, split_vids)
        return tubes_dwein


class Ncfg_nicphil_rcnn:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_defaults("""
        rcnn:
            PIXEL_MEANS: [102.9801, 115.9465, 122.7717]
            TEST_SCALES: [600,]
            TEST_MAX_SIZE: 1000
        """)

    @staticmethod
    def resolve_helper(cf):
        neth = Nicolas_net_helper(cf['rcnn.PIXEL_MEANS'],
                cf['rcnn.TEST_SCALES'], cf['rcnn.TEST_MAX_SIZE'])
        return neth


class Ncfg_tube_eval:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        tube_eval:
            enabled: [True, bool]
            minscore_cutoff: [0.00, float]
            nms:
                enabled: [True, bool]
                thresh: [0.5, float]
            iou_thresholds: [[0.3, 0.5, 0.7], list]
        """)

    @staticmethod
    def evalprint_if(cf,
            av_stubes: AV_dict[T_dwein_scored],
            av_gt_tubes: AV_dict[T_dgt]):
        if not cf['tube_eval.enabled']:
            return
        av_stubes = av_stubes_above_score(
                av_stubes, cf['tube_eval.minscore_cutoff'])
        if cf['tube_eval.nms.enabled']:
            av_stubes = compute_nms_for_av_stubes(
                    av_stubes, cf['tube_eval.nms.thresh'])
        dfdict = _compute_quick_stats(
                av_gt_tubes, av_stubes, cf['tube_eval.iou_thresholds'])
        _print_quick_evaluation_stats(dfdict)

    @staticmethod
    def eval_as_df(cf,
            av_stubes: AV_dict[T_dwein_scored],
            av_gt_tubes: AV_dict[T_dgt]):
        assert cf['tube_eval.enabled']
        av_stubes = av_stubes_above_score(
                av_stubes, cf['tube_eval.minscore_cutoff'])
        if cf['tube_eval.nms.enabled']:
            av_stubes = compute_nms_for_av_stubes(
                    av_stubes, cf['tube_eval.nms.thresh'])
        dfdict = _compute_quick_stats(
                av_gt_tubes, av_stubes, cf['tube_eval.iou_thresholds'])
        return dfdict


def _recreate_actobject_datalist(dataset, split_vids):
    # /// Recreate the datalist that was used for detections
    datalist = simplest_daly_to_datalist_v2(dataset, split_vids)
    object_names, datalist_converter = \
            get_datalist_action_object_converter(dataset)
    datalist = datalist_converter(datalist)
    return datalist


def _resolve_actobjects(cf, dataset, split_vids):
    # / Assign objects to tubes
    # // Create objaction_dets in video frames
    objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]] = {}
    datalist = _recreate_actobject_datalist(dataset, split_vids)
    if cf['actobjects.source'] == 'detected':
        # /// Load detections themselves
        actobjects_evaluated = small.load_pkl(cf['actobjects.detected.path'])
        # /// Assign objactions
        for dl_item, pred_item in zip(datalist, actobjects_evaluated):
            pred_boxes = pred_item.pred_boxes.tensor.numpy()
            scores = pred_item.scores.numpy()
            pred_classes = pred_item.pred_classes.numpy()
            pred_classes = np.array([dataset.action_names[i]
                for i in pred_classes])
            detections: Objaction_dets = {
                    'pred_boxes': pred_boxes,
                    'scores': scores,
                    'pred_classes': pred_classes}
            (objactions_vf
                .setdefault(dl_item['vid'], {})
                [dl_item['video_frame_number']]) = detections
    elif cf['actobjects.source'] == 'gt':
        # /// Create fake "perfect" detections
        for dl_item in datalist:
            pred_boxes = []
            pred_classes = []
            for anno in dl_item['annotations']:
                pred_boxes.append(anno['bbox'])
                pred_classes.append(
                        dataset.action_names[anno['category_id']])
            pred_boxes = np.array(pred_boxes)
            pred_classes = np.array(pred_classes)
            scores = np.ones(len(pred_boxes))
            detections: Objaction_dets = {
                    'pred_boxes': pred_boxes,
                    'scores': scores,
                    'pred_classes': pred_classes}
            (objactions_vf
                .setdefault(dl_item['vid'], {})
                [dl_item['video_frame_number']]) = detections
    else:
        raise NotImplementedError()
    return objactions_vf


def _weingroup_nms(
        av_stubes: AV_dict[T_dwein_scored],
        nms_thresh):

    # Break-up detections into weingroups
    awg_stubes: Dict[Action_name_daly,
            Dict[I_weingroup, List[T_dwein_scored]]] = {}
    for a, v_stubes in av_stubes.items():
        for vid, stubes in v_stubes.items():
            for i, stube in enumerate(stubes):
                (vid, bunch_id, tube_id) = stube['index']
                (awg_stubes.setdefault(a, {})
                    .setdefault((vid, bunch_id), []).append(stube))

    # Perform NMS within each weingroup
    awg_stubes_nmsed: Dict[Action_name_daly,
            Dict[I_weingroup, List[T_dwein_scored]]] = {}
    for a, wg_stubes in awg_stubes.items():
        for iwg, stubes in wg_stubes.items():
            nmsed_stubes = compute_nms_for_stubes(
                    stubes, nms_thresh)
            awg_stubes_nmsed.setdefault(a, {})[iwg] = nmsed_stubes

    # Ungroup back
    av_stubes_nmsed: AV_dict[T_dwein_scored] = {}
    for a, wg_stubes in awg_stubes_nmsed.items():
        for iwg, stubes in wg_stubes.items():
            vid, bunch_id = iwg
            for stube in stubes:
                (av_stubes_nmsed
                    .setdefault(a, {})
                    .setdefault(vid, [])
                    .append(stube))
    return av_stubes_nmsed


def _get_df_daly_groundtruth(gt_tubes: Dict[I_dgt, T_dgt]):
    dgt_frange_ = np.array([
        (gt_tube['start_frame'], gt_tube['end_frame'])
        for gt_tube in gt_tubes.values()])
    df_dgt = pd.DataFrame(dgt_frange_,
            pd.MultiIndex.from_tuples(
                list(gt_tubes.keys()), names=['vid', 'act', 'id']),
            columns=['start', 'end'],)
    return df_dgt


def _get_df_weingroup_range(dwein_tubes: Dict[I_dwein, T_dwein]):
    dwt_frange_ = [
        (tube['frame_inds'].min(), tube['frame_inds'].max())
        for tube in dwein_tubes.values()]
    dwt_df = pd.DataFrame(dwt_frange_, pd.MultiIndex.from_tuples(
        dwein_tubes.keys(), names=['vid', 'gid', 'tid']),
        columns=['start', 'end'])

    dwt_df_grouped_ = dwt_df.groupby(level=[0, 1]).agg(lambda x: set(x))
    assert dwt_df_grouped_.applymap(
            lambda x: len(x) == 1).all().all(), \
            'All groups must have equal size'
    weingroup_range = dwt_df_grouped_.applymap(lambda x: list(x)[0])
    return weingroup_range


def _reindex_avd_to_dgt(
        av_gt_tubes: AV_dict[T_dgt]
        ) -> Dict[I_dgt, T_dgt]:
    dgt_tubes = {}
    for a, v_gt_tubes in av_gt_tubes.items():
        for vid, gt_tube_list in v_gt_tubes.items():
            for i, gt_tube in enumerate(gt_tube_list):
                dgt_tubes[(vid, a, i)] = gt_tube
    return dgt_tubes


def _get_weingroup_assignment(
        dgt_tubes: Dict[I_dgt, T_dgt],
        tubes_dwein: Dict[I_dwein, T_dwein],
        ) -> Tuple[List[I_weingroup], List[I_dgt]]:

    easy_dgt_tubes: Dict[I_dgt, T_dgt] = {}
    for k, v in dgt_tubes.items():
        fl = v['flags']
        diff = fl['isReflection'] or fl['isAmbiguous']
        if not diff:
            easy_dgt_tubes[k] = v
    df_gt = _get_df_daly_groundtruth(easy_dgt_tubes)
    df_weingroup_range = _get_df_weingroup_range(tubes_dwein)

    wgi = []
    gti = []
    all_vids = df_gt.index.levels[0]
    for vid in all_vids:
        wg = df_weingroup_range.loc[vid].sort_values(['start', 'end'])
        gt = df_gt.loc[vid].sort_values(['start', 'end'])
        assert len(wg) == len(gt)
        ious = temporal_ious_NN(wg.to_numpy(), gt.to_numpy())
        assert (ious >= 0.75).all()
        for gid in wg.index:
            wgi.append((vid, gid))
        for (act, ind) in gt.index:
            gti.append((vid, act, ind))
    assert len(set(wgi)) == len(wgi) == len(set(gti)) == len(gti)
    return wgi, gti


def _gather_check_all_present(gather_paths, filenames):
    # Check missing
    missing_paths = []
    for path in gather_paths:
        for filename in filenames:
            fpath = Path(path)/filename
            if not fpath.exists():
                missing_paths.append(fpath)
    if len(missing_paths):
        log.error('Some paths are MISSING:\n{}'.format(
            pprint.pformat(missing_paths)))
        return False
    return True

def _len_rescore_avstubes(av_stubes):
    norm_av_stubes = {}
    for a, v_stubes in av_stubes.items():
        for v, stubes in v_stubes.items():
            for stube in stubes:
                nstube = copy.copy(stube)
                nstube['score'] = nstube['score']/len(nstube['frame_inds'])
                (norm_av_stubes.setdefault(a, {})
                    .setdefault(v, [])
                    .append(nstube))
    return norm_av_stubes


def _compute_iou_coverages(fgts, fdets,
        det_to_eligible_gt,
        matchable_ifgts,
        ifdet, spatiotemporal) -> List[float]:
    fdet = fdets[ifdet]
    gt_ids_that_overlap = det_to_eligible_gt.get(ifdet, {})
    # Compute IOUs
    iou_coverages: List[float] = []
    for gt_id, temp_iou in gt_ids_that_overlap.items():
        fgt = fgts[gt_id]
        spatial_miou = \
                spatial_tube_iou_v3(fdet['obj'], fgt['obj'])
        if spatiotemporal:
            iou = temp_iou * spatial_miou
        else:
            iou = spatial_miou
        iou_coverages.append(iou)
    return iou_coverages


def _meanpool_avstubes(
        abstubes_to_merge: Sequence[AV_dict[T_dwein_scored]]
        ) -> AV_dict[T_dwein_scored]:
    av_stubes: AV_dict[T_dwein_scored] = {}
    for a, v_dict in abstubes_to_merge[0].items():
        for vid, stubes in v_dict.items():
            for i, stube in enumerate(stubes):
                scores = [t[a][vid][i]['score']
                        for t in abstubes_to_merge]
                new_stube = stube.copy()
                new_stube['score'] = np.mean(scores)
                (av_stubes
                    .setdefault(a, {})
                    .setdefault(vid, [])
                    .append(new_stube))
    return av_stubes


# Experiments


def assign_objactions_to_tubes(workfolder, cfg_dict, add_args):
    """
    Score tubes by assigning objactions to them and pooling the scores,
    then evaluate resulting scored tubes
    - Objactions: detecton evaluated datalist or gt objects (per frame)
    - Tubes: philippe tubes
    - Assignment: inner overlap or iou scores
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    cfg.set_deftype("""
    actobjects:
        source: ['detected', ['detected', 'gt']]
        detected:
            path: [~, ~]

    obj_to_tube:
        overlap_type: ['inner_overlap', ['inner_overlap', 'iou']]
        overlap_cutoff: [0.2, float]
        score_cutoff: [0.2, float]
    """)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()

    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    # Inputs to the assignment routine
    ftubes: Dict[I_dwein, T_dwein] = \
            Ncfg_tubes.resolve_tubes_dwein(cf, split_vids)
    objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]] = \
            _resolve_actobjects(cf, dataset, split_vids)
    # Assignment itself
    overlap_type = cf['obj_to_tube.overlap_type']
    overlap_cutoff = cf['obj_to_tube.overlap_cutoff']
    score_cutoff = cf['obj_to_tube.score_cutoff']
    av_stubes: AV_dict[T_dwein_scored] = \
        score_ftubes_via_objaction_overlap_aggregation(
            dataset, objactions_vf, ftubes, overlap_type,
            overlap_cutoff, score_cutoff)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def apply_pncaffe_rcnn_in_frames(workfolder, cfg_dict, add_args):
    """
    Apply Phil-Nic rcnn model on tube boxes to extract per-action scores
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    Ncfg_nicphil_rcnn.set_defcfg(cfg)
    Ncfg_generic_rcnn_eval.set_defcfg(cfg)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()
    # Preparation
    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            Ncfg_tubes.resolve_tubes_dwein(cf, split_vids)
    neth: Nicolas_net_helper = Ncfg_nicphil_rcnn.resolve_helper(cf)
    # Experiment logic
    if cf['demo_run.enabled']:
        Ncfg_generic_rcnn_eval.demo_run(
            cf, out, dataset, split_vids, tubes_dwein, neth)
        return
    vf_connections_dwti, vf_cls_probs = \
        Ncfg_generic_rcnn_eval.evaluate_rcnn_boxes(
            cf, out, dataset, split_vids, tubes_dwein, neth)
    small.save_pkl(out/'vf_connections_dwti.pkl', vf_connections_dwti)
    small.save_pkl(out/'vf_cls_probs.pkl', vf_cls_probs)
    agg_kind = cf['score_agg_kind']
    av_stubes = Ncfg_generic_rcnn_eval.aggregate_rcnn_scores(
            dataset, tubes_dwein, vf_connections_dwti,
            vf_cls_probs, agg_kind)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    # Post experiment
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def apply_pfadet_rcnn_in_frames(workfolder, cfg_dict, add_args):
    """
    Apply trained d2 frcnn model on tube boxes to extract per-action scores
      - We dispense with the frcnn box predictions and only use per-roi scores
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['d2.'])
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    Ncfg_generic_rcnn_eval.set_defcfg(cfg)
    cfg.set_deftype("""
    d2_rcnn:
        model: [~, ~]
        conf_thresh: [0.0, float]
    """)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')
    # Preparation
    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            Ncfg_tubes.resolve_tubes_dwein(cf, split_vids)
    neth = D2_rcnn_helper(cf, cf_add_d2, dataset, out)
    # Experiment logic
    if cf['demo_run.enabled']:
        Ncfg_generic_rcnn_eval.demo_run(
            cf, out, dataset, split_vids, tubes_dwein, neth)
        return
    vf_connections_dwti, vf_cls_probs = \
        Ncfg_generic_rcnn_eval.evaluate_rcnn_boxes(
            cf, out, dataset, split_vids, tubes_dwein, neth)
    small.save_pkl(out/'vf_connections_dwti.pkl', vf_connections_dwti)
    small.save_pkl(out/'vf_cls_probs.pkl', vf_cls_probs)
    agg_kind = cf['score_agg_kind']
    av_stubes = Ncfg_generic_rcnn_eval.aggregate_rcnn_scores(
            dataset, tubes_dwein, vf_connections_dwti,
            vf_cls_probs, agg_kind)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    # Post experiment
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def gather_reapply_agg_rcnn_avstubes(workfolder, cfg_dict, add_args):
    """
    Will apply aggregation again
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['gather.paths'])
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    cfg.set_deftype("""
    gather:
        paths: [~, ~]
    score_agg_kind: ['mean', ['mean', 'max', 'sum']]
    """)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()
    # Preparation
    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            Ncfg_tubes.resolve_tubes_dwein(cf, split_vids)
    # Experiment logic
    gather_paths = cf['gather.paths']
    if not _gather_check_all_present(gather_paths, [
            'vf_cls_probs.pkl', 'vf_connections_dwti.pkl']):
        return
    vf_connections_dwti = {}
    vf_cls_probs = {}
    for path in gather_paths:
        path = Path(path)
        vf_cls_probs_ = small.load_pkl(
                path/'vf_cls_probs.pkl')
        vf_connections_dwti_ = small.load_pkl(
                path/'vf_connections_dwti.pkl')
        assert vf_cls_probs_.keys() == vf_connections_dwti_.keys()
        vf_cls_probs.update(vf_cls_probs_)
        vf_connections_dwti.update(vf_connections_dwti_)
    small.save_pkl(out/'vf_connections_dwti.pkl', vf_connections_dwti)
    small.save_pkl(out/'vf_cls_probs.pkl', vf_cls_probs)
    agg_kind = cf['score_agg_kind']
    av_stubes: AV_dict[T_dwein_scored] = \
        Ncfg_generic_rcnn_eval.aggregate_rcnn_scores(
            dataset, tubes_dwein, vf_connections_dwti,
            vf_cls_probs, agg_kind)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    # Post experiment
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def merge_scores_avstubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tube_eval.set_defcfg(cfg)
    cfg.set_defaults("""
    tube_dict: ~
    combinations:
        enabled: False
        sizes: ~
    """)
    cf = cfg.parse()

    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    ts = {k: small.load_pkl(v) for k, v in cfg_dict['tube_dict'].items()}
    if not cf['combinations.enabled']:
        av_stubes = _meanpool_avstubes(list(ts.values()))
        small.save_pkl(out/'merged_av_stubes.pkl', av_stubes)
        log.info('All combined score:')
        Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)
        return

    sizes = cf['combinations.sizes']
    combinations = [list(itertools.combinations(
        ts.keys(), r)) for r in sizes]
    combinations = list(itertools.chain(*combinations))
    log.info('Combinations: {}'.format(combinations))

    comb_dfdicts = {}
    for comb in combinations:
        comb_name = '+'.join(comb)
        comb_fold = small.mkdir(out/comb_name)

        def compute():
            to_merge = [ts[k] for k in comb]
            av_stubes = _meanpool_avstubes(to_merge)
            small.save_pkl(comb_fold/'av_stubes.pkl', av_stubes)
            dfdict = Ncfg_tube_eval.eval_as_df(
                    cf, av_stubes, av_gt_tubes)
            return dfdict

        dfdict = small.stash2(comb_fold/'stashed_dfdict.pkl')(compute)
        comb_dfdicts[comb_name] = dfdict

    log.info('Individual results:')
    for comb_name, dfdict in comb_dfdicts.items():
        log.info(f'Results for {comb_name}:')
        _print_quick_evaluation_stats(dfdict)

    log.info('Combined tables:')
    big_= {comb: pd.concat(dfdict)
            for comb, dfdict in comb_dfdicts.items()}
    big = pd.concat(big_, axis=1)
    for stat in big.index.levels[0]:
        log.info(f'=== {stat} ===')
        for thresh in big.columns.levels[1]:
            X = (big.loc['ap']
                .loc[:, pd.IndexSlice[:, thresh]]
                .droplevel(1, axis=1))
            table = snippets.df_to_table_v2((X*100).round(2))
            log.info(f'{stat} for IOU {thresh}:\n{table}')
        log.info('\n')

def eval_avstubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    cfg.set_defaults("""
    tubes_path: ~
    """)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()

    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    tubes_to_eval: AV_dict[T_dwein_scored] = \
            small.load_pkl(cf['tubes_path'])

    # Extended version of "Ncfg_tube_eval.evalprint_if"
    nms_thresh = cf['tube_eval.nms.thresh']
    iou_thresholds: List[float] = cf['tube_eval.iou_thresholds']
    minscore_cutoff = cf['tube_eval.minscore_cutoff']

    tubes_to_eval = av_stubes_above_score(
            tubes_to_eval, minscore_cutoff)
    tubes_to_eval = \
            compute_nms_for_av_stubes(tubes_to_eval, nms_thresh)
    dfdict = _compute_exhaustive_evaluation_stats(
            av_gt_tubes, tubes_to_eval, iou_thresholds)
    _print_exhaustive_evaluation_stats(dfdict)
