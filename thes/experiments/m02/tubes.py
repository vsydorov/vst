import copy
import pprint
import itertools
import pandas as pd
import warnings
import logging
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import (
        List, Tuple, Dict, cast, TypedDict, Set, Sequence, Optional)
from types import MethodType

import torch
from torch.nn import functional as F

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as d2_transforms
from detectron2.structures import Boxes, Instances

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.caffe import (Nicolas_net_helper)
from thes.detectron.cfg import (
    set_detectron_cfg_base, set_detectron_cfg_test,)
from thes.detectron.externals import (simple_d2_setup,)
from thes.detectron.daly import (
    get_daly_split_vids, simplest_daly_to_datalist_v2,
    get_datalist_action_object_converter,)
from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    loadconvert_tubes_dwein, get_daly_gt_tubes,
    push_into_avdict, dtindex_filter_split,
    Objaction_dets, Frametube,
    av_filter_split, av_stubes_above_score,
    AV_dict,)
from thes.data.tubes.routines import (
    score_ftubes_via_objaction_overlap_aggregation,)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df,
    compute_ap_for_avtubes_WG_as_df)
from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.tools import snippets


log = logging.getLogger(__name__)


class Box_connections_dwti(TypedDict):
    vid: Vid_daly
    frame_ind: int
    dwti_sources: List[I_dwein]  # N
    boxes: List[np.ndarray]  # N, 4


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
        seed: [42, int]
        """)

    @staticmethod
    def resolve_dataset_tubes(cf):
        dataset = Dataset_daly_ocv()
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


class Ncfg_generic_rcnn_eval:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        scoring:
            keyframes_only: [True, bool]
            agg_kind: ['mean', ['mean', 'max']]
        demo_run:
            enabled: [False, bool]
            N: [50, int]
            seed: [0, int]
        compute:
            save_period: ['::10', str]
            split:
                enabled: [False, bool]
                chunk: [0, "VALUE >= 0"]
                total: [1, int]
        """)

    @classmethod
    def _get_daly_keyframes(
            cls, dataset: Dataset_daly_ocv, split_vids
            ) -> Dict[Vid_daly, np.ndarray]:
        to_cover_: Dict[Vid_daly, Set] = {}
        for vid in split_vids:
            v = dataset.videos_ocv[vid]
            for action_name, instances in v['instances'].items():
                for ins_ind, instance in enumerate(instances):
                    frames = [kf['frame'] for kf in instance['keyframes']]
                    to_cover_[vid] = \
                            to_cover_.get(vid, set()) | set(list(frames))
        frames_to_cover = \
                {k: np.array(sorted(v)) for k, v in to_cover_.items()}
        return frames_to_cover

    @classmethod
    def _define_boxes_to_evaluate(cls, cf, dataset, split_vids, tubes_dwein):
        if cf['scoring.keyframes_only']:
            # Cover only keyframes when evaluating dwti tubes
            frames_to_cover = cls.get_daly_keyframes(dataset, split_vids)
        else:
            frames_to_cover = None
        vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]]
        vf_connections_dwti = \
                prepare_ftube_box_computations(tubes_dwein, frames_to_cover)
        return vf_connections_dwti

    @classmethod
    def _demovis_apply(cls,
            vfold, neth, dataset: Dataset_daly_ocv, vf_connections_dwti):
        nicolas_labels = ['background', ] + cast(List[str], dataset.action_names)
        for vid, f_connections_dwti in tqdm(
                vf_connections_dwti.items(), 'nicphil_demovis'):
            video_path = dataset.videos[vid]['path']
            finds = list(f_connections_dwti)
            with vt_cv.video_capture_open(video_path) as vcap:
                frames_u8 = vt_cv.video_sample(
                        vcap, finds, debug_filename=video_path)

            video_fold = small.mkdir(vfold/f'vid{vid}')

            for find, frame_BGR in zip(finds, frames_u8):
                connections_dwti = f_connections_dwti[find]
                boxes = connections_dwti['boxes']
                box_cls_probs = neth.score_boxes(frame_BGR, boxes)  # N, (bcg+10)
                # Draw and print
                txt_output = []
                image = frame_BGR.copy()
                for i, cls_probs in enumerate(box_cls_probs):
                    box = boxes[i]
                    best_score_id = np.argmax(cls_probs)
                    best_score = cls_probs[best_score_id]
                    best_nicolas_label = nicolas_labels[best_score_id]
                    snippets.cv_put_box_with_text(image, box,
                        text='{} {} {:.2f}'.format(
                            i, best_nicolas_label, best_score))
                    line = (' '.join([f'{y}: {x:.3f}'
                        for x, y in zip(cls_probs, nicolas_labels)])
                        + str(box))
                    txt_output.append(line)
                cv2.imwrite(str(
                    video_fold/'Fr{:05d}.png'.format(find)), image)
                with (video_fold/f'Fr{find:05d}_scores.txt').open('w') as f:
                    f.write('\n'.join(txt_output))

    @classmethod
    def _perform_split(cls, cf, vf_connections_dwti):
        # Reduce keys according to split
        vids_to_eval = list(vf_connections_dwti.keys())
        weights_dict = {k: len(v) for k, v in vf_connections_dwti.items()}
        weights = np.array(list(weights_dict.values()))
        cc, ct = (cf['compute.split.chunk'], cf['compute.split.total'])
        vids_split = snippets.weighted_array_split(
                vids_to_eval, weights, ct)
        ktw = dict(zip(vids_to_eval, weights))
        weights_split = [np.sum([ktw[vid] for vid in vids])
                for vids in vids_split]
        chunk_vids = vids_split[cc]
        log.info(f'Quick split stats [{cc,ct=}]: ''Vids(frames): {}({}) -> {}({})'.format(
            len(vids_to_eval), np.sum(weights),
            len(chunk_vids), weights_split[cc]))
        log.debug(f'Full stats [{cc,ct=}]:\n'
                f'vids_split={pprint.pformat(vids_split)}\n'
                f'{weights_split=}\n'
                f'{chunk_vids=}\n'
                f'{weights_split[cc]=}')
        chunk_vf_connections_dwti = {vid: vf_connections_dwti[vid]
                for vid in chunk_vids}
        return chunk_vf_connections_dwti

    @classmethod
    def _aggregate_scores(cls,
            dataset, tubes_dwein,
            vf_connections_dwti,
            vf_cls_probs,
            agg_kind):
        # Pretty clear we'll be summing up the scores anyway
        assert vf_connections_dwti.keys() == vf_cls_probs.keys()

        if agg_kind == 'mean':
            ftube_sum = {}
            ftube_counts = {}
            for vid, f_cls_probs in vf_cls_probs.items():
                for f, cls_probs in f_cls_probs.items():
                    dwtis = vf_connections_dwti[vid][f]['dwti_sources']
                    for dwti, prob in zip(dwtis, cls_probs):
                        ftube_sum[dwti] = \
                                ftube_sum.get(dwti, np.zeros(11)) + prob
                        ftube_counts[dwti] = \
                                ftube_counts.get(dwti, 0) + 1
            ftube_scores = {k: v/ftube_counts[k]
                    for k, v in ftube_sum.items()}
        elif agg_kind == 'max':
            ftube_scores = {}
            for vid, f_cls_probs in vf_cls_probs.items():
                for f, cls_probs in f_cls_probs.items():
                    dwtis = vf_connections_dwti[vid][f]['dwti_sources']
                    for dwti, prob in zip(dwtis, cls_probs):
                        ftube_scores[dwti] = \
                            np.maximum(ftube_scores.get(
                                dwti, np.zeros(11)), prob)
        else:
            raise NotImplementedError()

        # Create av_stubes
        av_stubes: AV_dict[T_dwein_scored] = {}
        for dwt_index, scores in ftube_scores.items():
            (vid, bunch_id, tube_id) = dwt_index
            for action_name, score in zip(
                    dataset.action_names, scores[1:]):
                stube = tubes_dwein[dwt_index].copy()
                stube = cast(T_dwein_scored, stube)
                stube['score'] = score
                (av_stubes
                        .setdefault(action_name, {})
                        .setdefault(vid, []).append(stube))
        return av_stubes

    @classmethod
    def _simple_gpu_compute(
            cls, out, dataset, neth, vf_connections_dwti
            ) -> Dict[Vid_daly, Dict[int, np.ndarray]]:
        """Progress saved on video-level scale"""
        def isaver_eval_func(vid):
            f_connections_dwti = vf_connections_dwti[vid]
            video_path = dataset.videos[vid]['path']
            finds = list(f_connections_dwti)
            with vt_cv.video_capture_open(video_path) as vcap:
                frames_u8 = vt_cv.video_sample(
                        vcap, finds, debug_filename=video_path)
            f_cls_probs = {}
            for find, frame_BGR in zip(finds, frames_u8):
                connections_dwti = f_connections_dwti[find]
                boxes = connections_dwti['boxes']
                cls_probs = neth.score_boxes(frame_BGR, boxes)  # N, (bcg+10)
                f_cls_probs[find] = cls_probs
            return f_cls_probs
        vids_to_eval = list(vf_connections_dwti.keys())
        isaver = snippets.Simple_isaver(
                small.mkdir(out/'isave_rcnn_vid_eval'),
                vids_to_eval, isaver_eval_func,
                '::10', 120)
        isaver_items = isaver.run()
        vf_cls_probs: Dict[Vid_daly, Dict[int, np.ndarray]]
        vf_cls_probs = dict(zip(vids_to_eval, isaver_items))
        return vf_cls_probs

    @classmethod
    def _involved_gpu_compute(
            cls, out, dataset, neth,
            vf_connections_dwti,
            size_video_chunk,
            ) -> Dict[Vid_daly, Dict[int, np.ndarray]]:
        frame_chunks = []
        vids_to_eval = list(vf_connections_dwti.keys())
        for vid in vids_to_eval:
            f_connections_dwti = vf_connections_dwti[vid]
            finds = np.array(list(f_connections_dwti))
            finds_split = snippets.leqn_split(
                    finds, size_video_chunk)
            for subset_finds in finds_split:
                frame_chunks.append((vid, subset_finds))

        def isaver_eval_func(frame_chunk):
            vid, finds = frame_chunk
            f_connections_dwti = vf_connections_dwti[vid]
            video_path = dataset.videos[vid]['path']
            with vt_cv.video_capture_open(video_path) as vcap:
                frames_u8 = vt_cv.video_sample(
                        vcap, finds, debug_filename=video_path)
            f_cls_probs = {}
            for find, frame_BGR in zip(finds, frames_u8):
                connections_dwti = f_connections_dwti[find]
                boxes = connections_dwti['boxes']
                cls_probs = neth.score_boxes(
                        frame_BGR, boxes)  # N, (bcg+10)
                f_cls_probs[find] = cls_probs
            return f_cls_probs
        isaver = snippets.Simple_isaver(
                small.mkdir(out/'isave_rcnn_vid_eval'),
                frame_chunks, isaver_eval_func,
                save_interval=60,
                log_interval=300)
        isaver_items = isaver.run()
        vf_cls_probs: Dict[Vid_daly, Dict[int, np.ndarray]]
        vf_cls_probs = {}
        for (vid, subset_finds), f_cls_probs in zip(
                frame_chunks, isaver_items):
            vf_cls_probs.setdefault(vid, {}).update(f_cls_probs)
        return vf_cls_probs

    @classmethod
    def score_tubes(
            cls, cf, out,
            dataset: Dataset_daly_ocv,
            split_vids,
            tubes_dwein: Dict[I_dwein, T_dwein],
            neth) -> AV_dict[T_dwein_scored]:
        """
        Logic behind simple "evaluate boxes" experiment
        """
        vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]] = \
            cls._define_boxes_to_evaluate(cf, dataset, split_vids, tubes_dwein)

        if cf['compute.split.enabled']:
            vf_connections_dwti = cls._perform_split(
                    cf, vf_connections_dwti)

        if cf['scoring.keyframes_only']:
            vf_cls_probs = cls._simple_gpu_compute(
                out, dataset, neth, vf_connections_dwti)
        else:
            size_video_chunk = 300
            vf_cls_probs = cls._involved_gpu_compute(
                out, dataset, neth, vf_connections_dwti, size_video_chunk)

        agg_kind = cf['scoring.agg_kind']
        av_stubes = cls._aggregate_scores(
                dataset, tubes_dwein, vf_connections_dwti,
                vf_cls_probs, agg_kind)

        small.save_pkl(out/'vf_connections_dwti.pkl', vf_connections_dwti)
        small.save_pkl(out/'vf_cls_probs.pkl', vf_cls_probs)
        small.save_pkl(out/'av_stubes.pkl', av_stubes)
        return av_stubes

    @classmethod
    def demo_run(cls, cf, out, dataset, split_vids, tubes_dwein, neth):
        vf_connections_dwti = cls._define_boxes_to_evaluate(
                cf, dataset, split_vids, tubes_dwein)
        vf_connections_dwti = sample_dict(
            vf_connections_dwti, N=5, NP_SEED=0)
        vfold = small.mkdir(out/'demovis')
        cls._demovis_apply(vfold, neth, dataset, vf_connections_dwti)


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


def sample_dict(dct: Dict, N=10, NP_SEED=0) -> Dict:
    np_rstate = np.random.RandomState(NP_SEED)
    prm_key_indices = np_rstate.permutation(np.arange(len(dct)))
    key_list = list(dct.keys())
    some_keys = [key_list[i] for i in prm_key_indices[:N]]
    some_tubes = {k: dct[k] for k in some_keys}
    return some_tubes


def _recreate_actobject_datalist(dataset, split_vids):
    # /// Recreate the datalist that was used for detections
    datalist = simplest_daly_to_datalist_v2(dataset, split_vids)
    object_names, datalist_converter = \
            get_datalist_action_object_converter(dataset)
    datalist = datalist_converter(datalist)
    return datalist


def prepare_ftube_box_computations(
        tubes_dwein: Dict[I_dwein, T_dwein],
        frames_to_cover: Optional[Dict[Vid_daly, np.ndarray]]
        ) -> Dict[Vid_daly, Dict[int, Box_connections_dwti]]:
    """
    Assign boxes (and keep connections to the original ftubes)
    If frames_to_cover passed - compute only in those frames
    """
    vf_connections_dwti_list: Dict[Vid_daly, Dict[int,
        List[Tuple[I_dwein, np.ndarray]]]] = {}
    for dwt_index, tube in tubes_dwein.items():
        (vid, bunch_id, tube_id) = dwt_index
        tube_finds = tube['frame_inds']
        if frames_to_cover is None:
            common_finds = tube_finds
            good_tube_boxes = tube['boxes']
        else:
            good_finds = frames_to_cover[vid]
            common_finds, comm1, comm2 = np.intersect1d(
                tube_finds, good_finds,
                assume_unique=True, return_indices=True)
            if len(common_finds) == 0:
                continue
            good_tube_boxes = tube['boxes'][comm1]
        for find, box in zip(common_finds, good_tube_boxes):
            (vf_connections_dwti_list
                .setdefault(vid, {})
                .setdefault(find, []).append((dwt_index, box)))
    # Prettify
    vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]] = {}
    for vid, f_connections_dwti_list in vf_connections_dwti_list.items():
        for find, connections_dwti_list in f_connections_dwti_list.items():
            lsources, lboxes = zip(*connections_dwti_list)
            boxes = np.vstack(lboxes)
            bcs: Box_connections_dwti = {
                'vid': vid,
                'frame_ind': find,
                'dwti_sources': lsources,
                'boxes': boxes
            }
            vf_connections_dwti.setdefault(vid, {})[find] = bcs
    return vf_connections_dwti


def _predict_rcnn_given_box_resized_proposals(
        box4, frame_u8, transform_gen, model):

    o_height, o_width = frame_u8.shape[:2]
    got_transform = transform_gen.get_transform(frame_u8)

    # Transform image
    image = got_transform.apply_image(frame_u8)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    imshape = tuple(image.shape[1:3])

    # / Transform box
    assert box4.shape == (4,)
    boxes_unscaled = box4[None]
    t_boxes = torch.as_tensor(boxes_unscaled.astype("float32"))
    transformed_t_boxes = got_transform.apply_box(t_boxes)
    # // Proposals w.r.t transformed imagesize
    proposal = Instances(imshape)
    tb_boxes = Boxes(transformed_t_boxes)
    proposal.proposal_boxes = tb_boxes

    inputs = {
            "image": image,
            "proposals": proposal,
            "height": o_height,
            "width": o_width}

    with torch.no_grad():
        predictions = model([inputs])[0]
    return predictions


def genrcnn_rcnn_roiscores_forward(self, batched_inputs):
    """
    Replacing detectron2/detectron2/modeling/meta_arch/rcnn.py (GeneralizedRCNN.forward)
    """
    assert not self.training
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    assert "proposals" in batched_inputs[0]
    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
    del images
    # Borrowed from detectron2/detectron2/modeling/roi_heads/roi_heads.py (Res5ROIHeads.forward)
    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = self.roi_heads._shared_roi_transform(
        [features[f] for f in self.roi_heads.in_features], proposal_boxes
    )
    feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
    pred_class_logits, pred_proposal_deltas = \
            self.roi_heads.box_predictor(feature_pooled)
    pred_softmax = F.softmax(pred_class_logits, dim=-1)
    return pred_softmax


class D2_rcnn_helper(object):
    def __init__(self, cf, cf_add_d2, dataset, out):
        num_classes = len(dataset.action_names)
        TEST_DATASET_NAME = 'daly_objaction_test'

        # / Define d2 conf
        d2_output_dir = str(small.mkdir(out/'d2_output'))
        d_cfg = set_detectron_cfg_base(
                d2_output_dir, num_classes, cf['seed'])
        d_cfg = set_detectron_cfg_test(
                d_cfg, TEST_DATASET_NAME,
                cf['d2_rcnn.model'], cf['d2_rcnn.conf_thresh'], cf_add_d2,
                freeze=False)
        d_cfg.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
        d_cfg.freeze()

        # / Start d2
        simple_d2_setup(d_cfg)

        # Predictor without proposal generator
        model = build_model(d_cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)

        checkpointer.load(d_cfg.MODEL.WEIGHTS)
        MIN_SIZE_TEST = d_cfg.INPUT.MIN_SIZE_TEST
        MAX_SIZE_TEST = d_cfg.INPUT.MAX_SIZE_TEST
        transform_gen = d2_transforms.ResizeShortestEdge(
            [MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST)

        # Instance monkeypatching
        # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module/50600307#50600307
        model.forward = MethodType(genrcnn_rcnn_roiscores_forward, model)

        self.d_cfg = d_cfg
        self.rcnn_roiscores_model = model
        self.cpu_device = torch.device("cpu")
        self.transform_gen = transform_gen

    def score_boxes(self, frame_BGR, boxes) -> np.ndarray:
        o_height, o_width = frame_BGR.shape[:2]
        got_transform = self.transform_gen.get_transform(frame_BGR)
        # Transform image
        image = got_transform.apply_image(frame_BGR)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        imshape = tuple(image.shape[1:3])
        # Transform box
        t_boxes = torch.as_tensor(boxes.astype("float32"))
        transformed_t_boxes = got_transform.apply_box(t_boxes)
        # Proposals w.r.t transformed imagesize
        proposal = Instances(imshape)
        tb_boxes = Boxes(transformed_t_boxes)
        proposal.proposal_boxes = tb_boxes
        inputs = {
                "image": image,
                "proposals": proposal,
                "height": o_height,
                "width": o_width}
        with torch.no_grad():
            pred_softmax = self.rcnn_roiscores_model([inputs])
        X = pred_softmax.to(self.cpu_device).numpy()
        # To conform to caffe style put background cls at 0th position
        X_caffelike = np.c_[X[:, -1:], X[:, :-1]]
        return X_caffelike

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


from thes.data.dataset.external import (
    Action_name_daly)
from thes.data.tubes.nms import compute_nms_for_stubes
I_weingroup = Tuple[Vid_daly, int]


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


from thes.data.tubes.routines import temporal_ious_NN


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


def _gather_check_all_present(gather_paths):
    # Check missing
    missing_paths = []
    for path in gather_paths:
        path = Path(path)
        if not path.exists():
            missing_paths.append(path)
    if len(missing_paths):
        log.info('Some paths are MISSING:\n{}'.format(
            pprint.pformat(missing_paths)))
        return False
    else:
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


def _vis_scoresorted_tubes(out, dataset, wnms_av_stubes):
    action = 'Drinking'
    vfold = small.mkdir(out/action)
    v_stubes = wnms_av_stubes[action]
    flat_tubes = []
    for vid, stubes in v_stubes.items():
        for i_stube, stube in enumerate(stubes):
            flat_tubes.append({'tube': stube, 'ind': (vid, i_stube)})
    sorted_flat_tubes = sorted(flat_tubes,
            key=lambda x: x['tube']['score'], reverse=True)

    for i_sorted, flat_tube in enumerate(sorted_flat_tubes):
        vid, i_stube = flat_tube['ind']
        tube = flat_tube['tube']
        score = tube['score']
        sf, ef = tube['start_frame'], tube['end_frame']
        frame_inds = tube['frame_inds']
        video_fold = small.mkdir(vfold/f'{i_sorted:04d}_vid{vid}_{sf}_to_{ef}_score{score:02f}')
        video_path = dataset.videos[vid]['path']

        # Extract
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)
        # Draw
        drawn_frames_u8 = []
        for i, (find, frame_BGR) in enumerate(zip(frame_inds, frames_u8)):
            image = frame_BGR.copy()
            box = tube['boxes'][i]
            snippets.cv_put_box_with_text(image, box,
                text='{} {} {:.2f}'.format(
                    i, action, score))
            drawn_frames_u8.append(image)

        # # Save as images
        # for find, image in zip(frame_inds, drawn_frames_u8):
        #     cv2.imwrite(str(
        #         video_fold/'Fr{:05d}.png'.format(find)), image)

        # Save as video
        snippets.qsave_video(video_fold/'overlaid.mp4', drawn_frames_u8)


from thes.evaluation.ap.convert import (
        _convert_to_flat_representation,
        _compute_eligible_tubes_for_eval_weingroup
        )
from thes.evaluation.ap.core import (voc_ap)
from thes.data.tubes.routines import (
        spatial_tube_iou_v3)


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

    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            Ncfg_tubes.resolve_tubes_dwein(cf, split_vids)
    neth: Nicolas_net_helper = Ncfg_nicphil_rcnn.resolve_helper(cf)

    if cf['demo_run.enabled']:
        Ncfg_generic_rcnn_eval.demo_run(
            cf, out, dataset, split_vids, tubes_dwein, neth)
        return
    av_stubes = Ncfg_generic_rcnn_eval.score_tubes(
            cf, out, dataset, split_vids, tubes_dwein, neth)
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def apply_pfadet_rcnn_in_frames(workfolder, cfg_dict, add_args):
    """
    Apply trained d2 frcnn model on tube boxes to extract per-action scores
      - We dispense with the frcnn box predictions and only use per-roi scores
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_dataset.set_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    cfg.set_deftype("""
    d2_rcnn:
        model: [~, ~]
        conf_thresh: [0.0, float]
    """)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()
    cf_add_d2 = cfg.without_prefix('d2.')

    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            Ncfg_tubes.resolve_tubes_dwein(cf, split_vids)
    neth = D2_rcnn_helper(cf, cf_add_d2, dataset, out)

    raise NotImplementedError()
    # av_stubes = Ncfg_nicphil_rcnn.score_tubes(
    #         cf, out, dataset, split_vids, tubes_dwein, neth)
    # small.save_pkl(out/'av_stubes.pkl', av_stubes)
    # Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def gather_avstubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    cfg.set_deftype("""
    gather:
        kind: ['explicit', ['explicit', 'rcnn']]
        paths: [~, ~]
    """)
    Ncfg_tube_eval.set_defcfg(cfg)
    cf = cfg.parse()

    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)

    # Experiment logic
    gather_paths = cf['gather.paths']
    if not _gather_check_all_present(gather_paths):
        return

    av_stubes: AV_dict[T_dwein_scored]
    if cf['gather.kind'] == 'explicit':
        av_stubes = {}
        for path in gather_paths:
            av_stubes_ = small.load_pkl(path)
            for a, v_stubes in av_stubes_.items():
                for v, stubes in v_stubes.items():
                    av_stubes.setdefault(a, {})[v] = stubes
    elif cf['gather.kind'] == 'rcnn':
        av_stubes = {}
        vf_connections_dwti = {}
        vf_cls_probs = {}
        for path in gather_paths:
            path = Path(path)
            av_stubes_ = small.load_pkl(path)
            vf_cls_probs_ = small.load_pkl(
                    path.parent/'vf_cls_probs.pkl')
            vf_connections_dwti_ = small.load_pkl(
                    path.parent/'vf_connections_dwti.pkl')
            for a, v_stubes in av_stubes_.items():
                assert (v_stubes.keys() == vf_cls_probs_.keys()
                        == vf_connections_dwti_.keys())
                for v, stubes in v_stubes.items():
                    av_stubes.setdefault(a, {})[v] = stubes

            vf_cls_probs.update(vf_cls_probs_)
            vf_connections_dwti.update(vf_connections_dwti_)
        small.save_pkl(out/'vf_connections_dwti.pkl', vf_connections_dwti)
        small.save_pkl(out/'vf_cls_probs.pkl', vf_cls_probs)
    else:
        raise NotImplementedError()

    small.save_pkl(out/'av_stubes.pkl', av_stubes)
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
    for comb, dfdict in comb_dfdicts.items():
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


def vis_stubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    Ncfg_dataset.set_dataset_seed(cfg)
    cfg.set_defaults("""
    paths:
        av_stubes: ~
        vf_connections_dwti: ~
        vf_cls_probs: ~
    """)
    cf = cfg.parse()

    dataset, split_vids, av_gt_tubes = \
            Ncfg_dataset.resolve_dataset_tubes(cf)
    av_stubes: AV_dict[T_dwein_scored] = \
            small.load_pkl(cf['paths.av_stubes'])
    vf_connections_dwti: Dict[Vid_daly, Dict[int, Box_connections_dwti]] = \
            small.load_pkl(cf['paths.vf_connections_dwti'])
    vf_cls_probs: Dict[Vid_daly, Dict[int, np.ndarray]] = \
            small.load_pkl(cf['paths.vf_cls_probs'])

    # WGI GTI
    dgt_tubes: Dict[I_dgt, T_dgt] = _reindex_avd_to_dgt(av_gt_tubes)
    tubes_dwein: Dict[I_dwein, T_dwein] = loadconvert_tubes_dwein('/home/vsydorov/projects/deployed/2019_12_Thesis/dervo/thes/gpuhost7/2020_03_31/10_clean_inputs/20_wein_tubes/thes.gpuhost7.2020_03_31.10_clean_inputs.20_wein_tubes/194e3a301c202bc5e818dca26953ddb268aa98b3/out/extracted_tubes.pkl')
    tubes_dwein = dtindex_filter_split(tubes_dwein, split_vids)
    wgi, gti = _get_weingroup_assignment(dgt_tubes, tubes_dwein)
    wgi_to_gti: Dict[I_weingroup, I_dgt] = dict(zip(wgi, gti))

    nms_thresh = 0.5
    wnms_av_stubes = \
            (small.stash2(out/'_temp_wnms_norm_av_stubes.pkl')
            (_weingroup_nms, av_stubes, nms_thresh))

    iou_thresholds = [0.3, 0.5]
    co_wnms_av_stubes = av_stubes_above_score(
            wnms_av_stubes, 0.05)
    wnms_ap = compute_ap_for_avtubes_as_df(
            av_gt_tubes, co_wnms_av_stubes, iou_thresholds, False, False)

    action = 'Drinking'
    action_dwti_ind = dataset.action_names.index(action) + 1
    vfold = small.mkdir(out/'ap_emulate'/action)
    v_stubes = wnms_av_stubes[action]
    v_gt_tubes = av_gt_tubes[action]

    # ap-like preparations
    fgts, fdets = _convert_to_flat_representation(v_gt_tubes, v_stubes)
    det_to_eligible_gt = _compute_eligible_tubes_for_eval_weingroup(
            fgts, fdets, wgi_to_gti)
    gt_already_matched = np.zeros(len(fgts), dtype=bool)
    nd = len(fdets)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    npos = len([x for x in fgts if not x['diff']])
    detection_scores = np.array([x['score'] for x in fdets])
    detection_scores = detection_scores.round(3)
    sorted_inds = np.argsort(-detection_scores)
    iou_thresh = 0.5
    use_diff = False

    # Provenance
    detection_matched_to_which_gt = np.ones(len(fdets), dtype=int)*-1
    ifdet_to_iou_coverages = {}

    for d, ifdet in enumerate(sorted_inds):
        matchable_ifgts = list(det_to_eligible_gt.get(ifdet, {}))
        if not len(matchable_ifgts):
            fp[d] = 1
            continue

        iou_coverages: List[float] = \
            _compute_iou_coverages(fgts, fdets, det_to_eligible_gt,
                    matchable_ifgts, ifdet, False)

        ifdet_to_iou_coverages[ifdet] = iou_coverages

        max_coverage_local_id: int = np.argmax(iou_coverages)
        max_coverage: float = iou_coverages[max_coverage_local_id]
        max_coverage_ifgt = matchable_ifgts[max_coverage_local_id]
        if max_coverage > iou_thresh:
            if (not use_diff) and fgts[max_coverage_ifgt]['diff']:
                continue
            if not gt_already_matched[max_coverage_ifgt]:
                tp[d] = 1
                gt_already_matched[max_coverage_ifgt] = True
                detection_matched_to_which_gt[ifdet] = max_coverage_ifgt
            else:
                fp[d] = 1
        else:
            fp[d] = 1

    cumsum_fp = np.cumsum(fp)
    cumsum_tp = np.cumsum(tp)
    rec = cumsum_tp / float(npos)
    prec = cumsum_tp / np.maximum(cumsum_tp + cumsum_fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, False)

    log.info(f'{ap=}')
    # Visualize this
    for d, ifdet in enumerate(sorted_inds):
        if not tp[d]:
            continue

        tube = fdets[ifdet]['obj']
        score = tube['score']
        sf, ef = tube['start_frame'], tube['end_frame']
        vid = tube['index'][0]
        frame_inds = tube['frame_inds']
        video_fold = small.mkdir(vfold/f'D{d:04d}_IFDET{ifdet:04d}_V({vid})_{sf}_to_{ef}_score{score:02f}')
        video_path = dataset.videos[vid]['path']

        # Extract
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, frame_inds, debug_filename=video_path)
        # Draw
        drawn_frames_u8 = []
        for i, (find, frame_BGR) in enumerate(zip(frame_inds, frames_u8)):
            image = frame_BGR.copy()
            box = tube['boxes'][i]

            connections_dwti = vf_connections_dwti[vid][find]
            scores_dwti = vf_cls_probs[vid][find]
            source_id = connections_dwti['dwti_sources'].index(tube['index'])

            dwti_box = connections_dwti['boxes'][source_id]
            dwti_score = scores_dwti[source_id][action_dwti_ind]
            assert np.allclose(dwti_box, box)

            snippets.cv_put_box_with_text(image, box,
                text='{}({}); {:.2f}/{:.2f}; {}'.format(
                    i, find, dwti_score, score, action))
            drawn_frames_u8.append(image)

        # # Save as images
        # for find, image in zip(frame_inds, drawn_frames_u8):
        #     cv2.imwrite(str(
        #         video_fold/'Fr{:05d}.png'.format(find)), image)

        # Save as video
        snippets.qsave_video(video_fold/'overlaid.mp4', drawn_frames_u8)


def eval_stubes_experimental(workfolder, cfg_dict, add_args):
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

    dgt_tubes: Dict[I_dgt, T_dgt] = _reindex_avd_to_dgt(av_gt_tubes)
    tubes_dwein: Dict[I_dwein, T_dwein]
    tubes_dwein = loadconvert_tubes_dwein('/home/vsydorov/projects/deployed/2019_12_Thesis/dervo/thes/gpuhost7/2020_03_31/10_clean_inputs/20_wein_tubes/thes.gpuhost7.2020_03_31.10_clean_inputs.20_wein_tubes/194e3a301c202bc5e818dca26953ddb268aa98b3/out/extracted_tubes.pkl')
    tubes_dwein = dtindex_filter_split(tubes_dwein, split_vids)

    wgi, gti = _get_weingroup_assignment(dgt_tubes, tubes_dwein)
    wgi_to_gti: Dict[I_weingroup, I_dgt] = dict(zip(wgi, gti))

    # Evaluation
    keyframe_av_stubes = small.load_pkl('/home/vsydorov/projects/deployed/2019_12_Thesis/dervo/thes/gpuhost7/2020_03_31/30_whole_video/30_pncaffe_rcnn/05_develop/10_splits_keyframes/thes.gpuhost7.2020_03_31.30_whole_video.30_pncaffe_rcnn.05_develop.10_splits_keyframes/RAW/out/av_stubes.pkl')

    # Recall without NMS
    iou_thresholds = cf['tube_eval.iou_thresholds']
    nms_thresh = 0.5
    recall_norm = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, norm_av_stubes, iou_thresholds, False)[0]
    recall_kf = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, keyframe_av_stubes, iou_thresholds, False)[0]
    # Ap without NMS
    ap_norm = compute_ap_for_avtubes_as_df(
            av_gt_tubes, norm_av_stubes, iou_thresholds, False, False)
    ap_kf = compute_ap_for_avtubes_as_df(
            av_gt_tubes, keyframe_av_stubes, iou_thresholds, False, False)

    # # Apply "special NMS"
    # wnms_norm_av_stubes = _weingroup_nms(norm_av_stubes, nms_thresh)
    # wnms_keyframe_av_stubes = _weingroup_nms(keyframe_av_stubes, nms_thresh)

    # Apply "special NMS"
    wnms_norm_av_stubes = \
            (small.stash2(out/'_temp_wnms_norm_av_stubes.pkl')
            (_weingroup_nms, norm_av_stubes, nms_thresh))
    wnms_keyframe_av_stubes = \
            (small.stash2(out/'_temp_wnms_keyframe_av_stubes.pkl')
            (_weingroup_nms, keyframe_av_stubes, nms_thresh))


    # Recall with "special" NMS
    wnms_recall_norm = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, wnms_norm_av_stubes, iou_thresholds, False)[0]
    wnms_recall_kf = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, wnms_keyframe_av_stubes, iou_thresholds, False)[0]

    # AP with "special" NMS.
    wnms_ap_norm = compute_ap_for_avtubes_as_df(
            av_gt_tubes, wnms_norm_av_stubes, iou_thresholds, False, False)
    wnms_ap_kf = compute_ap_for_avtubes_as_df(
            av_gt_tubes, wnms_keyframe_av_stubes, iou_thresholds, False, False)

    # WG AP with "special" NMS
    wnms_wap_norm = compute_ap_for_avtubes_WG_as_df(wgi_to_gti,
            av_gt_tubes, wnms_norm_av_stubes, iou_thresholds, False, False)
    wnms_wap_kf = compute_ap_for_avtubes_WG_as_df(wgi_to_gti,
            av_gt_tubes, wnms_keyframe_av_stubes, iou_thresholds, False, False)

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    # Ncfg_tube_eval.evalprint_if(cf, norm_av_stubes, av_gt_tubes)
    # Ncfg_tube_eval.evalprint_if(cf, keyframe_av_stubes, av_gt_tubes)
    # Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)
