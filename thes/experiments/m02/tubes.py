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
    compute_ap_for_avtubes_as_df,)
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
    def _perform_split(cls, cf, vf_connections_dwti, vids_to_eval):
        # Reduce keys according to split
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
            len(vids_to_eval), np.sum(weights), len(chunk_vids), weights_split[cc]))
        log.debug(f'Full stats [{cc,ct=}]:\n'
                f'vids_split={pprint.pformat(vids_split)}\n'
                f'{weights_split=}\n'
                f'{chunk_vids=}\n'
                f'{weights_split[cc]=}')
        return chunk_vids

    @classmethod
    def _aggregate_scores(cls,
            dataset, tubes_dwein,
            vf_connections_dwti, vf_cls_probs):
        # Pretty clear we'll be summing up the scores anyway
        ftube_scores = {k: np.zeros(11) for k in tubes_dwein}
        for vid, f_cls_probs in vf_cls_probs.items():
            for f, cls_probs in f_cls_probs.items():
                dwtis = vf_connections_dwti[vid][f]['dwti_sources']
                for dwti, prob in zip(dwtis, cls_probs):
                    ftube_scores[dwti] += prob

        # Create av_stubes
        av_stubes: AV_dict[T_dwein_scored] = {}
        for dwt_index, tube in tubes_dwein.items():
            (vid, bunch_id, tube_id) = dwt_index
            for action_name, score in zip(
                    dataset.action_names, ftube_scores[dwt_index][1:]):
                stube = tube.copy()
                stube = cast(T_dwein_scored, stube)
                stube['score'] = score
                (av_stubes
                        .setdefault(action_name, {})
                        .setdefault(vid, []).append(stube))
        return av_stubes

    @classmethod
    def demo_run(cls, cf, out, dataset, split_vids, tubes_dwein, neth):
        vf_connections_dwti = cls._define_boxes_to_evaluate(
                cf, dataset, split_vids, tubes_dwein)
        vf_connections_dwti = sample_dict(
            vf_connections_dwti, N=5, NP_SEED=0)
        vfold = small.mkdir(out/'demovis')
        cls._demovis_apply(vfold, neth, dataset, vf_connections_dwti)

    @classmethod
    def score_tubes(
            cls, cf, out,
            dataset: Dataset_daly_ocv,
            split_vids,
            tubes_dwein: Dict[I_dwein, T_dwein],
            neth) -> Optional[AV_dict[T_dwein_scored]]:
        """
        Logic behind simple "evaluate boxes" experiment
        """
        vf_connections_dwti = cls._define_boxes_to_evaluate(
                cf, dataset, split_vids, tubes_dwein)

        vids_to_eval = list(vf_connections_dwti.keys())
        if cf['compute.split.enabled']:
            vids_to_eval = cls._perform_split(
                    cf, vf_connections_dwti, vids_to_eval)

        # GPU utilized here
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
        isaver = snippets.Simple_isaver(
                small.mkdir(out/'isave_rcnn_vid_eval'),
                vids_to_eval, isaver_eval_func,
                cf['compute.save_period'], 120)
        isaver_items = isaver.run()
        vf_cls_probs: Dict[Vid_daly, Dict[int, np.ndarray]]
        vf_cls_probs = dict(zip(vids_to_eval, isaver_items))

        av_stubes = cls._aggregate_scores(
                dataset, tubes_dwein, vf_connections_dwti, vf_cls_probs)

        return av_stubes


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
        Ncfg_generic_rcnn_eval.demo_run()
        return
    av_stubes = Ncfg_generic_rcnn_eval.score_tubes(
            cf, out, dataset, split_vids, tubes_dwein, neth)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
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
    Ncfg_generic_rcnn_eval.set_defcfg(cfg)
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

    nms_thresh = cf['tube_eval.nms.thresh']
    iou_thresholds: List[float] = cf['tube_eval.iou_thresholds']
    minscore_cutoff = 0.00

    tubes_to_eval = av_stubes_above_score(
            tubes_to_eval, minscore_cutoff)
    tubes_to_eval = \
            compute_nms_for_av_stubes(tubes_to_eval, nms_thresh)
    dfdict = _compute_exhaustive_evaluation_stats(
            av_gt_tubes, tubes_to_eval, iou_thresholds)
    _print_exhaustive_evaluation_stats(dfdict)
