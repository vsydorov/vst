import pandas as pd
import warnings
import logging
import re
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import (List, Tuple, Dict, List, TypeVar, cast, TypedDict, Set)  # NOQA

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.tools import snippets
from thes.caffe import (Nicolas_net_helper)
from thes.data.tubes.types import (
    DALY_wein_tube,
    DALY_wein_tube_index, DALY_gt_tube_index,
    Objaction_dets, Frametube, Sframetube,
    convert_dwein_tube, convert_dgt_tubes,
    dtindex_filter_split, av_filter_split,
    get_daly_gt_tubes, AV_dict
)
from thes.data.tubes.routines import (
    filter_tube_keyframes_only_gt,
    filter_tube_keyframes_only_gt_v2,
    nicphil_evaluations_to_tubes,
    compute_nms_for_av_stubes,
    score_ftubes_via_objaction_overlap_aggregation,
    _create_objdetection_helper_structure,
    _match_objectdetections_to_tubes,
    av_stubes_above_score
)
from thes.data.dataset.external import (
    DatasetDALY, DALY_action_name, DALY_object_name, DALY_vid)
from thes.detectron.daly import (
    get_daly_split_vids,
    simplest_daly_to_datalist_v2,
    get_datalist_action_object_converter,
)
from thes.evaluation.routines import (
    compute_recall_for_avtubes,
    compute_ap_for_avtubes
)


log = logging.getLogger(__name__)


class Box_connections_dwti(TypedDict):
    vid: DALY_vid
    frame_ind: int
    dwti_sources: List[DALY_wein_tube_index]  # N
    boxes: List[np.ndarray]  # N, 4


def computeprint_recall_ap_for_avtubes(
        av_gt_tubes: AV_dict[Frametube],
        av_stubes: AV_dict[Sframetube],
        iou_thresholds: List[float]):
    """
    Will compute tube ap per threshold, print table per thresh,
    print aggregate table
    """
    table_recall_s, table_recall_st = compute_recall_for_avtubes(
            av_gt_tubes, av_stubes, iou_thresholds)
    table_ap_s = compute_ap_for_avtubes(
            av_gt_tubes, av_stubes, iou_thresholds, False)
    table_ap_st = compute_ap_for_avtubes(
            av_gt_tubes, av_stubes, iou_thresholds, True)
    # // Print
    log.info('Spatial Recall:\n{}'.format(table_recall_s))
    log.info('Spatiotemp Recall:\n{}'.format(table_recall_st))
    log.info('Spatial AP:\n{}'.format(table_ap_s))
    log.info('Spatiotemp AP:\n{}'.format(table_ap_st))


def _set_defcfg_dataset_seed(cfg):
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['test', str]
    seed: [42, int]
    """)


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
                leave_only_gt_keyframes:
                    enabled: [False, bool]
                    keep_temporal: [True, bool]
        """)

    @staticmethod
    def resolve_tubes(
            cf,
            av_gt_tubes: AV_dict[Frametube],
            split_vids: List[DALY_vid]
            ) -> Dict[DALY_wein_tube_index, Frametube]:
        ftubes: Dict[DALY_wein_tube_index, Frametube]
        if cf['tubes.source'] == 'wein':
            ftubes = resolve_wein_tubes(
                    av_gt_tubes, split_vids, cf['tubes.wein.path'],
                    cf['tubes.wein.leave_only_gt_keyframes.enabled'],
                    cf['tubes.wein.leave_only_gt_keyframes.keep_temporal'])
        elif cf['tubes.source'] == 'gt':
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return ftubes


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
            minscore_cutoff: [0.05, float]
            nms:
                enabled: [True, bool]
                thresh: [0.5, float]
            iou_thresholds: [[0.3, 0.5, 0.7], list]
        """)

    @staticmethod
    def evalprint_if(cf,
            av_stubes: AV_dict[Sframetube],
            av_gt_tubes: AV_dict[Frametube]):
        if not cf['tube_eval.enabled']:
            return
        av_stubes = av_stubes_above_score(
                av_stubes, cf['tube_eval.minscore_cutoff'])
        if cf['tube_eval.nms.enabled']:
            av_stubes = compute_nms_for_av_stubes(
                    av_stubes, cf['tube_eval.nms.thresh'])
        computeprint_recall_ap_for_avtubes(
                av_gt_tubes, av_stubes, cf['tube_eval.iou_thresholds'])


def resolve_wein_tubes(
        av_gt_tubes: AV_dict[Frametube],
        split_vids: List[DALY_vid],
        wein_path: Path,
        l_enabled: bool,
        l_keeptemp: bool):
    dwein_tubes: Dict[DALY_wein_tube_index, DALY_wein_tube] = \
            small.load_pkl(wein_path)
    dwein_tubes = dtindex_filter_split(dwein_tubes, split_vids)
    # Convert dwein_tubes to sparse tubes
    ftubes = {k: convert_dwein_tube(t) for k, t in dwein_tubes.items()}
    # Filter tubes optionally
    if l_enabled:
        ftubes = filter_tube_keyframes_only_gt_v2(
                ftubes, av_gt_tubes, l_keeptemp)
    return ftubes


def _resolve_actobjects(cf, dataset, split_vids):
    # / Assign objects to tubes
    # // Create objaction_dets in video frames
    objactions_vf: Dict[DALY_vid, Dict[int, Objaction_dets]] = {}
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


def _set_tubes(cf, dataset):
    tubes_per_video: Dict[DALY_wein_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['tubes.imported_wein_tubes'])
    if cf['tubes.filter_gt']:
        tubes_per_video = filter_tube_keyframes_only_gt(
                dataset, tubes_per_video)
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    tubes_per_video = dtindex_filter_split(split_vids, tubes_per_video)
    return tubes_per_video


def _get_gt_sparsetubes(dataset, split_vids, gt_tubes) -> AV_dict[Frametube]:
    warnings.warn('Deprecated')
    av_gttubes: AV_dict[Frametube] = {}

    for ckey, gt_tube in gt_tubes.items():
        (vid, action_name, ins_ind) = ckey
        if vid not in split_vids:
            continue
        vmp4 = dataset.source_videos[vid]
        height = vmp4['height']
        width = vmp4['width']
        ocv_video_fps = vmp4['frames_reached']/vmp4['length_reached']
        frame_inds = np.array(gt_tube['frame_inds'])
        unscaled_boxes = np.array(gt_tube['boxes'])
        boxes = unscaled_boxes * np.tile([width, height], 2)
        start_frame = int(gt_tube['start_time'] * ocv_video_fps)
        end_frame = int(gt_tube['end_time'] * ocv_video_fps)
        sparse_tube: Frametube = {
                'frame_inds': frame_inds,
                'boxes': boxes,
                'start_frame': start_frame,
                'end_frame': end_frame}
        (av_gttubes
                .setdefault(action_name, {})
                .setdefault(vid, [])).append(sparse_tube)
    return av_gttubes


def _daly_tube_map(
        cf, out, dataset,
        stubes_va: AV_dict[Sframetube],
        gttubes_va: AV_dict[Frametube]):

    # Apply per-class NMS
    if cf['tube_nms.enabled']:
        tube_nms_thresh = cf['tube_nms.thresh']
        stubes_va = compute_nms_for_av_stubes(stubes_va, tube_nms_thresh)
    iou_thresholds = cf['eval.iou_thresholds']
    computeprint_recall_ap_for_avtubes(
            gttubes_va, stubes_va, iou_thresholds)


def _recreate_actobject_datalist(dataset, split_vids):
    # /// Recreate the datalist that was used for detections
    datalist = simplest_daly_to_datalist_v2(dataset, split_vids)
    object_names, datalist_converter = \
            get_datalist_action_object_converter(dataset)
    datalist = datalist_converter(datalist)
    return datalist


def equal_tube_split(tubes_per_video, ct, split_kind):
    key_indices = np.arange(len(tubes_per_video))
    key_list = list(tubes_per_video.keys())

    # Simple tube df
    nframes_df = []
    for k, v in tubes_per_video.items():
        vid = k[0]
        nframes = len(v['frame_inds'])
        nframes_df.append([vid, nframes])
    nframes_df = pd.DataFrame(nframes_df, columns=['vid', 'nframes'])
    nframes_df['keys'] = key_list

    # Divide indices
    if split_kind == 'tubes':
        equal_split = np.array_split(key_indices, ct)
    elif split_kind == 'frames':
        approx_nframes_per_split = nframes_df.nframes.sum() // ct
        approx_split_indices = approx_nframes_per_split * np.arange(1, ct)
        split_indices = np.searchsorted(
                nframes_df.nframes.cumsum(), approx_split_indices)
        equal_split = np.array_split(key_indices, split_indices)
    else:
        raise NotImplementedError()

    # Assign splits
    for i, inds in enumerate(equal_split):
        nframes_df.loc[inds, 'split'] = i
    nframes_df['split'] = nframes_df['split'].astype(int)

    # Compute stats
    gb_chunk = nframes_df.groupby('split')
    all_nvids = gb_chunk['vid'].unique().apply(len)
    all_nframes = gb_chunk['nframes'].sum()
    split_stats = pd.concat((all_nvids, all_nframes), axis=1)

    # Divide tubes
    split_tubes = [{} for i in range(ct)]
    for i, group in gb_chunk.groups.items():
        keys = nframes_df.loc[group, 'keys'].tolist()
        for k in keys:
            split_tubes[i][k] = tubes_per_video[k]
    return split_tubes, split_stats


def _parcel_management(cf, tubes_per_video):
    # // Computation of parcels
    cc, ct = (cf['compute.chunk'], cf['compute.total'])
    split_kind = cf['compute.equal_split']
    split_tubes, split_stats = \
            equal_tube_split(tubes_per_video, ct, split_kind)
    ctubes_per_video = split_tubes[cc]
    # Logging part
    log.info('Chunk {}/{}: {} -> {}'.format(
        cc, ct, len(tubes_per_video), len(ctubes_per_video)))
    log.info('split_stats:\n{}'.format(split_stats))
    return ctubes_per_video


def get_daly_keyframes(dataset, split_vids) -> Dict[DALY_vid, np.ndarray]:
    to_cover_: Dict[DALY_vid, Set] = {}
    for vid in split_vids:
        v = dataset.video_odict[vid]
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                keyframes = [kf['frameNumber'] for kf in instance['keyframes']]
                to_cover_[vid] = to_cover_.get(vid, set()) | set(list(keyframes))
    frames_to_cover = \
            {k: np.array(sorted(v)) for k, v in to_cover_.items()}
    return frames_to_cover


def prepare_ftube_box_computations(
        ftubes: Dict[DALY_wein_tube_index, Frametube],
        frames_to_cover: Dict[DALY_vid, np.ndarray]
        ) -> Dict[DALY_vid, Dict[int, Box_connections_dwti]]:
    # Assign boxes (and keep connections to original ftubes)
    vf_connections_dwti_list: Dict[DALY_vid, Dict[int,
        List[Tuple[DALY_wein_tube_index, np.ndarray]]]] = {}
    for dwt_index, tube in ftubes.items():
        (vid, bunch_id, tube_id) = dwt_index
        tube_finds = tube['frame_inds']
        good_finds = frames_to_cover[vid]
        common_finds, comm1, comm2 = np.intersect1d(
            tube_finds, good_finds, assume_unique=True, return_indices=True)
        if len(common_finds) == 0:
            continue
        good_tube_boxes = tube['boxes'][comm1]
        for find, box in zip(common_finds, good_tube_boxes):
            (vf_connections_dwti_list
                .setdefault(vid, {})
                .setdefault(find, []).append((dwt_index, box)))
    # Prettify
    vf_connections_dwti: Dict[DALY_vid, Dict[int, Box_connections_dwti]] = {}
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


def _demovis_apply_pncaffe_rcnn(
        neth, dataset, vf_connections_dwti, out):
    vfold = small.mkdir(out/'demovis')
    nicolas_labels = ['background', ] + dataset.action_names
    for vid, f_connections_dwti in tqdm(
            vf_connections_dwti.items(), 'nicphil_demovis'):
        vmp4 = dataset.source_videos[vid]
        video_path = vmp4['video_path']
        finds = list(f_connections_dwti)

        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, finds, debug_filename=video_path)

        video_fold = small.mkdir(vfold/f'vid{vid}')

        for find, frame_BGR in zip(finds, frames_u8):
            connections_dwti = f_connections_dwti[find]
            boxes = connections_dwti['boxes']
            cls_probs = neth.score_boxes(frame_BGR, boxes)  # N, (bcg+10)

        txt_output = []
        for i, (frame, score) in enumerate(zip(frames_u8, scores_per_frame)):
            image = frame.copy()
            box = tube['boxes'][i]
            real_framenum = tube['frame_inds'][i]
            best_score_id = np.argmax(score)
            best_score = score[best_score_id]
            best_nicolas_label = nicolas_labels[best_score_id]
            snippets.cv_put_box_with_text(
                    image, box,
                    text='{} {} {} {:.2f}'.format(
                        i, real_framenum,
                        best_nicolas_label, best_score))
            line = ' '.join([f'{y}: {x:.3f}'
                for x, y in zip(score, nicolas_labels)])
            txt_output.append(line)
            cv2.imwrite(
                    str(video_fold/'frame{:03d}_{:03d}.jpg'.format(
                        i, real_framenum)),
                    image)
        with (video_fold/'scores.txt').open('w') as f:
            f.write('\n'.join(txt_output))

# Experiments


def load_wein_tubes(workfolder, cfg_dict, add_args):
    """
    Philippe tubes:
        tube:
             (one row per frame):
                index of the frame (starting at 1)
                x1 y1 x2 y2
                score of the generic human detector
                score of the instance-specific detector
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    wein_tubes: [~, str]
    dataset:
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    # Dataset
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    # Tubes
    # For reason I forgot we only care about [0] element
    wein_package = small.load_py2_pkl(cf['wein_tubes'])[0]
    # We got a dictionary of filenames (w .mp4 suffix)
    extracted_tubes: Dict[DALY_wein_tube_index, DALY_wein_tube] = {}
    for vid_mp4, wein_bunches in wein_package.items():
        vid = re.search(r'(.*)\.mp4', vid_mp4).group(1)
        vmp4 = dataset.source_videos[vid]
        for bunch_id, wein_tubes in enumerate(wein_bunches):
            for tube_id, wein_tube in enumerate(wein_tubes):
                frame_inds = wein_tube[:, 0].astype(np.int) - 1
                assert max(frame_inds) < vmp4['frames_reached']
                boxes_ltrd = wein_tube[:, 1:5]  # ltrd
                human_scores = wein_tube[:, 5]
                instance_scores = wein_tube[:, 6]
                tube = {
                        'frame_inds': frame_inds,
                        'boxes': boxes_ltrd,
                        'hscores': human_scores,
                        'iscores': instance_scores}
                extracted_tubes[(vid, bunch_id, tube_id)] = tube
    small.save_pkl(out/'extracted_tubes.pkl', extracted_tubes)


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
    _set_defcfg_dataset_seed(cfg)
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

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    av_gt_tubes: AV_dict[Frametube] = \
            convert_dgt_tubes(get_daly_gt_tubes(dataset))
    av_gt_tubes = av_filter_split(av_gt_tubes, split_vids)
    # Inputs to assignment routine
    ftubes: Dict[DALY_wein_tube_index, Frametube] = \
            Ncfg_tubes.resolve_tubes(cf, av_gt_tubes, split_vids)
    objactions_vf: Dict[DALY_vid, Dict[int, Objaction_dets]] = \
            _resolve_actobjects(cf, dataset, split_vids)
    # Assignment itself
    overlap_type = cf['obj_to_tube.overlap_type']
    overlap_cutoff = cf['obj_to_tube.overlap_cutoff']
    score_cutoff = cf['obj_to_tube.score_cutoff']
    av_stubes: AV_dict[Sframetube] = \
        score_ftubes_via_objaction_overlap_aggregation(
        objactions_vf, ftubes, overlap_type, overlap_cutoff, score_cutoff)
    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)


def apply_pncaffe_rcnn_in_frames(workfolder, cfg_dict, add_args):
    """
    Run Philippes/Nicolas caffe model to extract 'rcnn scores'
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    _set_defcfg_dataset_seed(cfg)
    Ncfg_tubes.set_defcfg(cfg)
    Ncfg_nicphil_rcnn.set_defcfg(cfg)
    Ncfg_tube_eval.set_defcfg(cfg)
    cfg.set_deftype("""
    demo_run: [False, bool]
    compute:
        save_period: ['::10', str]
        split:
            enabled: [False, bool]
            chunk: [0, "VALUE >= 0"]
            total: [1, int]
    """)
    cf = cfg.parse()
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    av_gt_tubes: AV_dict[Frametube] = \
            convert_dgt_tubes(get_daly_gt_tubes(dataset))
    av_gt_tubes = av_filter_split(av_gt_tubes, split_vids)
    ftubes: Dict[DALY_wein_tube_index, Frametube] = \
            Ncfg_tubes.resolve_tubes(cf, av_gt_tubes, split_vids)

    neth: Nicolas_net_helper = Ncfg_nicphil_rcnn.resolve_helper(cf)

    # Cover only keyframes when evaluating dwti tubes
    frames_to_cover = get_daly_keyframes(dataset, split_vids)
    vf_connections_dwti: Dict[DALY_vid, Dict[int, Box_connections_dwti]] = \
            prepare_ftube_box_computations(ftubes, frames_to_cover)

    if cf['demo_run']:
        vf_connections_dwti = sample_dict(
            vf_connections_dwti, N=10, NP_SEED=0)
        _demovis_apply_pncaffe_rcnn(
                neth, dataset, vf_connections_dwti, out)
        raise NotImplementedError()
        return

    def isaver_eval_func(vid):
        f_connections_dwti = vf_connections_dwti[vid]
        vmp4 = dataset.source_videos[vid]
        video_path = vmp4['video_path']
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

    assert not cf['compute.split.enabled']

    isaver_keys = list(vf_connections_dwti.keys())
    isaver = snippets.Simple_isaver(
            small.mkdir(out/'isave_caffe_vid_eval'),
            isaver_keys, isaver_eval_func,
            cf['compute.save_period'], 120)

    isaver_items = isaver.run()
    vf_cls_probs: Dict[DALY_vid, Dict[int, np.ndarray]]
    vf_cls_probs = dict(zip(isaver_keys, isaver_items))

    # Pretty clear we'll be summing up the scores anyway
    ftube_scores = {k: np.zeros(11) for k in ftubes}
    for vid, f_cls_probs in vf_cls_probs.items():
        for f, cls_probs in f_cls_probs.items():
            dwtis = vf_connections_dwti[vid][f]['dwti_sources']
            for dwti, prob in zip(dwtis, cls_probs):
                ftube_scores[dwti] += prob

    # Create av_stubes
    av_stubes: AV_dict[Sframetube] = {}
    for dwt_index, tube in ftubes.items():
        (vid, bunch_id, tube_id) = dwt_index
        for action_name, score in zip(
                dataset.action_names, ftube_scores[dwt_index][1:]):
            stube = tube.copy()
            stube['score'] = score
            stube = cast(Sframetube, stube)
            (av_stubes
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))

    small.save_pkl(out/'av_stubes.pkl', av_stubes)
    Ncfg_tube_eval.evalprint_if(cf, av_stubes, av_gt_tubes)

    # # Cover demo case
    # if cf['demo_run']:
    #     _perform_tube_demovis(dataset, some_tubes, out,
    #             PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE)
    #     return
    #
    # # // Computation of the parcels inside chosen chunk
    # chunk = (cf['compute.chunk'], cf['compute.total'])
    # cc, ct = chunk
    #
    # key_indices = np.arange(len(tubes_per_video))
    # key_list = list(tubes_per_video.keys())
    # chunk_indices = np.array_split(key_indices, ct)[cc]
    # chunk_keys = [key_list[i] for i in chunk_indices]
    # log.info('Chunk {}: {} -> {}'.format(
    #     chunk, len(key_indices), len(chunk_indices)))
    # small.save_pkl(out/'tubescores_dict.pkl', tubescores_dict)


def hacky_gather_evaluated_tubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    cfg.set_deftype("""
    etubes: [~, ~]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    """)
    cf = cfg.parse()

    # Read tubes, merge dicts
    tubescores_dict = {}
    for tubepath in cf['etubes']:
        tubes = small.load_pkl(tubepath)
        tubescores_dict.update(tubes)

    # Confirm that keys match
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    tubes_per_video = _set_tubes(cf, dataset)

    assert tubes_per_video.keys() == tubescores_dict.keys(), \
            "Keys should match"

    small.save_pkl(out/'tubescores_dict.pkl', tubescores_dict)


def actual_eval_of_nicphil_etubes(workfolder, cfg_dict, add_args):
    """
    Evaluate "tubes" as if they were VOC objects
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    tubescores_dict: [~, ~]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    tube_nms:
        enabled: [True, bool]
        thresh: [0.5, float]
    eval:
        iou_thresholds: [[0.3, 0.5, 0.7], list]
        spatiotemporal: [False, bool]
        use_07_metric: [False, bool]
        use_diff: [False, bool]
    """)
    cf = cfg.parse()

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # // Obtain GT tubes
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    gt_tubes = get_daly_gt_tubes(dataset)
    gttubes_va = \
            _get_gt_sparsetubes(dataset, split_vids, gt_tubes)

    # // Obtain detected tubes
    tubes_per_video: \
            Dict[DALY_wein_tube_index, DALY_wein_tube] = _set_tubes(cf, dataset)
    # Refer to the originals for start_frame/end_frame
    original_tubes_per_video: Dict[DALY_wein_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['tubes.imported_wein_tubes'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    original_tubes_per_video = dtindex_filter_split(
            split_vids, original_tubes_per_video)

    tubescores_dict = small.load_pkl(cf['tubescores_dict'])
    stubes_va = nicphil_evaluations_to_tubes(
            dataset, tubes_per_video,
            original_tubes_per_video, tubescores_dict)
    _daly_tube_map(cf, out, dataset, stubes_va, gttubes_va)


def actual_eval_of_action_object_predictions(workfolder, cfg_dict, add_args):
    """
    Evaluate "tubes" as if they were VOC objects
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    tube_nms:
        enabled: [True, bool]
        thresh: [0.5, float]
    eval:
        iou_thresholds: [[0.3, 0.5, 0.7], list]
        spatiotemporal: [False, bool]
        use_07_metric: [False, bool]
        use_diff: [False, bool]
    actobjects_evaluated: [~, ~]
    obj_to_tube:
        overlap_type: ['inner_overlap', ['inner_overlap', 'iou']]
        overlap_cutoff: [0.2, float]
        score_cutoff: [0.2, float]
    """)
    cf = cfg.parse()

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # // Obtain GT tubes
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    gt_tubes = get_daly_gt_tubes(dataset)
    gttubes_va = \
            _get_gt_sparsetubes(dataset, split_vids, gt_tubes)

    # // Obtain detected tubes
    tubes_per_video: \
            Dict[DALY_wein_tube_index, DALY_wein_tube] = _set_tubes(cf, dataset)
    # Refer to the originals for start_frame/end_frame
    original_tubes_per_video: Dict[DALY_wein_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['tubes.imported_wein_tubes'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    original_tubes_per_video = \
            dtindex_filter_split(split_vids, original_tubes_per_video)

    actobjects_evaluated = small.load_pkl(cf['actobjects_evaluated'])

    preds_per_framevid = _create_objdetection_helper_structure(
        dataset, split_label, actobjects_evaluated)
    overlap_type = cf['obj_to_tube.overlap_type']
    overlap_cutoff = cf['obj_to_tube.overlap_cutoff']
    score_cutoff = cf['obj_to_tube.score_cutoff']
    stubes_va = _match_objectdetections_to_tubes(
        dataset, tubes_per_video, original_tubes_per_video,
        preds_per_framevid, overlap_type, overlap_cutoff, score_cutoff)
    _daly_tube_map(cf, out, dataset, stubes_va, gttubes_va)


def eval_daly_tubes_RGB_with_pfadet_gather_evaluated(
        workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    cfg.set_deftype("""
    etubes: [~, ~]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    """)
    cf = cfg.parse()

    # Read tubes, merge dicts
    tubescores_dict = {}
    for tubepath in tqdm(cf['etubes'], 'loading etubes'):
        tubes = small.load_pkl(tubepath)
        tubescores_dict.update(tubes)

    # Confirm that keys match
    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    tubes_per_video = _set_tubes(cf, dataset)

    assert tubes_per_video.keys() == tubescores_dict.keys(), \
            "Keys should match"

    small.save_pkl(out/'tube_instances_dict.pkl', tubescores_dict)


def map_score_tubes_and_pfadet_rcnn_scores(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    tube_instances_dict: [~, ~]
    dataset:
        name: [~, ['daly']]
        cache_folder: [~, str]
        subset: ['train', str]
    tubes:
        imported_wein_tubes: [~, ~]
        filter_gt: [False, bool]
    rcnn_assignment:
        use_boxes: [False, bool]
    tube_nms:
        enabled: [True, bool]
        thresh: [0.5, float]
    eval:
        iou_thresholds: [[0.3, 0.5, 0.7], list]
        spatiotemporal: [False, bool]
        use_07_metric: [False, bool]
        use_diff: [False, bool]
    """)
    cf = cfg.parse()

    dataset = DatasetDALY()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # // Obtain GT tubes
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    gt_tubes = get_daly_gt_tubes(dataset)
    gttubes_va = \
            _get_gt_sparsetubes(dataset, split_vids, gt_tubes)

    # // Obtain detected tubes
    tubes_per_video: \
            Dict[DALY_wein_tube_index, DALY_wein_tube] = _set_tubes(cf, dataset)
    # Refer to the originals for start_frame/end_frame
    original_tubes_per_video: Dict[DALY_wein_tube_index, DALY_wein_tube] = \
        small.load_pkl(cf['tubes.imported_wein_tubes'])
    split_label = cf['dataset.subset']
    split_vids = get_daly_split_vids(dataset, split_label)
    original_tubes_per_video = \
            dtindex_filter_split(split_vids, original_tubes_per_video)

    tube_instances_dict = small.load_pkl(cf['tube_instances_dict'])
    stubes_va: \
            Dict[DALY_action_name,
                    Dict[DALY_vid, List[Sframetube]]] = {}

    assert cf['rcnn_assignment.use_boxes'] is False
    # # Only record scores > 0.01
    # score_record_thresh = 0.01
    for ckey, tube in tubes_per_video.items():
        (vid, bunch_id, tube_id) = ckey
        original_tube = original_tubes_per_video[ckey]
        tube_instances = tube_instances_dict[ckey]
        # Ignore boxes in tube instances
        scores_per_actid = np.zeros(len(dataset.action_names))
        for i, ins in enumerate(tube_instances):
            for pred_cls, score in zip(ins.pred_classes, ins.scores):
                scores_per_actid[pred_cls] += score
        start_frame = original_tube['frame_inds'].min()
        end_frame = original_tube['frame_inds'].max()
        for action_name, score in zip(
                dataset.action_names, scores_per_actid):
            sparse_scored_tube = {
                    'frame_inds': tube['frame_inds'],
                    'boxes': tube['boxes'],
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'score': score}
            (stubes_va
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(sparse_scored_tube))
    _daly_tube_map(cf, out, dataset, stubes_va, gttubes_va)
