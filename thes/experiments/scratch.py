import logging
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from thes.data.dataset.external import (Dataset_daly_ocv)
from thes.tools import snippets


log = logging.getLogger(__name__)


def _examine_stats(self, rstats):
    vids = list(self.videos.keys())
    d_rstats = pd.DataFrame(rstats).T.loc[vids]
    d_pmeta = pd.DataFrame(self.provided_metas).T.loc[vids]

    # frame count mismatch
    bad_nframes = d_rstats.loc[
            (d_rstats['frame_count'] != d_pmeta['nbframes_ffmpeg'])]
    d_pmeta.loc[bad_nframes.index]
    d_rstats.loc[bad_nframes.index]
    d_rstats['meta_duration'] = d_pmeta['duration']
    d_rstats['meta_fps'] = d_pmeta['fps']
    d_rstats['meta_frame_count'] = d_pmeta['nbframes_ffmpeg']
    d_rstats['max_time'] = d_rstats['max_pos_msec']/1000
    d_rstats['max_frame'] = d_rstats['max_pos_frames'] - 1
    d_rstats['est_fps'] = d_rstats['max_frame']/d_rstats['max_time']
    d_rstats['est_length'] = d_rstats['max_time'] + 1/d_rstats['est_fps']
    # Count mismatches
    d_rstats['diff_fps'] = (d_rstats['est_fps']-d_rstats['meta_fps']).abs()
    d_rstats['diff_nframes'] = (
            d_rstats['max_pos_frames']-d_rstats['meta_frame_count']).abs()
    d_rstats['diff_length'] = (
            d_rstats['est_length']-d_rstats['meta_duration']).abs()
    diff_columns = ['diff_fps', 'diff_nframes', 'diff_length']
    diff_describe = d_rstats[diff_columns].describe()
    d_bad = pd.concat([d_rstats[d] > 1e-8 for d in diff_columns], axis=1)
    d_bad.sum()
    # Let's aggregate the keyframes
    kflist = []
    for vid, video in self.videos.items():
        for action_name, ains in video['instances'].items():
            for ins_ind, instance in enumerate(ains):
                for kf_ind, kf in enumerate(instance['keyframes']):
                    kfitem = {}
                    kfitem.update(kf)
                    kfitem.update({
                        'vid': vid,
                        'action_name': action_name,
                        'ins_ind': ins_ind,
                        'kf_ind': kf_ind})
                    kflist.append(kfitem)
    d_kf = pd.DataFrame(kflist)
    d_kf['meta_fps'] = d_rstats.loc[d_kf['vid']].reset_index()['meta_fps']
    d_kf['est_fps'] = d_rstats.loc[d_kf['vid']].reset_index()['est_fps']
    d_kf['x'] = (d_kf['time']*d_kf['meta_fps']).apply(np.ceil)
    d_kf[d_kf['x'] != d_kf['frameNumber']]

def _examine_kframes(self, rstats):
    vids = list(self.videos.keys())
    d_rstats = pd.DataFrame(rstats).T.loc[vids]
    d_rstats['est_fps'] = (d_rstats['max_pos_frames']-1)*\
            1000/d_rstats['max_pos_msec']

    action_list = []
    for vid, video in self.videos.items():
        for action_name, ains in video['instances'].items():
            for ins_ind, instance in enumerate(ains):
                actitem = {}
                actitem['beginTime'] = instance['beginTime']
                actitem['endTime'] = instance['endTime']
                actitem.update({
                    'vid': vid,
                    'action_name': action_name,
                    'ins_ind': ins_ind})
                action_list.append(actitem)
    dact = pd.DataFrame(action_list)
    dact['est_fps'] = d_rstats.loc[dact['vid']].reset_index()['est_fps']
    dact['total_frames'] = d_rstats.loc[
            dact['vid']].reset_index()['max_pos_frames']
    dact['xbegin'] = (dact['beginTime']*dact['est_fps']
                      ).apply(lambda x: max(1, np.ceil(x)))
    dact['xend'] = (dact['endTime']*dact['est_fps']).apply(np.ceil)
    dact.loc[dact['xend']>dact['total_frames'], 'xend'] = dact['total_frames']

    kflist = []
    for vid, video in self.videos.items():
        for action_name, ains in video['instances'].items():
            for ins_ind, instance in enumerate(ains):
                for kf_ind, kf in enumerate(instance['keyframes']):
                    kfitem = {}
                    kfitem['time'] = kf['time']
                    kfitem['frameNumber'] = kf['frameNumber']
                    kfitem.update({
                        'vid': vid,
                        'action_name': action_name,
                        'ins_ind': ins_ind,
                        'kf_ind': kf_ind})
                    kflist.append(kfitem)
    dkf = pd.DataFrame(kflist)
    dkf['est_fps'] = d_rstats.loc[dkf['vid']].reset_index()['est_fps']
    dkf['orig_fps'] = d_rstats.loc[dkf['vid']].reset_index()['fps']
    # eps = np.finfo(np.float32).eps
    dkf['x_orig'] = (dkf['time']*dkf['orig_fps']).apply(np.ceil)
    dkf[dkf['x_orig'] != dkf['frameNumber']]
    dkf['x_est'] = (dkf['time']*dkf['est_fps']).apply(np.ceil)
    dkf[dkf['x_est'] != dkf['frameNumber']]


def compare_data(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(raise_without_defaults=False)
    cfg.set_defaults("""
    cache_folders: ~
    """)
    cf = cfg.parse()

    dataset = Dataset_daly_ocv()
    dataset.precompute_to_folder(out)
    # # Dataset per each cache folder
    # ddict = {}
    # for k, v in cfg_dict['cache_folders'].items():
    #     dataset = DatasetDALY()
    #     dataset.populate_from_folder(v)
    #     ddict[k] = dataset
    #
    # odict_pds = {}
    # sv_pds = {}
    # for k, v in ddict.items():
    #     odict_pds[k] = pd.DataFrame(v.video_odict).T.sort_index()
    #     sv_pds[k] = pd.DataFrame(v.source_videos).T.sort_index()
    #
    # sv_pds['d18'] == sv_pds['m317']
    # (sv_pds['d18']['frames_reached'] == sv_pds['m317']['frames_reached']).all()

    # vid10 = list(ddict['d18'].video_odict.keys())[:10]
    # newstats = {}
    # for vid in tqdm(vid10):
    #     vmp4_d18 = ddict['d18'].source_videos[vid]
    #     vcap = cv2.VideoCapture(str(vmp4_d18['video_path']))
    #     vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    #     while True:
    #         frames_reached = int(vcap.get(cv2.CAP_PROP_POS_FRAMES))
    #         ms_reached = int(vcap.get(cv2.CAP_PROP_POS_MSEC))
    #         ret = vcap.grab()
    #         if ret is False:
    #             break
    #     vcap.close()
    #     newstats[vid] = (frames_reached, ms_reached/1000)
    # new4 = pd.DataFrame(newstats).T
    # new4.columns = ['frames', 'length']
    # new4['fps'] = new4['frames']/new4['length']
    #
    # old4 = sv_pds['d18'].loc[vid10][['frames_reached', 'length_reached']]
    # old4.columns = ['frames', 'length']
    # old4['fps'] = old4['frames']/old4['length']
    #
    # odict = odict_pds['d18'].loc[vid10][['nbframes_ffmpeg', 'duration', 'fps']]


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

def faster_sample_via_pyav(video_path, finds_to_sample, threading=True):
    # A bit broken
    margin = 1024
    skip_offset = 2048
    stream_name = {"video": 0}
    buffer_size = 0

    container = av.open(str(video_path))
    frames_length = container.streams.video[0].frames
    # how many time_base units
    duration = container.streams.video[0].duration
    timebase = int(duration/frames_length)
    pts_to_sample = timebase * finds_to_sample

    stream = container.streams.video[0]

    iter_pts_to_sample = iter([int(p) for p in pts_to_sample])
    next_pts = next(iter_pts_to_sample)
    # Initial seek
    seek_offset = max(next_pts - margin, 0)
    container.seek(seek_offset, any_frame=False,
            backward=True, stream=stream)
    if threading:
        container.streams.video[0].thread_type = 'AUTO'
    # Decode some
    buffer_count = -1
    pts_to_frames = {}
    pts_to_frames_packet = {}
    for packet in container.demux(**stream_name):
        if (next_pts - packet.pts) > skip_offset:
            continue
        for frame in packet.decode():
            pts_to_frames[frame.pts] = frame
            pts_to_frames_packet[frame.pts] = packet
        max_pts_to_frames = max(pts_to_frames, default=0)
        try:
            if max_pts_to_frames >= next_pts:
                next_pts = next(iter_pts_to_sample)
        except StopIteration:
            buffer_count += 1
        if buffer_count >= buffer_size:
            break
    container.close()

    pts_we_got = np.array(list(pts_to_frames.keys()))
    ssorted_indices = np.searchsorted(pts_we_got, pts_to_sample)

    sampled_frames = []
    for pts in pts_we_got[ssorted_indices]:
        sampled_frames.append(pts_to_frames[pts])

    sampled_frames_np = [frame.to_rgb().to_ndarray()
            for frame in sampled_frames]
    sampled_frames_np = np.stack(sampled_frames_np)
    return sampled_frames_np


def _perframe_detection_display(out, test_kfs, Y_conf_scores_sm, dataset):
    # Display our detections
    det2_fold = small.mkdir(out/'det2')
    state = np.random.RandomState(400)
    iter_index = state.permutation(np.arange(len(test_kfs)))[:400]
    for i, ii in enumerate(tqdm(iter_index)):
        kf = test_kfs[ii]
        scores = Y_conf_scores_sm[ii]
        vid = kf['vid']
        frame0 = kf['frame0']
        pred_box = kf['bbox']
        video_path = dataset.videos[vid]['path']
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(
                    vcap, [frame0], debug_filename=video_path)
        frame_u8 = frames_u8[0]
        act_scores = pd.Series(dict(zip(dataset.action_names, scores)))
        act_scores = act_scores.sort_values(ascending=False)[:3]
        stract = ' '.join([f'{x[:5]}: {y:.2f}' for x, y in act_scores.items()])
        rec_color = (0, 0, 80)
        good = kf['action_name'] == act_scores.index[0]
        if good:
            rec_color = (0, 80, 0)
        snippets.cv_put_box_with_text(frame_u8, pred_box,
            text='{}'.format(stract), rec_color=rec_color)
        cv2.imwrite(str(det2_fold/f'{i:05d}_{vid}_frame{frame0:05d}.jpg'), frame_u8)


def _tube_detection_display(out, av_stubes_, dataset):
    action = 'Drinking'
    vfold = small.mkdir(out/'det3_tube'/action)
    v_stubes = av_stubes_[action]
    flat_tubes = []
    for vid, stubes in v_stubes.items():
        for i_stube, stube in enumerate(stubes):
            flat_tubes.append({'tube': stube, 'ind': (vid, i_stube)})
    sorted_flat_tubes = sorted(flat_tubes,
            key=lambda x: x['tube']['score'], reverse=True)
    sorted_flat_tubes = sorted_flat_tubes[:10]

    for i_sorted, flat_tube in enumerate(tqdm(sorted_flat_tubes)):
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

        snippets.qsave_video(video_fold/'overlaid.mp4', drawn_frames_u8)
        break
