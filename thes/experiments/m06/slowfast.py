import numpy as np
import itertools
import pandas as pd
import logging
import time
import copy
from pathlib import Path
from types import MethodType
from typing import (Dict, Any)
from tqdm import tqdm
import cv2
import concurrent.futures
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (StandardScaler)
from sklearn.metrics import (
    average_precision_score, accuracy_score, roc_auc_score)
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.utils.data

import slowfast.models
import slowfast.utils.misc as misc
import slowfast.utils.checkpoint as cu

from detectron2.layers import ROIAlign

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv
from vsydorov_tools import log as vt_log

from thes.data.dataset.external import (
        Dataset_daly_ocv, Vid_daly,
        get_daly_split_vids, split_off_validation_set)
from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    get_daly_gt_tubes, push_into_avdict,
    AV_dict, loadconvert_tubes_dwein, Objaction_dets,
    dtindex_filter_split, av_stubes_above_score)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)
from thes.tools import snippets
from thes.slowfast.cfg import (base_sf_i3d_config)
from thes.data.tubes.routines import (
        score_ftubes_via_objaction_overlap_aggregation)

log = logging.getLogger(__name__)


def np_to_gpu(X):
    X = torch.from_numpy(np.array(X))
    X = X.type(torch.cuda.FloatTensor)
    return X


norm_mean = np.array([0.45, 0.45, 0.45])
norm_mean_t = np_to_gpu(norm_mean)
norm_std = np.array([0.225, 0.225, 0.225])
norm_std_t = np_to_gpu(norm_std)
test_crop_size = 255
# DETECTION.ROI_XFORM_RESOLUTION
xform_resolution = 7
# SPATIAL_SCALE_FACTOR
spatial_scale_factor = 16
i3d_poolsize = [[2, 1, 1]]
# [[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]]

def monkey_forward(self, x):
    x = self.s1(x)
    x = self.s2(x)
    for pathway in range(self.num_pathways):
        pool = getattr(self, "pathway{}_pool".format(pathway))
        x[pathway] = pool(x[pathway])
    x = self.s3(x)
    x = self.s4(x)
    x = self.s5(x)
    return x

class Extractor_roi(object):
    def __init__(self, model, model_nframes):
        self._model = copy.copy(model)
        self._model.forward = MethodType(monkey_forward, self._model)

        resolution = [xform_resolution] * 2
        # Definitions
        tpool_size = [model_nframes//i3d_poolsize[0][0], 1, 1]
        self.t_pool = nn.AvgPool3d(tpool_size, stride=1)
        self.roi_align = ROIAlign(resolution,
                spatial_scale=1.0/32,
                sampling_ratio=0,
                aligned=True)
        self.s_pool = nn.MaxPool2d(resolution, stride=1)

    def forward(self, X, bboxes):
        with torch.no_grad():
            # Forward through model
            x = self._model(X)
            # Forward
            assert len(x) == 1
            out = self.t_pool(x[0])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)
            out = self.roi_align(out, bboxes)
            out = self.s_pool(out)
        return out


def yana_size_query(X, dsize):
    # https://github.com/hassony2/torch_videovision
    def _get_resize_sizes(im_h, im_w, size):
        if im_w < im_h:
            ow = size
            oh = int(size * im_h / im_w)
        else:
            oh = size
            ow = int(size * im_w / im_h)
        return oh, ow

    if isinstance(dsize, int):
        im_h, im_w, im_c = X[0].shape
        new_h, new_w = _get_resize_sizes(im_h, im_w, dsize)
        isize = (new_w, new_h)
    else:
        assert len(dsize) == 2
        isize = dsize[1], dsize[0]
    return isize

def threaded_ocv_resize_clip(
        X, dsize, max_workers=8,
        interpolation=cv2.INTER_LINEAR):
    isize = yana_size_query(X, dsize)
    thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)
    futures = []
    for img in X:
        futures.append(thread_executor.submit(
            cv2.resize, img, isize,
            interpolation=interpolation))
    concurrent.futures.wait(futures)
    thread_executor.shutdown()
    scaled = np.array([x.result() for x in futures])
    return scaled


def tfm_video_resize_threaded(X, dsize, max_workers=8):
    # 256 resize, normalize, group,
    h_before, w_before = X.shape[1:3]
    X = threaded_ocv_resize_clip(X, dsize, max_workers)
    h_resized, w_resized = X.shape[1:3]
    params = {'h_before': h_before, 'w_before': w_before,
              'h_resized': h_resized, 'w_resized': w_resized}
    return X, params


def tfm_video_center_crop(first64, th, tw):
    h_before, w_before = first64.shape[1:3]
    ccrop_i = int((h_before-th)/2)
    ccrop_j = int((w_before-tw)/2)
    first64 = first64[:,
            ccrop_i:ccrop_i+th,
            ccrop_j:ccrop_j+tw, :]
    params = {'h_before': h_before, 'w_before': w_before,
              'i': ccrop_i, 'j': ccrop_j,
              'th': th, 'tw': tw}
    return first64, params


def prepare_video(frames_u8):
    frames_rgb = np.flip(frames_u8, -1)
    # Resize
    X, resize_params = tfm_video_resize_threaded(
            frames_rgb, test_crop_size)
    # Centercrop
    X, ccrop_params = tfm_video_center_crop(
            X, test_crop_size, test_crop_size)
    # Convert to torch, add batch dimension
    Xt = torch.from_numpy(X)
    return Xt, resize_params, ccrop_params

def to_gpu_normalize_permute(Xt):
    X_f32c = Xt.type(torch.cuda.FloatTensor)
    X_f32c /= 255
    # Normalization after float conversion
    X_f32c = (X_f32c-norm_mean_t)/norm_std_t
    # Pad 0 dim and permute done last
    assert len(X_f32c.shape) == 5
    X_f32c = X_f32c.permute(0, 4, 1, 2, 3)
    return X_f32c

def prepare_box(bbox_ltrd, resize_params, ccrop_params):
    # X is NCHW
    # Resize bbox
    bbox_tldr = bbox_ltrd[[1, 0, 3, 2]]
    real_scale_h = resize_params['h_resized']/resize_params['h_before']
    real_scale_w = resize_params['w_resized']/resize_params['w_before']
    real_scale = np.tile(np.r_[real_scale_h, real_scale_w], 2)
    bbox_tldr = (bbox_tldr * real_scale)
    # Offset box
    i, j = ccrop_params['i'], ccrop_params['j']
    bbox_tldr -= [i, j, i, j]
    box_maxsize = np.tile(
            np.r_[ccrop_params['th'], ccrop_params['tw']], 2)
    bbox_tldr = np.clip(bbox_tldr, [0, 0, 0, 0], box_maxsize)
    return bbox_tldr

def _vis_boxes(out, bbox, frames_u8, X_f32c, bbox_tldr):
    fullsize_w_boxes = frames_u8.copy()
    for i, frame in enumerate(fullsize_w_boxes):
        snippets.cv_put_box_with_text(frame, bbox)
    snippets.qsave_video(out/'fullsize_w_boxes.mp4', fullsize_w_boxes)

    small_w_boxes = np.flip(X_f32c.cpu().numpy()[0].transpose(1, 2, 3, 0), -1)
    small_w_boxes = small_w_boxes.copy()
    small_w_boxes = small_w_boxes*norm_std + norm_mean
    small_w_boxes = (small_w_boxes * 255).astype(np.uint8)
    for i, frame in enumerate(small_w_boxes):
        snippets.cv_put_box_with_text(frame, bbox_tldr[[1, 0, 3, 2]])
    snippets.qsave_video(out/'small_w_boxes.mp4', small_w_boxes)


class TDataset_over_keyframes(torch.utils.data.Dataset):
    def __init__(self, keyframes, model_nframes, model_sample):
        self.keyframes = keyframes
        center_frame = (model_nframes-1)//2
        self.sample_grid0 = (np.arange(model_nframes)-center_frame)*model_sample

    def __getitem__(self, index):
        keyframe = self.keyframes[index]
        video_path = keyframe['video_path']
        i0 = keyframe['frame0']
        finds_to_sample = i0 + self.sample_grid0
        finds_to_sample = np.clip(
                finds_to_sample, 0, keyframe['nframes']-1)
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(vcap, finds_to_sample)
        frames_u8 = np.array(frames_u8)

        # frames_u8 is N,H,W,C
        Xt, resize_params, ccrop_params = prepare_video(frames_u8)
        bbox_tldr = prepare_box(keyframe['bbox'], resize_params, ccrop_params)
        bbox_tldr0 = np.r_[0, bbox_tldr]
        return index, Xt, bbox_tldr0

    def __len__(self) -> int:
        return len(self.keyframes)


def create_keyframelist(dataset):
    # Record keyframes
    keyframes = []
    for vid, ovideo in dataset.videos_ocv.items():
        nframes = ovideo['nframes']
        for action_name, instances in ovideo['instances'].items():
            for ins_ind, instance in enumerate(instances):
                fl = instance['flags']
                diff = fl['isReflection'] or fl['isAmbiguous']
                if diff:
                    continue
                for kf_ind, keyframe in enumerate(instance['keyframes']):
                    frame0 = keyframe['frame']
                    action_id = dataset.action_names.index(action_name)
                    kf_dict = {
                            'vid': vid,
                            'action_id': action_id,
                            'action_name': action_name,
                            'ins_ind': ins_ind,
                            'kf_ind': kf_ind,
                            'bbox': keyframe['bbox_abs'],
                            'video_path': ovideo['path'],
                            'frame0': int(frame0),
                            'nframes': nframes,
                            'height': ovideo['height'],
                            'width': ovideo['width'],
                            }
                    keyframes.append(kf_dict)
    return keyframes


class Dataloader_isaver(
        snippets.Isaver_mixin_restore_save, snippets.Isaver_base):
    """
    Will process a list with a 'func', 'prepare_func(start_i)' is to be run before processing
    """
    def __init__(self, folder,
            total, func, prepare_func,
            save_every=0,
            save_interval=120,  # every 2 minutes by default
            log_interval=None,):
        super().__init__(folder, total)
        self.func = func
        self.prepare_func = prepare_func
        self._save_every = save_every
        self._save_interval = save_interval
        self._log_interval = log_interval
        self.result = []

    def run(self):
        i_last = self._restore()
        self._time_last_save = time.perf_counter()
        self._time_last_log = time.perf_counter()
        self._i_last_saved = 0

        Ys_np = []

        def flush_purge():
            self.result.extend(Ys_np)
            Ys_np.clear()
            self._save(i_last)
            self._purge_intermediate_files()

        loader = self.prepare_func(i_last)
        pbar = tqdm(loader, total=len(loader))
        for i_batch, data_input in enumerate(pbar):
            II_np, Y_np = self.func(data_input)
            Ys_np.append(Y_np)
            i_last = II_np[-1]
            PURGE = False
            if self._save_every > 0:
                PURGE |= (i_last - self._i_last_saved) >= self._save_every
            if self._save_interval:
                since_last_save = time.perf_counter() - self._time_last_save
                PURGE |= since_last_save > self._save_interval
            if PURGE:
                flush_purge()
                self._time_last_save = time.perf_counter()
            if self._log_interval:
                since_last_log = time.perf_counter() - self._time_last_log
                if since_last_log > self._log_interval:
                    log.info(snippets._tqdm_str(pbar))
                    self._time_last_log = time.perf_counter()
        flush_purge()
        return self.result

def extract_slowfast_feats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
        subset: ['train', ~]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    # split_label = cf['dataset.subset']
    # split_vids = get_daly_split_vids(dataset, split_label)

    sf_cfg = base_sf_i3d_config()
    sf_cfg.NUM_GPUS = 1
    sf_cfg.TEST.BATCH_SIZE = 16
    # Load model
    model = slowfast.models.build_model(sf_cfg)
    model.eval()
    # misc.log_model_info(model, sf_cfg, is_train=False)

    CHECKPOINT_FILE_PATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_checkpoints/I3D_8x8_R50.pkl'
    with vt_log.logging_disabled(logging.WARNING):
        cu.load_checkpoint(
            CHECKPOINT_FILE_PATH, model, False, None,
            inflation=False, convert_from_caffe2=True,)

    keyframes = create_keyframelist(dataset)

    model_nframes = sf_cfg.DATA.NUM_FRAMES
    model_sample = sf_cfg.DATA.SAMPLING_RATE
    extractor_roi = Extractor_roi(model, model_nframes)

    BATCH_SIZE = 8
    NUM_WORKERS = 12

    def prepare_func(start_i):
        remaining_keyframes = keyframes[start_i+1:]
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes, model_nframes, model_sample)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True)
        return loader

    def func(data_input):
        II, Xts, bboxes = data_input
        Xs_f32c = to_gpu_normalize_permute(Xts)
        bboxes_c = bboxes.type(torch.cuda.FloatTensor)
        Y = extractor_roi.forward([Xs_f32c], bboxes_c)
        II_np = II.cpu().numpy()
        Y_np = Y.cpu().numpy()
        return II_np, Y_np

    disaver_fold = small.mkdir(out/'disaver')
    total = len(keyframes)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
            save_interval=60)
    outputs = disaver.run()
    sq_outputs = np.vstack(outputs).squeeze()
    small.save_pkl(out/'sq_outputs.pkl', sq_outputs)
    small.save_pkl(out/'keyframes.pkl', keyframes)


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

def create_kinda_objaction_struct(dataset, test_kfs, Y_conf_scores_sm):
    # // Creating kinda objaction structure
    # Group vid -> frame
    grouped_kfscores_vf: Dict[Vid_daly, Dict[int, Any]] = {}
    for kf, scores in zip(test_kfs, Y_conf_scores_sm):
        vid = kf['vid']
        frame0 = kf['frame0']
        pred_box = kf['bbox']
        (grouped_kfscores_vf
                .setdefault(vid, {})
                .setdefault(frame0, [])
                .append([pred_box, scores]))
    # fake objactions
    objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]] = {}
    for vid, grouped_kfscores_f in grouped_kfscores_vf.items():
        for frame_ind, gkfscores in grouped_kfscores_f.items():
            all_scores, all_boxes, all_classes = [], [], []
            for (box, scores) in gkfscores:
                all_boxes.append(np.tile(box, (len(scores), 1)))
                all_classes.append(np.array(dataset.action_names))
                all_scores.append(scores)
            all_scores_ = np.hstack(all_scores)
            all_classes_ = np.hstack(all_classes)
            all_boxes_ = np.vstack(all_boxes)
            detections = {
                    'pred_boxes': all_boxes_,
                    'scores': all_scores_,
                    'pred_classes': all_classes_}
            objactions_vf.setdefault(vid, {})[frame_ind] = detections
    return objactions_vf


def train_mlp_over_extracted_feats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
        subset: ['train', ~]
    tubes_dwein: [~, str]
    computed_featfold: [~, str]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    val_vids, train_vids = split_off_validation_set(dataset, 0.1)
    trainval_vids = get_daly_split_vids(dataset, 'train')
    test_vids = get_daly_split_vids(dataset, 'test')

    computed_featfold = Path(cf['computed_featfold'])
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'sq_outputs.pkl')
    kf_vids = [kf['vid'] for kf in keyframes]

    def split_off(X, linked_vids, good_vids):
        if isinstance(X, np.ndarray):
            isin = np.in1d(linked_vids, good_vids)
            result = X[isin]
        elif isinstance(X, list):
            result = [x for x, v in zip(X, linked_vids) if v in good_vids]
        else:
            raise RuntimeError()
        return result

    X_trainval = split_off(outputs, kf_vids, trainval_vids)
    kf_trainval = split_off(keyframes, kf_vids, trainval_vids)
    kf_vids_trainval = [kf['vid'] for kf in kf_trainval]
    aid_trainval = [kf['action_id'] for kf in kf_trainval]

    scaler = StandardScaler()
    X_trainval = scaler.fit_transform(X_trainval)

    X_test = split_off(outputs, kf_vids, test_vids)
    kf_test = split_off(keyframes, kf_vids, test_vids)
    kf_vids_test = [kf['vid'] for kf in kf_test]
    aid_test = [kf['action_id'] for kf in kf_test]
    X_test = scaler.transform(X_test)

    X_train = split_off(X_trainval, kf_vids_trainval, train_vids)
    kf_train = split_off(kf_trainval, kf_vids_trainval, train_vids)
    aid_train = [kf['action_id'] for kf in kf_train]

    X_val = split_off(X_trainval, kf_vids_trainval, val_vids)
    kf_val = split_off(kf_trainval, kf_vids_trainval, val_vids)
    aid_val = [kf['action_id'] for kf in kf_val]

    def compute_perf(IN):
        alpha, hl_size, max_iter = IN
        clf = MLPClassifier(hl_size, random_state=0, max_iter=max_iter, alpha=alpha)
        clf.fit(X_train, aid_train)
        Y_val = clf.predict(X_val)
        acc = accuracy_score(aid_val, Y_val)
        return acc

    alpha_values = np.logspace(-3, -1, 7)
    hlayer_sizes = ((200,), (250,))
    max_iter = (200, 400, 600)
    params = list(itertools.product(alpha_values, hlayer_sizes, max_iter))

    fold = small.mkdir(out/'isaver_ahl_perf')
    isaver = snippets.Isaver_threading(fold, params, compute_perf, 2, 24)
    perfs = isaver.run()

    # Train on trainval
    clf = MLPClassifier((200,), random_state=0, max_iter=300, alpha=0.02)
    clf.fit(X_trainval, aid_trainval)
    Y_test = clf.predict(X_test)
    acc = accuracy_score(aid_test, Y_test)
    log.info(f'Test {acc=}')

    Y_conf_scores_sm = clf.predict_proba(X_test)
    roc_auc = roc_auc_score(aid_test, Y_conf_scores_sm,
            multi_class='ovr')
    log.info(f'{roc_auc=}')

    # // Tube AP (very, very rought performance)
    # Dwein tubes
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    tubes_dwein_test = dtindex_filter_split(tubes_dwein, test_vids)
    # GT tubes
    dgt_tubes: Dict[I_dgt, T_dgt] = \
            get_daly_gt_tubes(dataset)
    dgt_tubes_test = dtindex_filter_split(dgt_tubes, test_vids)
    av_gt_tubes_test: AV_dict[T_dgt] = push_into_avdict(dgt_tubes_test)

    objactions_vf = create_kinda_objaction_struct(
            dataset, kf_test, Y_conf_scores_sm)
    # Assigning scores based on intersections
    av_stubes: AV_dict[T_dwein_scored] = \
        score_ftubes_via_objaction_overlap_aggregation(
            dataset, objactions_vf, tubes_dwein_test, 'iou',
            0.1, 0.0)
    av_stubes_ = av_stubes_above_score(
            av_stubes, 0.0)
    av_stubes_ = compute_nms_for_av_stubes(
            av_stubes_, 0.3)
    iou_thresholds = [.3, .5, .7]
    df_recall_s_nodiff = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes_test, av_stubes_, iou_thresholds, False)[0]
    df_ap_s_nodiff = compute_ap_for_avtubes_as_df(
            av_gt_tubes_test, av_stubes_, iou_thresholds, False, False)
    log.info(f'AP:\n{df_ap_s_nodiff}')


def train_extracted_feats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
        subset: ['train', ~]
    tubes_dwein: [~, str]
    computed_featfold: [~, str]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    trainval_vids = get_daly_split_vids(dataset, 'train')
    val_vids, train_vids = split_off_validation_set(dataset, 0.1)
    test_vids = get_daly_split_vids(dataset, 'test')


    computed_featfold = Path(cf['computed_featfold'])
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'sq_outputs.pkl')
    N = len(keyframes)


    train_feats = []
    train_kfs = []
    test_feats = []
    test_kfs = []
    for keyframe, feat in zip(keyframes, outputs):
        if keyframe['vid'] in train_vids:
            train_feats.append(feat)
            train_kfs.append(keyframe)
        if keyframe['vid'] in test_vids:
            test_feats.append(feat)
            test_kfs.append(keyframe)

    train_feats = np.array(train_feats)
    train_aids = np.array([x['action_id'] for x in train_kfs])

    test_feats = np.array(test_feats)
    test_aids = np.array([x['action_id'] for x in test_kfs])

    def scale_clf():
        scaler = StandardScaler()
        scaler.fit(train_feats)
        clf = LinearSVC(verbose=1, max_iter=2000)
        clf.fit(X_train, train_aids)
        return scaler, clf

    scaler, clf = small.stash2(out/'scaler_clf.pkl')(scale_clf)

    X_train = scaler.transform(train_feats)
    X_test = scaler.transform(test_feats)
    Y_test = clf.predict(X_test)

    Y_conf_scores = clf.decision_function(X_test)
    Y_conf_scores_sm = softmax(Y_conf_scores, axis=1)

    acc = accuracy_score(test_aids, Y_test)
    log.info(f'{acc=}')
    roc_auc = roc_auc_score(test_aids, Y_conf_scores_sm,
            multi_class='ovr')
    log.info(f'{roc_auc=}')
