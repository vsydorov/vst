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
import torch.nn.functional as F

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
from thes.slowfast.cfg import (basic_sf_cfg)
from thes.data.tubes.routines import (
        score_ftubes_via_objaction_overlap_aggregation)

log = logging.getLogger(__name__)


def np_to_gpu(X):
    X = torch.from_numpy(np.array(X))
    X = X.type(torch.cuda.FloatTensor)
    return X


norm_mean = np.array([0.45, 0.45, 0.45])
norm_std = np.array([0.225, 0.225, 0.225])
test_crop_size = 256

def sf_resnet_headless_forward(self, x):
    # slowfast/models/video_model_builder.py/ResNet.forward
    x = self.s1(x)
    x = self.s2(x)
    for pathway in range(self.num_pathways):
        pool = getattr(self, "pathway{}_pool".format(pathway))
        x[pathway] = pool(x[pathway])
    x = self.s3(x)
    x = self.s4(x)
    x = self.s5(x)
    return x

def sf_slowfast_headless_forward(self, x):
    # slowfast/models/video_model_builder.py/SlowFast.forward
    x = self.s1(x)
    x = self.s1_fuse(x)
    x = self.s2(x)
    x = self.s2_fuse(x)
    for pathway in range(self.num_pathways):
        pool = getattr(self, "pathway{}_pool".format(pathway))
        x[pathway] = pool(x[pathway])
    x = self.s3(x)
    x = self.s3_fuse(x)
    x = self.s4(x)
    x = self.s4_fuse(x)
    x = self.s5(x)
    return x

class Extractor_roi(nn.Module):
    def __init__(self,
            model, forward_method,
            pool_size, resolution, scale_factor):
        super().__init__()
        self._model = copy.copy(model)
        self._model.forward = MethodType(forward_method, self._model)
        self.num_pathways = len(pool_size)

        for pi in range(self.num_pathways):
            tpool = nn.AvgPool3d(
                    [pool_size[pi][0], 1, 1], stride=1)
            self.add_module(f's{pi}_tpool', tpool)
            roi_align = ROIAlign(
                    resolution[pi],
                    spatial_scale=1.0/scale_factor[pi],
                    sampling_ratio=0,
                    aligned=True)
            self.add_module(f's{pi}_roi', roi_align)
            spool = nn.MaxPool2d(resolution[pi], stride=1)
            self.add_module(f's{pi}_spool', spool)

    def forward(self, X, bboxes):
        # Forward through model
        x = self._model(X)
        # / Roi_Pooling
        pool_out = []
        for pi in range(self.num_pathways):
            t_pool = getattr(self, f's{pi}_tpool')
            out = t_pool(x[pi])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            roi_align = getattr(self, f's{pi}_roi')
            out = roi_align(out, bboxes)

            s_pool = getattr(self, f's{pi}_spool')
            out = s_pool(out)
            pool_out.append(out)
        # B C H W.
        x = torch.cat(pool_out, 1)
        x = x.view(x.shape[0], -1)
        result = {'roipooled': x}
        return result


class Extractor_fullframe(nn.Module):
    def __init__(self,
            model, forward_method,
            temp_pool_size, spat_pool_size):
        super().__init__()
        self._model = copy.copy(model)
        self._model.forward = MethodType(forward_method, self._model)
        self.num_pathways = len(temp_pool_size)

        for pi in range(self.num_pathways):
            # Avg.tpool, then Max.spool
            tpool = nn.AvgPool3d((temp_pool_size[pi], 1, 1))
            self.add_module(f's{pi}_tpool', tpool)
            spool = nn.MaxPool2d(
                    (spat_pool_size[pi], spat_pool_size[pi]), stride=1)
            self.add_module(f's{pi}_spool', spool)
            # Avg.stpool
            stpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.add_module(f's{pi}_stpool', stpool)

    def forward(self, X, bboxes):
        # Forward through model
        x = self._model(X)
        tavg_smax_pool_out = []
        st_avg_pool_out = []
        for pi in range(self.num_pathways):
            # Avg.tpool, Max.spool
            t_pool = getattr(self, f's{pi}_tpool')
            s_pool = getattr(self, f's{pi}_spool')
            out1 = t_pool(x[pi])
            assert out1.shape[2] == 1
            out1 = torch.squeeze(out1, 2)
            out1 = s_pool(out1)
            tavg_smax_pool_out.append(out1)
            # Avg. stpool
            st_pool = getattr(self, f's{pi}_stpool')
            out2 = st_pool(x[pi])
            st_avg_pool_out.append(out2)
        tavg_smax = torch.cat(tavg_smax_pool_out, 1)
        tavg_smax = tavg_smax.view(tavg_smax.shape[0], -1)
        st_avg = torch.cat(st_avg_pool_out, 1)
        st_avg = st_avg.view(st_avg.shape[0], -1)
        result = {'tavg_smax': tavg_smax, 'st_avg': st_avg}
        return result


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
    # frames_u8 is N,H,W,C (BGR)
    frames_rgb = np.flip(frames_u8, -1)
    # Resize
    X, resize_params = tfm_video_resize_threaded(
            frames_rgb, test_crop_size)
    # Centercrop
    X, ccrop_params = tfm_video_center_crop(
            X, test_crop_size, test_crop_size)
    # Convert to torch
    Xt = torch.from_numpy(X)
    return Xt, resize_params, ccrop_params

def to_gpu_normalize_permute(Xt, norm_mean_t, norm_std_t):
    # Convert to float on GPU
    X_f32c = Xt.type(torch.cuda.FloatTensor)
    X_f32c /= 255
    # Normalize
    X_f32c = (X_f32c-norm_mean_t)/norm_std_t
    # THWC -> CTHW
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
    def __init__(self, keyframes, model_nframes, model_sample,
            is_slowfast: bool, slowfast_alpha: int):
        self.keyframes = keyframes
        center_frame = (model_nframes-1)//2
        self.sample_grid0 = (np.arange(model_nframes)-center_frame)*model_sample
        self._is_slowfast = is_slowfast
        self._slowfast_alpha = slowfast_alpha

    def __getitem__(self, index):
        # Extract frames
        keyframe = self.keyframes[index]
        video_path = keyframe['video_path']
        i0 = keyframe['frame0']
        finds_to_sample = i0 + self.sample_grid0
        finds_to_sample = np.clip(
                finds_to_sample, 0, keyframe['nframes']-1)
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(vcap, finds_to_sample)
        frames_u8 = np.array(frames_u8)

        # Resize video, convert to torch tensor
        Xt, resize_params, ccrop_params = prepare_video(frames_u8)

        # Resolve pathways
        if self._is_slowfast:
            # slowfast/datasets/utils.py/pack_pathway_output
            TIME_DIM = 0
            fast_pathway = Xt
            slow_pathway = torch.index_select(
                Xt,
                TIME_DIM,
                torch.linspace(
                    0, Xt.shape[TIME_DIM] - 1,
                    Xt.shape[TIME_DIM] // self._slowfast_alpha
                ).long(),
            )
            frame_list = [slow_pathway, fast_pathway]
        else:
            frame_list = [Xt]

        # Bboxes
        bbox_tldr = prepare_box(keyframe['bbox'], resize_params, ccrop_params)
        bbox_tldr0 = np.r_[0, bbox_tldr]
        return index, frame_list, bbox_tldr0

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

def extract_sf_feats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
    model: [~, ['slowfast', 'i3d']]
    extraction_mode: ['roi', ['roi', 'fullframe']]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    model_str = cf['model']
    if model_str == 'slowfast':
        rel_yml_path = 'Kinetics/c2/SLOWFAST_4x16_R50.yaml'
        CHECKPOINT_FILE_PATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_zoo_checkpoints/kinetics400/SLOWFAST_4x16_R50.pkl'
        headless_forward_func = sf_slowfast_headless_forward
        is_slowfast = True
        _POOL_SIZE = [[1, 1, 1], [1, 1, 1]]
    elif model_str == 'i3d':
        rel_yml_path = 'Kinetics/c2/I3D_8x8_R50.yaml'
        CHECKPOINT_FILE_PATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_zoo_checkpoints/kinetics400/I3D_8x8_R50.pkl'
        headless_forward_func = sf_resnet_headless_forward
        is_slowfast = False
        _POOL_SIZE = [[2, 1, 1]]
    else:
        raise RuntimeError()

    # DETECTION.ROI_XFORM_RESOLUTION
    xform_resolution = 7

    # / Config and derived things
    sf_cfg = basic_sf_cfg(rel_yml_path)
    sf_cfg.NUM_GPUS = 1
    model_nframes = sf_cfg.DATA.NUM_FRAMES
    model_sample = sf_cfg.DATA.SAMPLING_RATE
    slowfast_alpha = sf_cfg.SLOWFAST.ALPHA

    # Load model
    model = slowfast.models.build_model(sf_cfg)
    model.eval()
    with vt_log.logging_disabled(logging.WARNING):
        cu.load_checkpoint(
            CHECKPOINT_FILE_PATH, model, False, None,
            inflation=False, convert_from_caffe2=True,)

    keyframes = create_keyframelist(dataset)

    if cf['extraction_mode'] == 'roi':
        if model_str == 'slowfast':
            head_pool_size = [
                [model_nframes//slowfast_alpha//_POOL_SIZE[0][0], 1, 1],
                [model_nframes//_POOL_SIZE[1][0], 1, 1]]
            resolution = [[xform_resolution] * 2] * 2
            scale_factor = [32] * 2
        elif model_str == 'i3d':
            head_pool_size = [
                    [model_nframes//_POOL_SIZE[0][0], 1, 1]]
            resolution = [[xform_resolution] * 2]
            scale_factor= [32]
        else:
            raise RuntimeError()
        extractor = Extractor_roi(
            model, headless_forward_func,
            head_pool_size, resolution, scale_factor)
    elif cf['extraction_mode'] == 'fullframe':
        if model_str == 'slowfast':
            temp_pool_size = [
                model_nframes//slowfast_alpha//_POOL_SIZE[0][0],
                model_nframes//_POOL_SIZE[1][0]]
            spat_pool_size = [test_crop_size//32, test_crop_size//32]
        elif model_str == 'i3d':
            temp_pool_size = [
                    model_nframes//_POOL_SIZE[0][0], ]
            spat_pool_size = [test_crop_size//32]
        else:
            raise RuntimeError()
        extractor = Extractor_fullframe(
            model, headless_forward_func,
            temp_pool_size, spat_pool_size)
    else:
        raise RuntimeError()

    BATCH_SIZE = 8
    NUM_WORKERS = 12
    # NUM_WORKERS = 0

    norm_mean_t = np_to_gpu(norm_mean)
    norm_std_t = np_to_gpu(norm_std)

    def prepare_func(start_i):
        remaining_keyframes = keyframes[start_i+1:]
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes, model_nframes, model_sample,
                is_slowfast, slowfast_alpha)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True)
        return loader

    def func(data_input):
        II, Xts, bboxes = data_input
        Xts_f32c = [to_gpu_normalize_permute(
            x, norm_mean_t, norm_std_t) for x in Xts]
        bboxes_c = bboxes.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            result = extractor.forward(Xts_f32c, bboxes_c)
        result_np = {k: v.cpu().numpy()
                for k, v in result.items()}
        II_np = II.cpu().numpy()
        return II_np, result_np

    disaver_fold = small.mkdir(out/'disaver')
    total = len(keyframes)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
            save_interval=60)
    outputs = disaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        stacked = np.vstack([o[k] for o in outputs])
        dict_outputs[k] = stacked
    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    small.save_pkl(out/'keyframes.pkl', keyframes)


from thes.caffe import (nicolas_net, model_test_get_image_blob)


def _caffe_feat_extract_func(net, keyframe):
    PIXEL_MEANS = [102.9801, 115.9465, 122.7717]
    TEST_SCALES = [600, ]
    TEST_MAX_SIZE = 1000

    i0 = keyframe['frame0']
    video_path = keyframe['video_path']

    boxes = keyframe['bbox'][None]

    with vt_cv.video_capture_open(video_path) as vcap:
        frame_u8 = vt_cv.video_sample(vcap, [i0])[0]

    blob_, im_scale_factors = model_test_get_image_blob(
            frame_u8, PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE)
    blob = blob_.transpose(0, 3, 1, 2)  # 1, H, W, 3 --> 1, 3, H, W
    im_scale_factor = im_scale_factors[0]

    net.blobs['data'].reshape(*blob.shape)
    net.blobs['data'].data[...] = blob
    sc_boxes = boxes * im_scale_factor
    boxes5 = np.c_[np.zeros(len(sc_boxes)), sc_boxes]

    net.blobs['rois'].reshape(len(boxes5), 5)
    net.blobs['rois'].data[...] = boxes5
    net_forwarded = net.forward()

    # 11
    cls_prob = net_forwarded['cls_prob']
    cls_prob_copy = cls_prob.copy().squeeze()

    # 4096
    fc7_feats = net.blobs['fc7'].data.copy().squeeze()

    # pool5 (spatial max-pool), 5012
    pool5_feats = net.blobs['pool5'].data.copy().squeeze()
    pool5_feats = pool5_feats.max(axis=(1, 2))

    feats = {
            'cls_prob': cls_prob_copy,
            'fc7': fc7_feats,
            'pool5_maxpool': pool5_feats}
    return feats


def extract_caffe_rcnn_feats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    net = nicolas_net()

    keyframes = create_keyframelist(dataset)

    def extract(i):
        keyframe = keyframes[i]
        feats = _caffe_feat_extract_func(net, keyframe)
        return feats

    isaver = snippets.Isaver_simple(
        small.mkdir(out/'isaver_extract'), range(len(keyframes)), extract)
    outputs = isaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        stacked = np.vstack([o[k] for o in outputs])
        dict_outputs[k] = stacked
    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
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


def split_off(X, linked_vids, good_vids):
    if isinstance(X, np.ndarray):
        isin = np.in1d(linked_vids, good_vids)
        result = X[isin]
    elif isinstance(X, list):
        result = [x for x, v in zip(X, linked_vids) if v in good_vids]
    else:
        raise RuntimeError()
    return result

class Ncfg_mlp:

    class Net_mlp_featcls(nn.Module):
        def __init__(self, D_in, D_out, H, dropout_rate=0.5):
            super().__init__()
            self.linear1 = nn.Linear(D_in, H)
            self.linear2 = nn.Linear(H, D_out)
            self.dropout = nn.Dropout(dropout_rate)
            # self.bn = nn.BatchNorm1d(H, momentum=0.1)

        def forward(self, x):
            x = self.linear1(x)
            x = F.relu(x)
            x = self.dropout(x)
            # x = self.bn(x)
            x = self.linear2(x)
            return x

    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        seed: [42, int]
        dataset:
            name: ['daly', ['daly']]
            cache_folder: [~, str]
            val_split:
                fraction: [0.1, float]
                nsamplings: [20, int]
                seed: [42, int]
        tubes_dwein: [~, str]
        inputs:
            featfold: [~, str]
            featname: [~, ~]
        """)
        cfg.set_defaults("""
        net:
            H: 32
        train:
            lr: 1.0e-5
            weight_decay: 5e-2
            niters: 2000
        n_trials: 5
        log_period: '::'
        """)

    @staticmethod
    def get_vids(cf, dataset):
        v_fraction = cf['dataset.val_split.fraction']
        v_nsamplings = cf['dataset.val_split.nsamplings']
        v_seed = cf['dataset.val_split.seed']

        class Vgroup:
            val, train = split_off_validation_set(
                    dataset, v_fraction, v_nsamplings, v_seed)
            trainval = get_daly_split_vids(dataset, 'train')
            test = get_daly_split_vids(dataset, 'test')
        return Vgroup

    @staticmethod
    def split_features(outputs, keyframes, Vgroup):
        global_kf_vids = [kf['vid'] for kf in keyframes]

        class D_trainval():
            X = split_off(outputs, global_kf_vids, Vgroup.trainval)
            kf = split_off(keyframes, global_kf_vids, Vgroup.trainval)
            kf_vids = [kf['vid'] for kf in kf]
            Y = np.array([kf['action_id'] for kf in kf])

        scaler = StandardScaler()
        D_trainval.X = scaler.fit_transform(D_trainval.X)

        class D_train():
            X = split_off(D_trainval.X, D_trainval.kf_vids, Vgroup.train)
            kf = split_off(D_trainval.kf, D_trainval.kf_vids, Vgroup.train)
            Y = np.array([kf['action_id'] for kf in kf])

        class D_val():
            X = split_off(D_trainval.X, D_trainval.kf_vids, Vgroup.val)
            kf = split_off(D_trainval.kf, D_trainval.kf_vids, Vgroup.val)
            Y = np.array([kf['action_id'] for kf in kf])

        class D_test():
            X = split_off(outputs, global_kf_vids, Vgroup.test)
            kf = split_off(keyframes, global_kf_vids, Vgroup.test)
            Y = np.array([kf['action_id'] for kf in kf])
            X = scaler.transform(X)

        return D_trainval, D_train, D_val, D_test

    @classmethod
    def create_model(self, dataset, outputs, H):
        D_in = outputs.shape[-1]
        D_out = len(dataset.action_names)
        H = 32
        model = self.Net_mlp_featcls(D_in, D_out, H)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        return model, loss_fn

    @staticmethod
    def qacc(pred, Y):
        return pred.argmax(1).eq(Y).sum().item()/len(Y) * 100

    @classmethod
    def optimize(self, cf, model, loss_fn, D_to_train, log_period='::'):
        optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])
        niters = cf['train.niters']
        X_to_train = torch.from_numpy(D_to_train.X)
        Y_to_train = torch.from_numpy(D_to_train.Y)
        model.train()
        for t in range(niters):
            pred_train = model(X_to_train)
            loss = loss_fn(pred_train, Y_to_train)
            if snippets.check_step_sslice(t, log_period):
                # Train perf
                model.eval()
                with torch.no_grad():
                    pred_trainval = model(X_to_train)
                    acc_trainval = self.qacc(pred_trainval, Y_to_train)
                model.train()
                log.info(f'{t}: {loss.item()} {acc_trainval=:.2f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @staticmethod
    def evaluate_acc(model, D_to_test):
        X_to_test = torch.from_numpy(D_to_test.X)

        model.eval()
        with torch.no_grad():
            Y_test = model(X_to_test)
        Y_test_np = Y_test.numpy()
        Y_test_preds = np.argmax(Y_test_np, axis=1)
        acc = accuracy_score(D_to_test.Y, Y_test_preds)

        Y_test_np_softmax = softmax(Y_test_np, axis=1)
        Y_test_np_softmax = Y_test_np_softmax.copy()
        roc_auc = roc_auc_score(D_to_test.Y, Y_test_np_softmax,
                multi_class='ovr')
        return Y_test_np_softmax, acc, roc_auc

    @staticmethod
    def evaluate_tubes(cf, Vgroup, D_test, dataset, Y_test_np_softmax):
        # // Tube AP (very, very rought performance)
        # Dwein tubes
        tubes_dwein: Dict[I_dwein, T_dwein] = \
                loadconvert_tubes_dwein(cf['tubes_dwein'])
        tubes_dwein_test = dtindex_filter_split(tubes_dwein, Vgroup.test)
        # GT tubes
        dgt_tubes: Dict[I_dgt, T_dgt] = \
                get_daly_gt_tubes(dataset)
        dgt_tubes_test = dtindex_filter_split(dgt_tubes, Vgroup.test)
        av_gt_tubes_test: AV_dict[T_dgt] = push_into_avdict(dgt_tubes_test)

        objactions_vf = create_kinda_objaction_struct(
                dataset, D_test.kf, Y_test_np_softmax)
        # Assigning scores based on intersections
        av_stubes: AV_dict[T_dwein_scored] = \
            score_ftubes_via_objaction_overlap_aggregation(
                dataset, objactions_vf, tubes_dwein_test, 'iou',
                0.1, 0.0, enable_tqdm=False)
        av_stubes_ = av_stubes_above_score(
                av_stubes, 0.0)
        av_stubes_ = compute_nms_for_av_stubes(
                av_stubes_, 0.3)
        iou_thresholds = [.3, .5, .7]
        df_recall_s_nodiff = compute_recall_for_avtubes_as_dfs(
                av_gt_tubes_test, av_stubes_, iou_thresholds, False)[0]
        df_ap_s_nodiff = compute_ap_for_avtubes_as_df(
                av_gt_tubes_test, av_stubes_, iou_thresholds, False, False)
        return df_recall_s_nodiff, df_ap_s_nodiff


def torchmlp_feat_classify_validate(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_mlp.set_defcfg(cfg)
    cf = cfg.parse()

    # Inputs
    initial_seed = cf['seed']
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_mlp.get_vids(cf, dataset)
    computed_featfold = Path(cf['inputs.featfold'])
    featname = cf['inputs.featname']
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'dict_outputs.pkl')[featname]
    D_trainval, D_train, D_val, D_test = \
            Ncfg_mlp.split_features(outputs, keyframes, Vgroup)

    torch.manual_seed(initial_seed)
    model, loss_fn = Ncfg_mlp.create_model(dataset, outputs, cf['net.H'])

    # model, loss_fn = Ncfg_mlp.create_model(cf['net.H'])


def torchmlp_feat_classify_test(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_mlp.set_defcfg(cfg)
    cf = cfg.parse()

    # Inputs
    initial_seed = cf['seed']
    n_trials = cf['n_trials']
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_mlp.get_vids(cf, dataset)
    computed_featfold = Path(cf['inputs.featfold'])
    featname = cf['inputs.featname']
    log_period = cf['log_period']
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'dict_outputs.pkl')[featname]
    D_trainval, D_train, D_val, D_test = \
            Ncfg_mlp.split_features(outputs, keyframes, Vgroup)

    def experiment(i):
        torch.manual_seed(initial_seed+i)
        model, loss_fn = Ncfg_mlp.create_model(dataset, outputs, cf['net.H'])
        Ncfg_mlp.optimize(cf, model, loss_fn, D_trainval, log_period)
        Y_test_np_softmax, acc, roc_auc = Ncfg_mlp.evaluate_acc(model, D_test)
        df_recall_s_nodiff, df_ap_s_nodiff = Ncfg_mlp.evaluate_tubes(
                cf, Vgroup, D_test, dataset, Y_test_np_softmax)
        return [acc, roc_auc, df_recall_s_nodiff, df_ap_s_nodiff]

    isaver = snippets.Isaver_simple(small.mkdir(out/'isaver_ntrials'), range(n_trials), experiment)
    trial_results = isaver.run()
    [acc, roc_auc, recall, ap] = zip(*trial_results)
    acc, roc_auc = map(lambda x: np.array(x)*100, (acc, roc_auc))
    recall, ap = map(lambda x: pd.concat(
        x, keys=range(5), axis=1).mean(axis=1, level=1), (recall, ap))
    with small.np_printoptions(precision=2):
        log.info('Accuracy; mean: {:.2f}, all: {} '.format(np.mean(acc), acc))
        log.info('Roc_auc; mean: {:.2f} all: {} '.format(np.mean(roc_auc), roc_auc))
    log.info(f'Mean Recall\n{recall}')
    log.info(f'mean AP:\n{ap}')
