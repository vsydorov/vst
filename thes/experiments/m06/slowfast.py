import numpy as np
import pprint
import pandas as pd
import logging
import time
import copy
from pathlib import Path
from types import MethodType
from typing import (  # NOQA
        Dict, Any, List, Optional, Tuple, TypedDict, Set)
from tqdm import tqdm
import cv2
import av
import concurrent.futures
from sklearn.preprocessing import (StandardScaler)
from sklearn.metrics import (
    accuracy_score, roc_auc_score)
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import slowfast.models
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
from thes.caffe import (nicolas_net, model_test_get_image_blob)
from thes.data.tubes.routines import (
        score_ftubes_via_objaction_overlap_aggregation)

log = logging.getLogger(__name__)


# def profile(func):
#     return func


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

    def forward(self, X, bboxes0):
        # Forward through model
        x = self._model(X)
        # / Roi_Pooling
        pool_out = []
        for pi in range(self.num_pathways):
            t_pool = getattr(self, f's{pi}_tpool')
            out = t_pool(x[pi])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            # Roi, assuming 1 box per 1 batch_ind
            roi_align = getattr(self, f's{pi}_roi')
            out = roi_align(out, bboxes0)

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
    scaled = [x.result() for x in futures]
    return scaled


def tfm_video_resize_threaded(X_list, dsize, max_workers=8):
    # 256 resize, normalize, group,
    h_before, w_before = X_list[0].shape[0:2]
    X_list = threaded_ocv_resize_clip(X_list, dsize, max_workers)
    h_resized, w_resized = X_list[0].shape[0:2]
    params = {'h_before': h_before, 'w_before': w_before,
              'h_resized': h_resized, 'w_resized': w_resized}
    return X_list, params


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


def prepare_video(X_list, flip):
    # Resize video, flip to RGB, convert to torch tensor
    if flip:
        if isinstance(X_list, list):
            X_list = [np.flip(x, -1) for x in X_list]
        else:
            X_list = np.flip(X_list, -1)
    # Resize
    X_list, resize_params = tfm_video_resize_threaded(
            X_list, test_crop_size)
    X = np.stack(X_list)
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

def sample_via_pyav(video_path, finds_to_sample, threading=True):
    container = av.open(str(video_path))

    frames_length = container.streams.video[0].frames
    duration = container.streams.video[0].duration  # how many time_base units

    timebase = int(duration/frames_length)
    pts_to_sample = timebase * finds_to_sample
    start_pts = int(pts_to_sample[0])
    end_pts = int(pts_to_sample[-1])

    margin = 1024
    seek_offset = max(start_pts - margin, 0)
    stream_name = {"video": 0}
    buffer_size = 0

    stream = container.streams.video[0]
    container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    if threading:
        container.streams.video[0].thread_type = 'AUTO'
    frames = {}
    buffer_count = 0
    max_pts = 0
    for frame in container.decode(**stream_name):
        max_pts = max(max_pts, frame.pts)
        if frame.pts < start_pts:
            continue
        if frame.pts <= end_pts:
            frames[frame.pts] = frame
        else:
            buffer_count += 1
            frames[frame.pts] = frame
            if buffer_count >= buffer_size:
                break
    pts_we_got = np.array(sorted(frames))
    ssorted_indices = np.searchsorted(pts_we_got, pts_to_sample)

    sampled_frames = []
    for pts in pts_we_got[ssorted_indices]:
        sampled_frames.append(frames[pts])
    container.close()

    sampled_framelist = [frame.to_rgb().to_ndarray()
            for frame in sampled_frames]
    return sampled_framelist


class TDataset_over_keyframes(torch.utils.data.Dataset):
    def __init__(self, keyframes, model_nframes, model_sample,
            is_slowfast: bool, slowfast_alpha: int,
            load_method='opencv'):
        self.keyframes = keyframes
        center_frame = (model_nframes-1)//2
        self.sample_grid0 = (np.arange(model_nframes)-center_frame)*model_sample
        self._is_slowfast = is_slowfast
        self._slowfast_alpha = slowfast_alpha
        self._load_method = load_method

    def __getitem__(self, index):
        # Extract frames
        keyframe = self.keyframes[index]
        video_path = keyframe['video_path']
        i0 = keyframe['frame0']
        finds_to_sample = i0 + self.sample_grid0
        finds_to_sample = np.clip(
                finds_to_sample, 0, keyframe['nframes']-1)

        if self._load_method == 'opencv':
            with vt_cv.video_capture_open(video_path) as vcap:
                framelist_u8_bgr = vt_cv.video_sample(
                        vcap, finds_to_sample)
            Xt, resize_params, ccrop_params = prepare_video(
                    framelist_u8_bgr, flip=True)
        elif self._load_method == 'pyav':
            framelist_u8_rgb = sample_via_pyav(
                    video_path, finds_to_sample, True)
            Xt, resize_params, ccrop_params = prepare_video(
                    framelist_u8_rgb, flip=False)
        else:
            raise RuntimeError()

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
        bbox_tldr = prepare_box(
                keyframe['bbox'], resize_params, ccrop_params)
        return index, frame_list, bbox_tldr

    def __len__(self) -> int:
        return len(self.keyframes)


class TDataset_over_connections(torch.utils.data.Dataset):
    def __init__(self, connections_flist, dataset,
            model_nframes, model_sample,
            is_slowfast: bool, slowfast_alpha: int,
            load_method='opencv'):
        self.connections_flist = connections_flist
        self.dataset = dataset
        center_frame = (model_nframes-1)//2
        self.sample_grid0 = (np.arange(model_nframes)-center_frame)*model_sample
        self._is_slowfast = is_slowfast
        self._slowfast_alpha = slowfast_alpha
        self._load_method = load_method

    def __getitem__(self, index):
        # Extract frames
        connections = self.connections_flist[index]
        vid = connections['vid']
        i0 = connections['frame_ind']

        video_path = str(self.dataset.videos_ocv[vid]['path'])
        nframes = self.dataset.videos_ocv[vid]['nframes']

        finds_to_sample = i0 + self.sample_grid0
        finds_to_sample = np.clip(finds_to_sample, 0, nframes-1)

        if self._load_method == 'opencv':
            with vt_cv.video_capture_open(video_path) as vcap:
                framelist_u8_bgr = vt_cv.video_sample(
                        vcap, finds_to_sample)
            Xt, resize_params, ccrop_params = prepare_video(
                    framelist_u8_bgr, flip=True)
        elif self._load_method == 'pyav':
            framelist_u8_rgb = sample_via_pyav(
                    video_path, finds_to_sample, True)
            Xt, resize_params, ccrop_params = prepare_video(
                    framelist_u8_rgb, flip=False)
        else:
            raise RuntimeError()

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
        bboxes_tldr = []
        for box_ltrd in connections['boxes']:
            prepared_bbox_tldr = prepare_box(
                box_ltrd, resize_params, ccrop_params)
            bboxes_tldr.append(prepared_bbox_tldr)
        bboxes_tldr = {
                'boxes': np.stack(bboxes_tldr, axis=0),
                'do_not_collate': True}
        return index, frame_list, bboxes_tldr

    def __len__(self) -> int:
        return len(self.connections_flist)


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
            with small.QTimer('saving pkl'):
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
        mirror: ['uname', ~]
    model: [~, ['slowfast', 'i3d']]
    extraction_mode: ['roi', ['roi', 'fullframe']]
    extraction:
        batch_size: [8, int]
        num_workers: [12, int]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
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

    BATCH_SIZE = cf['extraction.batch_size']
    NUM_WORKERS = cf['extraction.num_workers']

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

    bboxes_batch_index = torch.arange(
        BATCH_SIZE).type(torch.DoubleTensor)[:, None]

    def func(data_input):
        II, Xts, bboxes = data_input
        Xts_f32c = [to_gpu_normalize_permute(
            x, norm_mean_t, norm_std_t) for x in Xts]

        bsize = bboxes.shape[0]
        bboxes0 = torch.cat(
                (bboxes_batch_index[:bsize], bboxes), axis=1)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            result = extractor.forward(Xts_f32c, bboxes0_c)
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
        mirror: ['uname', ~]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
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


class Box_connections_dwti(TypedDict):
    vid: Vid_daly
    frame_ind: int
    dwti_sources: List[I_dwein]  # N
    boxes: List[np.ndarray]  # N, 4


def group_tubes_on_frame_level(
        tubes_dwein: Dict[I_dwein, T_dwein],
        frames_to_cover: Optional[Dict[Vid_daly, np.ndarray]]
        ) -> Dict[Tuple[Vid_daly, int], Box_connections_dwti]:
    """
    Given dictionary of tubes, group them on frame level
    - If "frames_to_cover" is passed - group only in those frames

    This is a simplified _prepare_ftube_box_computations
    """
    connections: Dict[Tuple[Vid_daly, int], Box_connections_dwti] = {}
    for dwt_index, tube in tubes_dwein.items():
        (vid, _, _) = dwt_index
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
            c = connections.setdefault((vid, find),
                    Box_connections_dwti(
                        vid=vid, frame_ind=find,
                        dwti_sources=[], boxes=[]))
            c['dwti_sources'].append(dwt_index)
            c['boxes'].append(box)
    return connections


def get_daly_keyframes_to_cover(
        dataset, vids, add_keyframes: bool, every_n: int,
        ) -> Dict[Vid_daly, np.ndarray]:
    frames_to_cover: Dict[Vid_daly, np.ndarray] = {}
    for vid in vids:
        v = dataset.videos_ocv[vid]
        # general keyframe ranges of all instances
        instance_ranges = []
        for action_name, instances in v['instances'].items():
            for ins_ind, instance in enumerate(instances):
                s, e = instance['start_frame'], instance['end_frame']
                keyframes = [int(kf['frame'])
                        for kf in instance['keyframes']]
                instance_ranges.append((s, e, keyframes))
        good = set()
        for s, e, keyframes in instance_ranges:
            if add_keyframes:
                good |= set(keyframes)
            if every_n > 0:
                good |= set(range(s, e+1, every_n))
        frames_to_cover[vid] = np.array(sorted(good))
    return frames_to_cover


from torch.utils.data.dataloader import default_collate
import collections


def sequence_batch_collate_v2(batch):
    assert isinstance(batch[0], collections.abc.Sequence), \
            'Only sequences supported'
    # From gunnar code
    transposed = zip(*batch)
    collated = []
    for samples in transposed:
        if isinstance(samples[0], collections.abc.Mapping) \
               and 'do_not_collate' in samples[0]:
            c_samples = samples
        elif getattr(samples[0], 'do_not_collate', False) is True:
            c_samples = samples
        else:
            c_samples = default_collate(samples)
        collated.append(c_samples)
    return collated


def _perform_connections_split(connections_f, cc, ct):
    ckeys = list(connections_f.keys())
    weights_dict = {k: len(v['boxes']) for k, v in connections_f.items()}
    weights = np.array(list(weights_dict.values()))
    ii_ckeys_split = snippets.weighted_array_split(
            np.arange(len(ckeys)), weights, ct)
    ckeys_split = [[ckeys[i] for i in ii] for ii in ii_ckeys_split]
    ktw = dict(zip(ckeys, weights))
    weights_split = []
    for ckeys_ in ckeys_split:
        weight = np.sum([ktw[ckey] for ckey in ckeys_])
        weights_split.append(weight)
    chunk_ckeys = ckeys_split[cc]
    log.info(f'Quick split stats [{cc,ct=}]: ''Frames(boxes): {}({}) -> {}({})'.format(
        len(ckeys), np.sum(weights),
        len(chunk_ckeys), weights_split[cc]))
    log.debug(f'Full stats [{cc,ct=}]:\n'
            f'ckeys_split={pprint.pformat(ckeys_split)}\n'
            f'{weights_split=}\n'
            f'{chunk_ckeys=}\n'
            f'{weights_split[cc]=}')
    chunk_connections_f = {k: connections_f[k] for k in chunk_ckeys}
    return chunk_connections_f


def probe_philtubes_for_extraction(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
        mirror: ['uname', ~]
    tubes_dwein: [~, str]
    model: [~, ['slowfast', 'i3d']]
    extraction_mode: ['roi', ['roi', 'fullframe']]
    extraction:
        batch_size: [8, int]
        num_workers: [12, int]
    compute_split:
        enabled: [False, bool]
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # Load tubes
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    # Frames to cover: keyframes and every 16th frame
    vids = list(dataset.videos_ocv.keys())
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
            get_daly_keyframes_to_cover(dataset, vids, True, 16)
    connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    connections_f = group_tubes_on_frame_level(
            tubes_dwein, frames_to_cover)

    # Here we'll run our connection split
    if cf['compute_split.enabled']:
        cc, ct = (cf['compute_split.chunk'], cf['compute_split.total'])
        connections_f = _perform_connections_split(connections_f, cc, ct)

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

    BATCH_SIZE = cf['extraction.batch_size']
    NUM_WORKERS = cf['extraction.num_workers']

    norm_mean_t = np_to_gpu(norm_mean)
    norm_std_t = np_to_gpu(norm_std)

    def prepare_func(start_i):
        connections_flist = list(connections_f.values())
        remaining_connections = connections_flist[start_i+1:]
        tdataset_kf = TDataset_over_connections(
                remaining_connections, dataset,
                model_nframes, model_sample,
                is_slowfast, slowfast_alpha)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=sequence_batch_collate_v2)
        return loader

    def func(data_input):
        II, Xts, bbox_dicts = data_input
        Xts_f32c = [to_gpu_normalize_permute(
            x, norm_mean_t, norm_std_t) for x in Xts]
        # bbox transformations
        bboxes_np = [b['boxes'] for b in bbox_dicts]
        counts = np.array([len(x) for x in bboxes_np])
        batch_indices = np.repeat(np.arange(len(counts)), counts)
        bboxes0 = np.c_[batch_indices, np.vstack(bboxes_np)]
        bboxes0 = torch.from_numpy(bboxes0)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)

        with torch.no_grad():
            result = extractor.forward(Xts_f32c, bboxes0_c)
        result_np = {k: v.cpu().numpy()
                for k, v in result.items()}
        result_np = {k: np.split(
            result_np['roipooled'],
            np.cumsum(counts), axis=0)[:-1]
            for k, v in result_np.items()}
        # Split by batch size
        II_np = II.cpu().numpy()
        return II_np, result_np

    disaver_fold = small.mkdir(out/'disaver')
    total = len(connections_f)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
            save_interval=60)
    outputs = disaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        key_outputs = [oo for o in outputs for oo in o[k]]
        dict_outputs[k] = key_outputs

    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    small.save_pkl(out/'connections_f.pkl', connections_f)
