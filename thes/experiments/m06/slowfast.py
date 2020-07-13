import h5py
import numpy as np
import pprint
import logging
import time
import copy
from pathlib import Path
from types import MethodType
from typing import (  # NOQA
    Dict, Any, List, Optional, Tuple, TypedDict, Set)
from tqdm import tqdm
import concurrent.futures
import cv2
import av

import torch
import torch.nn as nn
import torch.utils.data
from detectron2.layers import ROIAlign
import slowfast.models
import slowfast.utils.checkpoint as cu

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv
from vsydorov_tools import log as vt_log

from thes.data.dataset.daly import (
    get_daly_keyframes_to_cover)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.tubes.types import (
    I_dwein, T_dwein,
    loadconvert_tubes_dwein, Box_connections_dwti)
from thes.data.tubes.routines import (
    group_tubes_on_frame_level)
from thes.slowfast.cfg import (basic_sf_cfg)
from thes.tools import snippets
from thes.tools.video import (
    tfm_video_resize_threaded, tfm_video_center_crop)
from thes.caffe import (nicolas_net, model_test_get_image_blob)
from thes.pytorch import sequence_batch_collate_v2

log = logging.getLogger(__name__)


def np_to_gpu(X):
    X = torch.from_numpy(np.array(X))
    X = X.type(torch.cuda.FloatTensor)
    return X


norm_mean = np.array([0.45, 0.45, 0.45])
norm_std = np.array([0.225, 0.225, 0.225])

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


def prepare_video_resize_crop_flip(
        X_list, crop_size, flip_lastdim):
    # Resize video, flip to RGB
    if flip_lastdim:
        if isinstance(X_list, list):
            X_list = [np.flip(x, -1) for x in X_list]
        else:
            X_list = np.flip(X_list, -1)
    # Resize
    X_list, resize_params = tfm_video_resize_threaded(
            X_list, crop_size)
    X = np.stack(X_list)
    # Centercrop
    X, ccrop_params = tfm_video_center_crop(
            X, crop_size, crop_size)
    return X, resize_params, ccrop_params


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
        test_crop_size = 256

        if self._load_method == 'opencv':
            with vt_cv.video_capture_open(video_path) as vcap:
                framelist_u8_bgr = vt_cv.video_sample(
                        vcap, finds_to_sample)
            X, resize_params, ccrop_params = \
                prepare_video_resize_crop_flip(
                    framelist_u8_bgr, test_crop_size, True)
        elif self._load_method == 'pyav':
            framelist_u8_rgb = sample_via_pyav(
                    video_path, finds_to_sample, True)
            X, resize_params, ccrop_params = \
                prepare_video_resize_crop_flip(
                    framelist_u8_rgb, test_crop_size, False)
        else:
            raise RuntimeError()

        # Convert to torch
        Xt = torch.from_numpy(X)

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
    def __init__(self, dict_f, dataset,
            model_nframes, model_sample,
            is_slowfast: bool, slowfast_alpha: int,
            load_method='opencv'):
        self.dict_f = dict_f
        self.dataset = dataset
        center_frame = (model_nframes-1)//2
        self.sample_grid0 = (
                np.arange(model_nframes)-center_frame)*model_sample
        self._is_slowfast = is_slowfast
        self._slowfast_alpha = slowfast_alpha
        self._load_method = load_method

    def __getitem__(self, index):
        # Extract frames
        ckey, connections = list(self.dict_f.items())[index]
        vid = connections['vid']
        i0 = connections['frame_ind']
        bboxes_ltrd = connections['boxes']
        assert ckey == (vid, i0)

        video_path = str(self.dataset.videos_ocv[vid]['path'])
        nframes = self.dataset.videos_ocv[vid]['nframes']

        finds_to_sample = i0 + self.sample_grid0
        finds_to_sample = np.clip(finds_to_sample, 0, nframes-1)

        test_crop_size = 256

        if self._load_method == 'opencv':
            with vt_cv.video_capture_open(video_path) as vcap:
                framelist_u8_bgr = vt_cv.video_sample(
                        vcap, finds_to_sample)
            X, resize_params, ccrop_params = \
                prepare_video_resize_crop_flip(
                    framelist_u8_bgr, test_crop_size, True)
        elif self._load_method == 'pyav':
            framelist_u8_rgb = sample_via_pyav(
                    video_path, finds_to_sample, True)
            X, resize_params, ccrop_params = \
                prepare_video_resize_crop_flip(
                    framelist_u8_rgb, test_crop_size, False)
        else:
            raise RuntimeError()

        # Convert to torch
        Xt = torch.from_numpy(X)

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
        prepared_bboxes = []
        for box_ltrd in bboxes_ltrd:
            prepared_bbox_tldr = prepare_box(
                box_ltrd, resize_params, ccrop_params)
            prepared_bboxes.append(prepared_bbox_tldr)
        meta = {
                'index': index,
                'ckey': ckey,
                'bboxes_tldr': np.stack(prepared_bboxes, axis=0),
                'do_not_collate': True}
        return frame_list, meta

    def __len__(self) -> int:
        return len(self.dict_f)


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

        result_cache = []

        def flush_purge():
            self.result.extend(result_cache)
            result_cache.clear()
            with small.QTimer('saving pkl'):
                self._save(i_last)
            self._purge_intermediate_files()

        loader = self.prepare_func(i_last)
        pbar = tqdm(loader, total=len(loader))
        for i_batch, data_input in enumerate(pbar):
            result_dict, i_last = self.func(data_input)
            result_cache.append(result_dict)
            SAVE = False
            if self._save_every > 0:
                SAVE |= (i_last - self._i_last_saved) >= self._save_every
            if self._save_interval:
                since_last_save = time.perf_counter() - self._time_last_save
                SAVE |= since_last_save > self._save_interval
            if SAVE:
                flush_purge()
                self._time_last_save = time.perf_counter()
                self._i_last_saved = i_last
            if self._log_interval:
                since_last_log = time.perf_counter() - self._time_last_log
                if since_last_log > self._log_interval:
                    log.info(snippets._tqdm_str(pbar))
                    self._time_last_log = time.perf_counter()
        flush_purge()
        return self.result


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


def _perform_connections_split(connections_f, cc, ct):
    ckeys = list(connections_f.keys())
    weights_dict = {k: len(v['boxes'])
            for k, v in connections_f.items()}
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
    log.info(f'Quick split stats [{cc,ct=}]: '
        'Frames(boxes): {}({}) -> {}({})'.format(
            len(ckeys), np.sum(weights),
            len(chunk_ckeys), weights_split[cc]))
    chunk_connections_f = {k: connections_f[k] for k in chunk_ckeys}
    return chunk_connections_f


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


# Experiments

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
    test_crop_size = 256

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
        save_interval: [300, int]
    frame_coverage:
        keyframes: [True, bool]
        subsample: [16, int]
    compute_split:
        enabled: [False, bool]
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
    """)
    cf = cfg.parse()
    test_crop_size = 256

    # DALY Dataset
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    # Load tubes
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    # Frames to cover: keyframes and every 16th frame
    vids = list(dataset.videos_ocv.keys())
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
            get_daly_keyframes_to_cover(dataset, vids,
                    cf['frame_coverage.keyframes'],
                    cf['frame_coverage.subsample'])
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
    # NUM_WORKERS = 0

    norm_mean_t = np_to_gpu(norm_mean)
    norm_std_t = np_to_gpu(norm_std)

    def prepare_func(start_i):
        # start_i defined wrt keys in connections_f
        remaining_dict = dict(list(
            connections_f.items())[start_i+1:])
        tdataset_kf = TDataset_over_connections(
            remaining_dict, dataset, model_nframes,
            model_sample, is_slowfast, slowfast_alpha)
        loader = torch.utils.data.DataLoader(tdataset_kf,
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            collate_fn=sequence_batch_collate_v2)
        return loader

    def func(data_input):
        Xts, metas = data_input
        Xts_f32c = [to_gpu_normalize_permute(
            x, norm_mean_t, norm_std_t) for x in Xts]
        # bbox transformations
        bboxes_np = [m['bboxes_tldr'] for m in metas]
        counts = np.array([len(x) for x in bboxes_np])
        batch_indices = np.repeat(np.arange(len(counts)), counts)
        bboxes0 = np.c_[batch_indices, np.vstack(bboxes_np)]
        bboxes0 = torch.from_numpy(bboxes0)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)

        with torch.no_grad():
            result = extractor.forward(Xts_f32c, bboxes0_c)
        result_dict = {}
        for k, v in result.items():
            out_np = v.cpu().numpy()
            out_split = np.split(out_np,
                np.cumsum(counts), axis=0)[:-1]
            result_dict[k] = out_split

        # Find last index over global structure
        # back to tuple, since dataloader casts to list
        ckey = tuple(metas[-1]['ckey'])
        ckeys = list(connections_f.keys())
        last_i = ckeys.index(ckey)
        return result_dict, last_i

    disaver_fold = small.mkdir(out/'disaver')
    total = len(connections_f)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
        save_interval=cf['extraction.save_interval'], log_interval=300)
    outputs = disaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        key_outputs = [oo for o in outputs for oo in o[k]]
        dict_outputs[k] = key_outputs

    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    small.save_pkl(out/'connections_f.pkl', connections_f)


def combine_probed_philtubes(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['inputs.cfolders'])
    cfg.set_deftype("""
    inputs:
        cfolders: [~, ~]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
        mirror: ['uname', ~]
    tubes_dwein: [~, str]
    frame_coverage:
        keyframes: [True, bool]
        subsample: [16, int]
    output_type: ['h5', ['h5', 'np', 'h5_chunked']]
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
            get_daly_keyframes_to_cover(dataset, vids,
                    cf['frame_coverage.keyframes'],
                    cf['frame_coverage.subsample'])
    connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    connections_f = group_tubes_on_frame_level(
            tubes_dwein, frames_to_cover)

    # Load inputs now
    input_cfolders = cf['inputs.cfolders']
    if not _gather_check_all_present(input_cfolders, [
            'dict_outputs.pkl', 'connections_f.pkl']):
        return

    # Loading all piecemeal connections
    i_cons = {}
    for i, path in enumerate(input_cfolders):
        path = Path(path)
        local_connections_f = small.load_pkl(path/'connections_f.pkl')
        i_cons[i] = local_connections_f
    # Check consistency
    grouped_cons = {}
    for c in i_cons.values():
        grouped_cons.update(c)
    if grouped_cons.keys() != connections_f.keys():
        log.error('Loaded connections inconsistent with expected ones')

    partbox_numbering = []
    for lc in i_cons.values():
        nboxes = np.sum([len(c['boxes']) for c in lc.values()])
        partbox_numbering.append(nboxes)
    partbox_numbering = np.r_[0, np.cumsum(partbox_numbering)]

    # Create mapping of indices
    box_inds = [0]
    for c in connections_f.values():
        box_inds.append(len(c['boxes']))
    box_inds = np.cumsum(box_inds)
    box_inds2 = np.c_[box_inds[:-1], box_inds[1:]]

    small.save_pkl(out/'connections_f.pkl', connections_f)
    small.save_pkl(out/'box_inds2.pkl', box_inds2)

    # Drop the pretense, we gonna use roipool feats here
    output_type = cf['output_type']
    if output_type == 'h5':
        hf = h5py.File(out/"feats.h5", "a", libver="latest")
        dset = hf.create_dataset("roipooled_feats",
                (partbox_numbering[-1], 2304), dtype=np.float16)
    elif output_type == 'h5_chunked':
        hf = h5py.File(out/"feats.h5", "a", libver="latest")
        dset = hf.create_dataset("roipooled_feats",
                (partbox_numbering[-1], 2304),
                chunks=True, dtype=np.float16)
    elif output_type == 'np':
        np_filename = str(out/'feats.npy')
        dset = np.lib.format.open_memmap(np_filename, 'w+',
                dtype=np.float16, shape=(partbox_numbering[-1], 2304))
    else:
        raise RuntimeError()

    # Piecemeal conversion
    for i, path in enumerate(input_cfolders):
        success_file = out/f'{i}.merged'
        if success_file.exists():
            continue
        log.info(f'Merging chunk {i=} at {path=}')
        path = Path(path)
        with small.QTimer('Unpickling'):
            local_dict_outputs = small.load_pkl(path/'dict_outputs.pkl')
        roipooled_feats = local_dict_outputs['roipooled']
        with small.QTimer('Vstack'):
            cat_roipooled_feats = np.vstack(roipooled_feats)
        with small.QTimer('to float16'):
            feats16 = cat_roipooled_feats.astype(np.float16)
        b, e = partbox_numbering[i], partbox_numbering[i+1]
        assert e-b == feats16.shape[0]
        with small.QTimer('Saving to disk chunk {i=}'):
            dset[b:e] = feats16
        success_file.touch()

    if output_type in ['h5', 'h5_chunked']:
        hf.close()
    elif output_type == 'np':
        del dset
    else:
        raise RuntimeError()
