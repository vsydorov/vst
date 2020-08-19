import logging
import time
import numpy as np
import collections
from typing import (  # NOQA
    Dict, Any, Literal, List, Optional, Tuple, TypedDict, Set)
from tqdm import tqdm
import av

import torch
from torch.utils.data.dataloader import default_collate  # type: ignore
from fvcore.common.config import CfgNode

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.tools import snippets
from thes.tools.video import (
    tfm_video_resize_threaded, tfm_video_center_crop)


log = logging.getLogger(__name__)


class Sampler_grid(object):
    def __init__(self, model_nframes, model_sample):
        center_frame = (model_nframes-1)//2
        self.sample_grid0 = \
                (np.arange(model_nframes)-center_frame)*model_sample

    def apply(self, i0, nframes):
        finds_to_sample = i0 + self.sample_grid0
        finds_to_sample = np.clip(finds_to_sample, 0, nframes-1)
        return finds_to_sample


class Frameloader_video_slowfast(object):
    def __init__(self, is_slowfast, slowfast_alpha,
            test_crop_size,
            box_orientation='ltrd',
            load_method='opencv'):
        self._is_slowfast = is_slowfast
        self._slowfast_alpha = slowfast_alpha
        self._test_crop_size = test_crop_size
        self._box_orientation = box_orientation
        self._load_method = load_method
        if self._load_method == 'tldr':
            log.warning('!!! TLDR IS A VERY BAD LOAD METHOD !!!')

    def prepare_frame_list(self, video_path, finds_to_sample):
        video_read_method = self._load_method
        is_slowfast = self._is_slowfast
        test_crop_size = self._test_crop_size
        slowfast_alpha = self._slowfast_alpha

        X, resize_params, ccrop_params = \
            extract_prepare_around_keyframe(
                video_path, test_crop_size,
                finds_to_sample, video_read_method)
        Xt = torch.from_numpy(X)
        frame_list = pack_pathway_output(
                Xt, is_slowfast, slowfast_alpha)
        return frame_list, resize_params, ccrop_params

    def prepare_box(self, bbox_ltrd, resize_params, ccrop_params):
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
        if self._box_orientation == 'tldr':
            bbox = bbox_tldr
        elif self._box_orientation == 'ltrd':
            bbox = bbox_tldr[[1, 0, 3, 2]]
        else:
            raise NotImplementedError()
        return bbox


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


def pack_pathway_output(Xt, is_slowfast, slowfast_alpha):
    if is_slowfast:
        # slowfast/datasets/utils.py/pack_pathway_output
        TIME_DIM = 0
        fast_pathway = Xt
        slow_pathway = torch.index_select(
            Xt,
            TIME_DIM,
            torch.linspace(
                0, Xt.shape[TIME_DIM] - 1,
                Xt.shape[TIME_DIM] // slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
    else:
        frame_list = [Xt]
    return frame_list

def extract_prepare_around_keyframe(
    video_path,
    test_crop_size,
    finds_to_sample,
    video_read_method: Literal['opencv', 'pyav'],
        ):
    if video_read_method == 'opencv':
        with vt_cv.video_capture_open(video_path) as vcap:
            framelist_u8_bgr = vt_cv.video_sample(
                    vcap, finds_to_sample)
        X, resize_params, ccrop_params = \
            prepare_video_resize_crop_flip(
                framelist_u8_bgr, test_crop_size, True)
    elif video_read_method == 'pyav':
        framelist_u8_rgb = sample_via_pyav(
                video_path, finds_to_sample, True)
        X, resize_params, ccrop_params = \
            prepare_video_resize_crop_flip(
                framelist_u8_rgb, test_crop_size, False)
    else:
        raise RuntimeError()
    return X, resize_params, ccrop_params


class TDataset_over_keyframes(torch.utils.data.Dataset):
    def __init__(self, dict_keyframes, sampler_grid, frameloader_vsf):
        self.dict_keyframes = dict_keyframes
        self.sampler_grid = sampler_grid
        self.frameloader_vsf = frameloader_vsf

    def __getitem__(self, index):
        # Extract frames
        kkey, keyframe = list(self.dict_keyframes.items())[index]
        video_path = keyframe['video_path']
        i0 = keyframe['frame0']

        finds_to_sample = self.sampler_grid.apply(
                i0, keyframe['nframes'])

        frame_list, resize_params, ccrop_params = \
            self.frameloader_vsf.prepare_frame_list(
                    video_path, finds_to_sample)
        bbox_tldr = self.frameloader_vsf.prepare_box(
                keyframe['bbox'], resize_params, ccrop_params)
        meta = {
                'index': index,
                'kkey': kkey,
                'do_not_collate': True}
        return meta, frame_list, bbox_tldr

    def __len__(self) -> int:
        return len(self.dict_keyframes)


class TDataset_over_connections(torch.utils.data.Dataset):
    def __init__(self, dict_f, dataset,
            sampler_grid, frameloader_vsf):
        self.dict_f = dict_f
        self.dataset = dataset
        self.sampler_grid = sampler_grid
        self.frameloader_vsf = frameloader_vsf

    def __getitem__(self, index):
        # Extract frames
        ckey, connections = list(self.dict_f.items())[index]
        vid = connections['vid']
        i0 = connections['frame_ind']
        bboxes_ltrd = connections['boxes']
        assert ckey == (vid, i0)

        video_path = str(self.dataset.videos_ocv[vid]['path'])
        nframes = self.dataset.videos_ocv[vid]['nframes']

        finds_to_sample = self.sampler_grid.apply(i0, nframes)
        frame_list, resize_params, ccrop_params = \
            self.frameloader_vsf.prepare_frame_list(
                    video_path, finds_to_sample)
        prepared_bboxes = []
        for box_ltrd in bboxes_ltrd:
            prepared_bbox_tldr = self.frameloader_vsf.prepare_box(
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

# Headless forward functions

def hforward_resnet_nopool(self, x):
    # slowfast/models/video_model_builder.py/ResNet.forward
    x = self.s1(x)
    x = self.s2(x)
    x = self.s3(x)
    x = self.s4(x)
    x = self.s5(x)
    return x

def hforward_resnet(self, x):
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

def hforward_slowfast(self, x):
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

# utilities


def cf_to_cfgnode(cf):
    cn = CfgNode(snippets.unflatten_nested_dict(cf), [])
    return cn


def merge_cf_into_cfgnode(d_cfg, cf_add_d2):
    d_cfg.merge_from_other_cfg(cf_to_cfgnode(cf_add_d2))
    return d_cfg


def np_to_gpu(X):
    X = torch.from_numpy(np.array(X))
    X = X.type(torch.cuda.FloatTensor)
    return X


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


class Dataloader_isaver(
        snippets.isaver.Isaver_mixin_restore_save, snippets.isaver.Isaver_base):
    """
    Will process a list with a 'func', 'prepare_func(start_i)' is to be run before processing
    """
    def __init__(self, folder,
            total, func, prepare_func,
            save_interval_iters=None,
            save_interval_seconds=120,  # every 2 minutes by default
            log_interval=None,):
        super().__init__(folder, total)
        self.func = func
        self.prepare_func = prepare_func
        self._save_interval_iters = save_interval_iters
        self._save_interval_seconds = save_interval_seconds
        self._log_interval = log_interval
        self.result = []

    def run(self):
        i_last = self._restore()
        countra = snippets.Counter_repeated_action(
                seconds=self._save_interval_seconds,
                iters=self._save_interval_iters)

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
            if countra.check(i_batch):
                flush_purge()
                log.debug(snippets.tqdm_str(pbar))
                countra.tic(i_batch)
        flush_purge()
        return self.result

class NumpyRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, rgen):
        self.data_source = data_source
        self.rgen = rgen

    def __iter__(self):
        n = len(self.data_source)
        return iter(self.rgen.permutation(n).tolist())

    def __len__(self):
        return len(self.data_source)
