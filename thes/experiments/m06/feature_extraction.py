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

from slowfast.models.video_model_builder import _POOL1 as SF_POOL1

from thes.data.dataset.daly import (
    Ncfg_daly, get_daly_keyframes_to_cover,
    create_keyframelist, to_keyframedict)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.tubes.types import (
    I_dwein, T_dwein,
    loadconvert_tubes_dwein, Box_connections_dwti)
from thes.data.tubes.routines import (
    group_tubes_on_frame_level, perform_connections_split)
from thes.slowfast.cfg import (basic_sf_cfg)
from thes.tools import snippets
from thes.tools.video import (
    tfm_video_resize_threaded, tfm_video_center_crop)
from thes.caffe import (nicolas_net, model_test_get_image_blob)
from thes.pytorch import (
    sequence_batch_collate_v2,
    hforward_resnet, hforward_resnet_nopool, hforward_slowfast,
    np_to_gpu, to_gpu_normalize_permute,
    TDataset_over_keyframes, TDataset_over_connections,
    Sampler_grid, Frameloader_video_slowfast, Dataloader_isaver)

log = logging.getLogger(__name__)

CHECKPOINTS_PREFIX = Path('/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_zoo_checkpoints/')
CHECKPOINTS = {
    'SLOWFAST_4x16_R50': 'kinetics400/SLOWFAST_4x16_R50.pkl',
    'I3D_8x8_R50': 'kinetics400/I3D_8x8_R50.pkl',
    'c2d': 'kin400_video_nonlocal/c2d_baseline_8x8_IN_pretrain_400k.pkl',  # https://github.com/facebookresearch/SlowFast/issues/163
    'c2d_1x1': 'kin400_video_nonlocal/c2d_baseline_8x8_IN_pretrain_400k.pkl',
    'c2d_imnet': 'imagenet/R50_IN1K.pyth',
}
REL_YAML_PATHS = {
    'SLOWFAST_4x16_R50': 'Kinetics/c2/SLOWFAST_4x16_R50.yaml',
    'I3D_8x8_R50': 'Kinetics/c2/I3D_8x8_R50.yaml',
    'c2d': 'Kinetics/C2D_8x8_R50_IN1K.yaml',
    'c2d_1x1': 'Kinetics/C2D_8x8_R50_IN1K.yaml',
    'c2d_imnet': 'Kinetics/C2D_8x8_R50_IN1K.yaml',
}

class Head_featextract_roi(nn.Module):
    def __init__(self, dim_in, temp_pool_size, resolution, scale_factor):
        super(Head_featextract_roi, self).__init__()
        self.dim_in = dim_in
        self.num_pathways = len(temp_pool_size)

        for pi in range(self.num_pathways):
            pi_temp_pool_size = temp_pool_size[pi]
            if pi_temp_pool_size is not None:
                tpool = nn.AvgPool3d(
                        [pi_temp_pool_size, 1, 1], stride=1)
                self.add_module(f's{pi}_tpool', tpool)
            roi_align = ROIAlign(
                    resolution[pi],
                    spatial_scale=1.0/scale_factor[pi],
                    sampling_ratio=0,
                    aligned=True)
            self.add_module(f's{pi}_roi', roi_align)
            spool = nn.MaxPool2d(resolution[pi], stride=1)
            self.add_module(f's{pi}_spool', spool)

    def forward(self, x, bboxes0):
        # / Roi_Pooling
        pool_out = []
        for pi in range(self.num_pathways):
            t_pool = getattr(self, f's{pi}_tpool', None)
            if t_pool is not None:
                out = t_pool(x[pi])
            else:
                out = x[pi]
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
        assert x.shape[-1] == sum(self.dim_in)
        result = {'roipooled': x}
        return result

class FExtractor(object):
    def __init__(self, sf_cfg, model_id):
        self.headless_define(sf_cfg, model_id)
        self.head_define(self.model, sf_cfg, model_id)

    def headless_define(self, sf_cfg, model_id):
        model = slowfast.models.build_model(sf_cfg)
        if isinstance(model, slowfast.models.video_model_builder.ResNet):
            if model_id == 'c2d_1x1':
                hforward = hforward_resnet_nopool
            else:
                hforward = hforward_resnet
        elif isinstance(model, slowfast.models.video_model_builder.SlowFast):
            hforward = hforward_slowfast
        else:
            raise RuntimeError()
        self.model = model
        self.model.forward = MethodType(hforward, self.model)

    def head_define(self, model, sf_cfg, model_id):
        model_nframes = sf_cfg.DATA.NUM_FRAMES
        POOL1 = SF_POOL1[sf_cfg.MODEL.ARCH]
        width_per_group = sf_cfg.RESNET.WIDTH_PER_GROUP
        xform_resolution = 7

        if isinstance(model, slowfast.models.video_model_builder.ResNet):
            dim_in = [width_per_group * 32]
            pool_size = [
                    [model_nframes//POOL1[0][0], 1, 1]]
            resolution = [[xform_resolution] * 2]
            scale_factor = [32]
        elif isinstance(model, slowfast.models.video_model_builder.SlowFast):
            # As per SlowFast._construct_network
            dim_in = [
                width_per_group * 32,
                width_per_group * 32 // sf_cfg.SLOWFAST.BETA_INV]
            pool_size = [
                [model_nframes//sf_cfg.SLOWFAST.ALPHA//POOL1[0][0], 1, 1],
                [model_nframes//POOL1[1][0], 1, 1]]
            resolution = [[xform_resolution] * 2] * 2
            scale_factor = [32] * 2
        else:
            raise RuntimeError()

        if model_id == 'c2d_1x1':
            temp_pool_size = [None]
        else:
            temp_pool_size = [s[0] for s in pool_size]
        self.head = Head_featextract_roi(
                dim_in, temp_pool_size, resolution, scale_factor)

    def forward(self, Xt_f32c, bboxes0_c):
        x = self.model(Xt_f32c)
        result = self.head(x, bboxes0_c)
        return result


class Ncfg_extractor:
    def set_defcfg(cfg):
        cfg.set_deftype("""
        extractor:
            model_id: [~, ['SLOWFAST_4x16_R50', 'I3D_8x8_R50', 'c2d', 'c2d_1x1', 'c2d_imnet']]
            extraction_mode: ['roi', ['roi', 'fullframe']]
        extraction:
            batch_size: [8, int]
            num_workers: [12, int]
            save_interval: [300, int]
        """)

    def prepare(cf):
        model_id = cf['extractor.model_id']
        rel_yml_path = REL_YAML_PATHS[model_id]
        sf_cfg = basic_sf_cfg(rel_yml_path)
        if model_id == 'c2d_1x1':
            sf_cfg.DATA.NUM_FRAMES = 1
            sf_cfg.DATA.SAMPLING_RATE = 1
        sf_cfg.NUM_GPUS = 1
        norm_mean = sf_cfg.DATA.MEAN
        norm_std = sf_cfg.DATA.STD

        fextractor = FExtractor(sf_cfg, model_id)

        # Load model
        CHECKPOINT_FILE_PATH = CHECKPOINTS_PREFIX/CHECKPOINTS[model_id]
        if model_id in ['c2d_imnet']:
            cu.load_checkpoint(
                CHECKPOINT_FILE_PATH, fextractor.model, False, None,
                inflation=True, convert_from_caffe2=False,)
        else:
            with vt_log.logging_disabled(logging.WARNING):
                cu.load_checkpoint(
                    CHECKPOINT_FILE_PATH, fextractor.model, False, None,
                    inflation=False, convert_from_caffe2=True,)

        norm_mean_t = np_to_gpu(norm_mean)
        norm_std_t = np_to_gpu(norm_std)

        model_nframes = sf_cfg.DATA.NUM_FRAMES
        model_sample = sf_cfg.DATA.SAMPLING_RATE
        slowfast_alpha = sf_cfg.SLOWFAST.ALPHA

        is_slowfast = isinstance(fextractor.model,
                slowfast.models.video_model_builder.SlowFast)
        sampler_grid = Sampler_grid(model_nframes, model_sample)
        frameloader_vsf = Frameloader_video_slowfast(
                is_slowfast, slowfast_alpha, 256)
        return norm_mean_t, norm_std_t, sampler_grid, frameloader_vsf, fextractor


class Isaver_extract_rgb(snippets.Isaver_base):
    def __init__(
            self, folder,
            total, func, prepare_func,
            interval_iters=None,
            interval_seconds=120,  # every 2 minutes by default
                ):
        super(Isaver_extract_rgb, self).__init__(folder, total)
        self.func = func
        self.prepare_func = prepare_func
        self._interval_iters = interval_iters
        self._interval_seconds = interval_seconds
        self.result = []
        self.npy_array = None

    def _get_filenames(self, i) -> Dict[str, Path]:
        base_filenames = {
            'finished': self._fmt_finished.format(i, self._total)}
        base_filenames['pkl'] = Path(
                base_filenames['finished']).with_suffix('.pkl')
        base_filenames['npy'] = Path(
                base_filenames['finished']).with_suffix('.npy')
        filenames = {k: self._folder/v
                for k, v in base_filenames.items()}
        return filenames

    def _restore(self):
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        start_i, ifiles = max(intermediate_files.items(),
                default=(-1, None))
        if ifiles is not None:
            self.result = small.load_pkl(ifiles['pkl'])
            self.npy_array = np.load(ifiles['npy'])
            log.info('Restore from pkl: {} npy: {}'.format(
                ifiles['pkl'], ifiles['npy']))
        return start_i

    def _save(self, i):
        ifiles = self._get_filenames(i)
        small.save_pkl(ifiles['pkl'], self.result)
        np.save(ifiles['npy'], self.npy_array)
        ifiles['finished'].touch()

    def run(self):
        i_last = self._restore()
        countra = snippets.Counter_repeated_action(
                seconds=self._interval_seconds,
                iters=self._interval_iters)

        pkl_cache = []
        npy_cache = []

        def flush_purge():
            self.result.extend(pkl_cache)
            if self.npy_array is None:
                to_stack = npy_cache
            else:
                to_stack = (self.npy_array, *npy_cache)
            self.npy_array = np.vstack(to_stack)
            pkl_cache.clear()
            npy_cache.clear()
            with small.QTimer('saving'):
                self._save(i_last)
            self._purge_intermediate_files()

        loader = self.prepare_func(i_last)
        pbar = tqdm(loader, total=len(loader))
        for i_batch, data_input in enumerate(pbar):
            pkl_part, npy_part, i_last = self.func(data_input)
            pkl_cache.append(pkl_part)
            npy_cache.append(npy_part)
            if countra.check(i_batch):
                flush_purge()
                log.debug(snippets._tqdm_str(pbar))
                countra.tic(i_batch)
        flush_purge()
        return self.result, self.npy_array

# Experiments


def extract_keyframe_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_extractor.set_defcfg(cfg)
    cf = cfg.parse()

    # prepare extractor
    norm_mean_t, norm_std_t, sampler_grid, frameloader_vsf, fextractor = \
            Ncfg_extractor.prepare(cf)
    BATCH_SIZE = cf['extraction.batch_size']
    NUM_WORKERS = cf['extraction.num_workers']
    # prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    keyframes = create_keyframelist(dataset)
    keyframes_dict = to_keyframedict(keyframes)

    # / extract
    def prepare_func(start_i):
        remaining_keyframes_dict = dict(list(
            keyframes_dict.items())[start_i+1:])
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes_dict, sampler_grid, frameloader_vsf)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=sequence_batch_collate_v2)
        return loader

    bboxes_batch_index = torch.arange(
        BATCH_SIZE).type(torch.DoubleTensor)[:, None]

    def func(data_input):
        metas, Xts, bboxes = data_input
        kkeys = [tuple(m['kkey']) for m in metas]
        Xts_f32c = [to_gpu_normalize_permute(
            x, norm_mean_t, norm_std_t) for x in Xts]

        bsize = bboxes.shape[0]
        bboxes0 = torch.cat(
                (bboxes_batch_index[:bsize], bboxes), axis=1)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            result = fextractor.forward(Xts_f32c, bboxes0_c)
        result_dict = {k: v.cpu().numpy()
                for k, v in result.items()}
        last_i = list(keyframes_dict.keys()).index(kkeys[-1])
        return result_dict, last_i

    disaver_fold = small.mkdir(out/'disaver')
    total = len(keyframes)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
            save_interval=cf['extraction.save_interval'])
    outputs = disaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        stacked = np.vstack([o[k] for o in outputs])
        dict_outputs[k] = stacked
    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    small.save_pkl(out/'keyframes.pkl', keyframes)


def extract_philtube_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_extractor.set_defcfg(cfg)
    cfg.set_deftype("""
    tubes_dwein: [~, str]
    frame_coverage:
        keyframes: [True, bool]
        subsample: [16, int]
    compute_split:
        enabled: [False, bool]
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
    """)
    cf = cfg.parse()

    # prepare extractor
    norm_mean_t, norm_std_t, sampler_grid, frameloader_vsf, fextractor = \
            Ncfg_extractor.prepare(cf)
    BATCH_SIZE = cf['extraction.batch_size']
    NUM_WORKERS = cf['extraction.num_workers']
    # / prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    # // Frames to cover: keyframes and every 16th frame
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
        connections_f = perform_connections_split(connections_f, cc, ct)

    # / extract
    def prepare_func(start_i):
        # start_i defined wrt keys in connections_f
        remaining_dict = dict(list(
            connections_f.items())[start_i+1:])
        tdataset_kf = TDataset_over_connections(
            remaining_dict, dataset, sampler_grid, frameloader_vsf)
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
            result = fextractor.forward(Xts_f32c, bboxes0_c)
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


def combine_split_philtube_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['inputs.cfolders'])
    Ncfg_daly.set_defcfg(cfg)
    cfg.set_deftype("""
    inputs:
        cfolders: [~, ~]
        dims: [~, int]
        key: ['roipooled', str]
    tubes_dwein: [~, str]
    frame_coverage:
        keyframes: [True, bool]
        subsample: [16, int]
    """)
    cf = cfg.parse()

    # / prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    # // Frames to cover: keyframes and every 16th frame
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
    if not snippets.gather_check_all_present(input_cfolders, [
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

    np_filename = str(out/'feats.npy')
    dset = np.lib.format.open_memmap(np_filename, 'w+',
        dtype=np.float16, shape=(partbox_numbering[-1], cf['inputs.dims']))

    # Piecemeal conversion
    for i, path in enumerate(input_cfolders):
        log.info(f'Merging chunk {i=} at {path=}')
        path = Path(path)
        with small.QTimer('Unpickling'):
            local_dict_outputs = small.load_pkl(path/'dict_outputs.pkl')
        roipooled_feats = local_dict_outputs[cf['inputs.key']]
        with small.QTimer('Vstack'):
            cat_roipooled_feats = np.vstack(roipooled_feats)
        with small.QTimer('to float16'):
            feats16 = cat_roipooled_feats.astype(np.float16)
        b, e = partbox_numbering[i], partbox_numbering[i+1]
        assert e-b == feats16.shape[0]
        with small.QTimer(f'Saving to disk chunk {i=}'):
            dset[b:e] = feats16

    del dset


def extract_keyframe_rgb(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    cf = cfg.parse()
    # prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    keyframes = create_keyframelist(dataset)
    keyframes_dict = to_keyframedict(keyframes)
    # prepare others
    NUM_WORKERS = 12
    BATCH_SIZE = 32

    model_nframes = 1
    model_sample = 1
    is_slowfast = False
    slowfast_alpha = 8
    sampler_grid = Sampler_grid(model_nframes, model_sample)
    frameloader_vsf = Frameloader_video_slowfast(
            is_slowfast, slowfast_alpha, 256)

    def prepare_func(start_i):
        remaining_keyframes_dict = dict(list(
            keyframes_dict.items())[start_i+1:])
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes_dict, sampler_grid, frameloader_vsf)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=False,
                collate_fn=sequence_batch_collate_v2)
        return loader

    def func(data_input):
        metas, frame_list, bboxes_tldr = data_input
        kkeys = [tuple(m['kkey']) for m in metas]
        bboxes_np = bboxes_tldr.cpu().numpy()
        Xts_np = frame_list[0].cpu().numpy()
        pkl_part = {
                'bboxes': bboxes_np,
                'kkeys': kkeys}
        npy_part = Xts_np

        last_i = list(keyframes_dict.keys()).index(kkeys[-1])
        return pkl_part, npy_part, last_i

    disaver_fold = small.mkdir(out/'disaver')
    total = len(keyframes)
    disaver = Isaver_extract_rgb(disaver_fold, total, func, prepare_func,
        interval_seconds=90)
    outputs, npy_array = disaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        key_outputs = [oo for o in outputs for oo in o[k]]
        dict_outputs[k] = key_outputs

    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    np.save(out/'rgb.npy', npy_array)
