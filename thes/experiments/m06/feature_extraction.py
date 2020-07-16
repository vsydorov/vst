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
    Ncfg_daly, get_daly_keyframes_to_cover, create_keyframelist)
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
    hforward_resnet, hforward_slowfast,
    np_to_gpu, to_gpu_normalize_permute,
    TDataset_over_keyframes, TDataset_over_connections,
    Sampler_grid, Frameloader_video_slowfast, Dataloader_isaver)

log = logging.getLogger(__name__)

CHECKPOINTS_PREFIX = Path('/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_zoo_checkpoints/')
CHECKPOINTS = {
    'SLOWFAST_4x16_R50': 'kinetics400/SLOWFAST_4x16_R50.pkl',
    'I3D_8x8_R50': 'kinetics400/I3D_8x8_R50.pkl',
    'C2D_NOPOOL_8x8_R50': 'kinetics400/C2D_NOPOOL_8x8_R50.pkl',  # does not work
    'c2d': 'kin400_video_nonlocal/c2d_baseline_8x8_IN_pretrain_400k.pkl',  # https://github.com/facebookresearch/SlowFast/issues/163
    'R50_IMNET': 'imagenet/R50_IN1K.pyth'
}
REL_YAML_PATHS = {
    'SLOWFAST_4x16_R50': 'Kinetics/c2/SLOWFAST_4x16_R50.yaml',
    'I3D_8x8_R50': 'Kinetics/c2/I3D_8x8_R50.yaml',
    'C2D_NOPOOL_8x8_R50': 'Kinetics/c2/C2D_NOPOOL_8x8_R50.yaml',  # does not work
    'c2d': 'Kinetics/C2D_8x8_R50_IN1K.yaml',
    'R50_IMNET': None
}

class Head_featextract_roi(nn.Module):
    def __init__(self, dim_in, pool_size, resolution, scale_factor):
        super(Head_featextract_roi, self).__init__()
        self.dim_in = dim_in
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

    def forward(self, x, bboxes0):
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
        assert x.shape[-1] == sum(self.dim_in)
        result = {'roipooled': x}
        return result

class FExtractor(object):
    def __init__(self, sf_cfg):
        self.headless_define(sf_cfg)
        self.head_define(self.model, sf_cfg)

    def headless_define(self, sf_cfg):
        model = slowfast.models.build_model(sf_cfg)
        if isinstance(model, slowfast.models.video_model_builder.ResNet):
            hforward = hforward_resnet
        elif isinstance(model, slowfast.models.video_model_builder.SlowFast):
            hforward = hforward_slowfast
        else:
            raise RuntimeError()
        self.model = model
        self.model.forward = MethodType(hforward, self.model)

    def head_define(self, model, sf_cfg):
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
        self.head = Head_featextract_roi(
                dim_in, pool_size, resolution, scale_factor)

    def forward(self, Xt_f32c, bboxes0_c):
        x = self.model(Xt_f32c)
        result = self.head(x, bboxes0_c)
        return result


class Ncfg_extractor:
    def set_defcfg(cfg):
        cfg.set_deftype("""
        extractor:
            model_id: [~, ['SLOWFAST_4x16_R50', 'I3D_8x8_R50', 'C2D_NOPOOL_8x8_R50', 'c2d', 'R50_IMNET']]
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
        sf_cfg.NUM_GPUS = 1
        norm_mean = sf_cfg.DATA.MEAN
        norm_std = sf_cfg.DATA.STD

        fextractor = FExtractor(sf_cfg)

        # Load model
        CHECKPOINT_FILE_PATH = CHECKPOINTS_PREFIX/CHECKPOINTS[model_id]
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

    # / extract
    def prepare_func(start_i):
        remaining_keyframes = keyframes[start_i+1:]
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes, sampler_grid, frameloader_vsf)
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
            result = fextractor.forward(Xts_f32c, bboxes0_c)
        result_np = {k: v.cpu().numpy()
                for k, v in result.items()}
        II_np = II.cpu().numpy()
        return II_np, result_np

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
