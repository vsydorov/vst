import logging
from types import MethodType

import torch
import torch.nn as nn

from detectron2.layers import ROIAlign

import slowfast.models
from slowfast.models.video_model_builder import _POOL1 as SF_POOL1
import slowfast.utils.checkpoint as sf_cu

import vst

from thes.pytorch import (
        hforward_resnet, hforward_resnet_nopool, hforward_slowfast,
        CHECKPOINTS_PREFIX, CHECKPOINTS, REL_YAML_PATHS,
        Sampler_grid, Frameloader_video_slowfast, np_to_gpu)
from thes.slowfast.cfg import (
        basic_sf_cfg)

log = logging.getLogger(__name__)

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


class Head_featextract_fullframe(nn.Module):
    def __init__(self, dim_in, temp_pool_size, spat_pool_size, ffmode):
        super(Head_featextract_fullframe, self).__init__()
        self.dim_in = dim_in
        self.num_pathways = len(temp_pool_size)
        self.ffmode = ffmode

        for pi in range(self.num_pathways):
            # Pooling goes: T) avg, S) max
            pi_temp_pool_size = temp_pool_size[pi]
            pi_spat_pool_size = spat_pool_size[pi]
            if self.ffmode == 'tavg_smax':
                if pi_temp_pool_size is not None:
                    tpool = nn.AvgPool3d(
                            [pi_temp_pool_size, 1, 1], stride=1)
                    self.add_module(f's{pi}_tpool', tpool)
                spool = nn.MaxPool2d(
                        (pi_spat_pool_size, pi_spat_pool_size), stride=1)
                self.add_module(f's{pi}_spool', spool)
            elif self.ffmode == 'ts_avg':
                pool_size = [
                    pi_temp_pool_size, pi_spat_pool_size, pi_spat_pool_size]
                if pi_temp_pool_size is None:
                    pool_size[0] = 1
                ts_pool = nn.AvgPool3d(pool_size)
                self.add_module(f's{pi}_ts_pool', ts_pool)
            else:
                raise RuntimeError()

    def forward(self, x):
        pool_out = []
        for pi in range(self.num_pathways):
            if self.ffmode == 'tavg_smax':
                t_pool = getattr(self, f's{pi}_tpool', None)
                if t_pool is not None:
                    out = t_pool(x[pi])
                else:
                    out = x[pi]
                assert out.shape[2] == 1
                out = torch.squeeze(out, 2)
                s_pool = getattr(self, f's{pi}_spool')
                out = s_pool(out)
            elif self.ffmode == 'ts_avg':
                ts_pool = getattr(self, f's{pi}_ts_pool')
                out = ts_pool(x[pi])
            else:
                raise RuntimeError()
            pool_out.append(out)
        # B C H W.
        x = torch.cat(pool_out, 1)
        x = x.view(x.shape[0], -1)
        assert x.shape[-1] == sum(self.dim_in)
        result = {'fullframe': x}
        return result

class FExtractor(object):
    def _headless_define(self, sf_cfg, model_id):
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
        self.model.eval()  # IMPORTANT

    def _head_define(self, model, sf_cfg, model_id):
        raise NotImplementedError()

    def forward(self, Xt_f32c, bboxes0_c):
        raise NotImplementedError()

class FExtractor_roi(FExtractor):
    def __init__(self, sf_cfg, model_id):
        self._headless_define(sf_cfg, model_id)
        self._head_define(self.model, sf_cfg, model_id)

    def _head_define(self, model, sf_cfg, model_id):
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

class FExtractor_fullframe(FExtractor):
    def __init__(self, sf_cfg, model_id, IMAGE_SIZE, ffmode):
        self._headless_define(sf_cfg, model_id)
        self._head_define(self.model, sf_cfg, model_id, IMAGE_SIZE, ffmode)

    def _head_define(self, model, sf_cfg, model_id, IMAGE_SIZE, ffmode):
        model_nframes = sf_cfg.DATA.NUM_FRAMES
        POOL1 = SF_POOL1[sf_cfg.MODEL.ARCH]
        width_per_group = sf_cfg.RESNET.WIDTH_PER_GROUP

        if isinstance(model, slowfast.models.video_model_builder.ResNet):
            dim_in = [width_per_group * 32]
            pool_size = [
                    [model_nframes//POOL1[0][0], 1, 1]]
            spat_pool_size = [IMAGE_SIZE//32]
        elif isinstance(model, slowfast.models.video_model_builder.SlowFast):
            # As per SlowFast._construct_network
            dim_in = [
                width_per_group * 32,
                width_per_group * 32 // sf_cfg.SLOWFAST.BETA_INV]
            pool_size = [
                [model_nframes//sf_cfg.SLOWFAST.ALPHA//POOL1[0][0], 1, 1],
                [model_nframes//POOL1[1][0], 1, 1]]
            spat_pool_size = [IMAGE_SIZE//32, IMAGE_SIZE//32]
        else:
            raise RuntimeError()

        if model_id == 'c2d_1x1':
            temp_pool_size = [None]
        else:
            temp_pool_size = [s[0] for s in pool_size]
        self.head = Head_featextract_fullframe(
                dim_in, temp_pool_size, spat_pool_size, ffmode)

    def forward(self, Xt_f32c, bboxes0_c):
        x = self.model(Xt_f32c)
        result = self.head(x)
        return result

class Ncfg_extractor:
    def set_defcfg(cfg):
        cfg.set_deftype("""
        extractor:
            model_id: [~, ['SLOWFAST_8x8_R50', 'SLOWFAST_4x16_R50',
                           'I3D_8x8_R50', 'c2d', 'c2d_1x1', 'c2d_imnet']]
            extraction_mode: ['roi', ['roi', 'fullframe']]
            fullframe_mode: ['tavg_smax', ['tavg_smax', 'ts_avg']]
            image_size: [256, int]
            sampling_rate: [~, ~]
        extraction:
            box_orientation: ['ltrd', ['tldr', 'ltrd']]  # The other way is wrong
            batch_size: [8, int]
            num_workers: [12, int]
            save_interval: [300, int]
        """)

    def set_defcfg_v2(cfg):
        cfg.set_defaults_yaml("""
        extractor:
            model_id: !def [~,
                ['SLOWFAST_8x8_R50', 'SLOWFAST_4x16_R50',
                 'I3D_8x8_R50', 'c2d', 'c2d_1x1', 'c2d_imnet']]
            extraction_mode: !def ['roi', ['roi', 'fullframe']]
            fullframe_mode: !def ['tavg_smax', ['tavg_smax', 'ts_avg']]
            image_size: 256
            sampling_rate: ~
        extraction:
            box_orientation: !def ['ltrd', ['tldr', 'ltrd']]  # The other way is wrong
            batch_size: 8
            num_workers: 12
            save_interval: 300
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

        IMAGE_SIZE = cf['extractor.image_size']
        ffmode = cf['extractor.fullframe_mode']

        if cf['extractor.extraction_mode'] == 'roi':
            fextractor = FExtractor_roi(sf_cfg, model_id)
        elif cf['extractor.extraction_mode'] == 'fullframe':
            fextractor = FExtractor_fullframe(sf_cfg, model_id, IMAGE_SIZE, ffmode)
        else:
            raise RuntimeError()

        # Load model
        CHECKPOINT_FILE_PATH = CHECKPOINTS_PREFIX/CHECKPOINTS[model_id]
        if model_id in ['c2d_imnet']:
            sf_cu.load_checkpoint(
                CHECKPOINT_FILE_PATH, fextractor.model, False, None,
                inflation=True, convert_from_caffe2=False,)
        else:
            with vst.logging_disabled(logging.WARNING):
                sf_cu.load_checkpoint(
                    CHECKPOINT_FILE_PATH, fextractor.model, False, None,
                    inflation=False, convert_from_caffe2=True,)

        norm_mean_t = np_to_gpu(norm_mean)
        norm_std_t = np_to_gpu(norm_std)

        model_nframes = sf_cfg.DATA.NUM_FRAMES
        model_sample = sf_cfg.DATA.SAMPLING_RATE
        if (samprate := cf['extractor.sampling_rate']) is not None:
            model_sample = samprate
        slowfast_alpha = sf_cfg.SLOWFAST.ALPHA

        is_slowfast = isinstance(fextractor.model,
                slowfast.models.video_model_builder.SlowFast)
        sampler_grid = Sampler_grid(model_nframes, model_sample)
        box_orientation = cf['extraction.box_orientation']
        frameloader_vsf = Frameloader_video_slowfast(
                is_slowfast, slowfast_alpha, IMAGE_SIZE, box_orientation)
        return norm_mean_t, norm_std_t, sampler_grid, frameloader_vsf, fextractor
