from tqdm import tqdm
import copy
import shutil
import cv2
from glob import glob
import os.path
import pprint
import numpy as np
import logging
from pathlib import Path
from typing import (  # NOQA
    Dict, Any, List, Optional, Tuple, TypedDict, Set, cast, Literal)
from sklearn.metrics import (
    accuracy_score)

import torch
import torch.nn as nn
import torch.utils.data

from detectron2.layers import ROIAlign

import slowfast.utils.checkpoint as sf_cu
import slowfast.utils.weight_init_helper as init_helper
import slowfast.utils.misc as sf_misc
from slowfast.datasets import cv2_transform as sf_cv2_transform
from slowfast.models.video_model_builder import SlowFast as M_slowfast
from slowfast.models.video_model_builder import ResNet as M_resnet
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.video_model_builder import _POOL1 as SF_POOL1

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv
from vsydorov_tools import log as vt_log

from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.dataset.daly import (
    Ncfg_daly, sample_daly_frames_from_instances,
    load_gt_and_wein_tubes, create_keyframelist,
    to_keyframedict, group_dwein_frames_wrt_kf_distance,
    Frame_labeled, Box_labeled,
    prepare_label_fullframes_for_training,
    prepare_label_roiboxes_for_training
)

from thes.data.tubes.types import (  # NOQA
    Box_connections_dwti, I_dwein, T_dwein,
    T_dwein_scored, I_dgt, T_dgt, AV_dict,
    Tube_daly_wein_as_provided,
    av_stubes_above_score, push_into_avdict
)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.data.tubes.routines import (
    group_tubes_on_frame_level,
    qload_synthetic_tube_labels,
    get_dwein_overlaps_per_dgt,
    select_fg_bg_tubes,
    perform_connections_split,
    compute_flattube_syntlabel_acc,
    quick_assign_scores_to_dwein_tubes
    )

from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)
from thes.evaluation.meta import (
    cheating_tube_scoring, quick_tube_eval,
    assign_scorefield, assign_scores_to_dwt_roipooled,
    assign_scores_to_dwt_fullframe)

from thes.slowfast.cfg import (basic_sf_cfg)
from thes.tools import snippets
from thes.tools.snippets import check_step_sslice as check_step
from thes.tools.video import (
    tfm_video_resize_threaded, tfm_video_center_crop)
from thes.pytorch import (
    sequence_batch_collate_v2, np_to_gpu,
    to_gpu_normalize_permute, Sampler_grid,
    Frameloader_video_slowfast, NumpyRandomSampler,
    merge_cf_into_cfgnode, pack_pathway_output,
    Dataloader_isaver, TDataset_over_connections,
    CHECKPOINTS_PREFIX, CHECKPOINTS, REL_YAML_PATHS)
from thes.training import (
    Manager_checkpoint_name)

from thes.slowfast import optimizer as tsf_optim

log = logging.getLogger(__name__)


def get_ll_generator(initial_seed):
    # For reproducibility, when freeze=5, dropout=0.0
    ll_generator = torch.Generator()
    ll_generator = ll_generator.manual_seed(67280421310679+initial_seed)
    return ll_generator


def _quick_shuffle_batches(dwti_to_inds_big, rgen, dwti_to_label,
        TUBES_PER_BATCH, FRAMES_PER_TUBE):
    # / Prepare dataset and data loader
    # // Batch tubes together (via their linds)
    # 11k labeled trainval tubes
    linds_order = rgen.permutation(np.arange(len(dwti_to_label)))
    b = np.arange(0, len(linds_order), TUBES_PER_BATCH)[1:]
    batched_linds = np.split(linds_order, b)

    dwti_to_label_kv = list(dwti_to_label.items())

    batches = []
    for linds in batched_linds:
        batch_binds = []
        batch_labels = []
        for li in linds:
            dwti, label = dwti_to_label_kv[li]
            binds = dwti_to_inds_big[dwti]
            replace = FRAMES_PER_TUBE > len(binds)
            chosen_binds = rgen.choice(binds, FRAMES_PER_TUBE, replace)
            batch_binds.extend(chosen_binds)
            batch_labels.extend([label]*FRAMES_PER_TUBE)
        batch_binds = np.array(batch_binds)
        batch_labels = np.array(batch_labels)
        batches.append([batch_binds, batch_labels])
    return batches


class TD_thin_over_BIG(torch.utils.data.Dataset):
    def __init__(self, BIG, batches, scaler):
        self.BIG = BIG
        self.batches = batches
        self.scaler = scaler

    def __getitem__(self, index):
        binds, labels = self.batches[index]
        # Perform h5 feature extraction
        feats = self.BIG[binds]
        feats = feats.astype(np.float32)
        if self.scaler is not None:
            feats = self.scaler.transform(feats)
        return feats, labels

    def __len__(self):
        return len(self.batches)


class Bc_dwti_labeled(Box_connections_dwti):
    labels: np.ndarray


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


class TDataset_over_box_connections_w_labels(torch.utils.data.Dataset):
    cls_vf: Dict[Tuple[Vid_daly, int], Bc_dwti_labeled] = {}
    keys_vf: List[Tuple[Vid_daly, int]]

    def __init__(self, cls_vf, dataset,
            sampler_grid, frameloader_vsf):
        self.cls_vf = cls_vf
        self.keys_vf = list(cls_vf.keys())
        self.dataset = dataset
        self.sampler_grid = sampler_grid
        self.frameloader_vsf = frameloader_vsf

    def __len__(self):
        return len(self.cls_vf)

    def __getitem__(self, index):
        key_vf = self.keys_vf[index]
        connections = self.cls_vf[key_vf]

        vid = connections['vid']
        i0 = connections['frame_ind']
        bboxes_ltrd = connections['boxes']
        labels = connections['labels']
        assert key_vf == (vid, i0)

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
            'labels': labels,
            'index': index,
            'ckey': key_vf,
            'bboxes': np.stack(prepared_bboxes, axis=0),
            'do_not_collate': True}
        return (frame_list, meta)


class Freezer(object):
    def __init__(self, model, freeze_level, bn_freeze):
        self.model = model
        self.freeze_level = freeze_level
        self.bn_freeze = bn_freeze

    @staticmethod
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    def maybe_freeze_batchnorm(self):
        if not self.bn_freeze:
            return
        for i_child, child in enumerate(self.model.children()):
            if i_child <= self.freeze_level:
                child.apply(self.set_bn_eval)

    def set_finetune_level(self):
        for param in self.model.parameters():
            param.requires_grad = True
        # freeze layers until freeze_level (inclusive)
        for i_child, child in enumerate(self.model.children()):
            if i_child <= self.freeze_level:
                for param in child.parameters():
                    param.requires_grad = False
        self.maybe_freeze_batchnorm()


class Head_roitune_c2d_1x1(nn.Module):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(Head_roitune_c2d_1x1, self).__init__()
        self._construct_head(cn, num_classes, dropout_rate)
        self.debug_outputs = debug_outputs

    def _construct_head(self, cn, num_classes, dropout_rate):
        # params
        xform_resolution = 7
        resolution = [[xform_resolution] * 2]
        scale_factor = [32]
        dim_in = [cn.RESNET.WIDTH_PER_GROUP * 32]

        pi = 0
        roi_align = ROIAlign(
                resolution[pi],
                spatial_scale=1.0/scale_factor[pi],
                sampling_ratio=0,
                aligned=True)
        self.add_module(f's{pi}_roi', roi_align)
        spool = nn.MaxPool2d(resolution[pi], stride=1)
        self.add_module(f's{pi}_spool', spool)

        if dropout_rate > 0.0:
            self.rt_dropout = nn.Dropout(dropout_rate)
        self.rt_projection = nn.Linear(
                sum(dim_in), num_classes, bias=True)
        self.rt_act = nn.Softmax(dim=-1)

    def forward(self, feats_in, bboxes0):
        out = feats_in[0]
        assert out.shape[2] == 1
        out = torch.squeeze(out, 2)
        out = self.s0_roi(out, bboxes0)
        out = self.s0_spool(out)

        # B C H W
        x = out.view(out.shape[0], -1)
        result = {}
        if self.debug_outputs:
            result['roipooled'] = x

        # Perform dropout.
        if hasattr(self, "rt_dropout"):
            x = self.rt_dropout(x)
        if self.debug_outputs:
            result['x_after_dropout'] = x

        x = x.view(x.shape[0], -1)

        if self.debug_outputs:
            result['x_after_view'] = x

        x = self.rt_projection(x)

        if self.debug_outputs:
            result['x_after_proj'] = x

        if not self.training:
            x = self.rt_act(x)
        result['x_final'] = x
        return result


class Head_fullframe_c2d_1x1(nn.Module):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(Head_fullframe_c2d_1x1, self).__init__()
        self._construct_head(cn, num_classes, dropout_rate)
        self.debug_outputs = debug_outputs

    def _construct_head(self, cn, num_classes, dropout_rate):
        # this is us following "resnet" archi
        POOL1 = SF_POOL1[cn.MODEL.ARCH]
        pool_size_head = [
            cn.DATA.NUM_FRAMES // POOL1[0][0],
            cn.DATA.CROP_SIZE // 32 // POOL1[0][1],
            cn.DATA.CROP_SIZE // 32 // POOL1[0][2]]
        # BUT, we are doing c2d_1x1, hence
        pool_size_head[0] = 1
        self.pool_size_head = pool_size_head
        dim_in = [cn.RESNET.WIDTH_PER_GROUP * 32]

        pi = 0
        avg_pool = nn.AvgPool3d(pool_size_head, stride=1)
        self.add_module(f's{pi}_avg_pool', avg_pool)

        if dropout_rate > 0.0:
            self.rt_dropout = nn.Dropout(dropout_rate)
        self.rt_projection = nn.Linear(
                sum(dim_in), num_classes, bias=True)
        self.rt_act = nn.Softmax(dim=-1)

    def forward(self, feats_in, bboxes0):
        out = feats_in[0]
        assert out.shape[2] == 1
        # Avg pool stuff
        pool_out = [self.s0_avg_pool(out)]
        x = torch.cat(pool_out, 1)
        # (n, c, t, h, w) -> (n, t, h, w, c)
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "rt_dropout"):
            x = self.rt_dropout(x)
        x = self.rt_projection(x)
        # x of shape [64, 1, 2, 2, 10] or [64, 1, 1, 1, 10]
        if not self.training:
            x = self.rt_act(x)
            x = x.mean([1, 2, 3])
        x = x.view(x.shape[0], -1)
        result = {'x_final': x}
        return result


class C2D_1x1_roitune(M_resnet):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(M_resnet, self).__init__()
        self.norm_module = get_norm(cn)
        self.enable_detection = cn.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cn)
        self.head = Head_roitune_c2d_1x1(cn, num_classes,
                dropout_rate, debug_outputs)

    def forward(self, x, bboxes0):
        # hforward_resnet_nopool
        # slowfast/models/video_model_builder.py/ResNet.forward
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x, bboxes0)
        return x

    def init_weights(self, init_std, ll_generator=None):
        # Init all weights
        init_helper.init_weights(
            self, init_std, False)

        if ll_generator is not None:
            self.head.rt_projection.weight.data.normal_(
                    mean=0.0, std=init_std, generator=ll_generator)
            self.head.rt_projection.bias.data.zero_()


class C2D_1x1_fullframe(M_resnet):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(M_resnet, self).__init__()
        self.norm_module = get_norm(cn)
        self.enable_detection = cn.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cn)
        self.head = Head_fullframe_c2d_1x1(cn, num_classes,
                dropout_rate, debug_outputs)

    def forward(self, x, bboxes0):
        # hforward_resnet_nopool
        # slowfast/models/video_model_builder.py/ResNet.forward
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.head(x, bboxes0)
        return x

    def init_weights(self, init_std, ll_generator=None):
        # Init all weights
        init_helper.init_weights(
            self, init_std, False)

        if ll_generator is not None:
            self.head.rt_projection.weight.data.normal_(
                    mean=0.0, std=init_std, generator=ll_generator)
            self.head.rt_projection.bias.data.zero_()


class Head_fullframe_sf_8x8(nn.Module):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(Head_fullframe_sf_8x8, self).__init__()
        self.num_pathways = 2
        self._construct_head(cn, num_classes, dropout_rate)
        self.debug_outputs = debug_outputs

    def _construct_head(self, cn, num_classes, dropout_rate):
        model_nframes = cn.DATA.NUM_FRAMES
        POOL1 = SF_POOL1[cn.MODEL.ARCH]
        width_per_group = cn.RESNET.WIDTH_PER_GROUP
        slowfast_alpha = cn.SLOWFAST.ALPHA

        dim_in = [
            width_per_group * 32,
            width_per_group * 32 // cn.SLOWFAST.BETA_INV]
        pool_size = [
            [model_nframes // slowfast_alpha // POOL1[0][0],
             cn.DATA.CROP_SIZE // 32 // POOL1[0][1],
             cn.DATA.CROP_SIZE // 32 // POOL1[0][2]],
            [model_nframes // POOL1[1][0],
             cn.DATA.CROP_SIZE // 32 // POOL1[1][1],
             cn.DATA.CROP_SIZE // 32 // POOL1[1][2]]
        ]
        # spat_pool_size = [IMAGE_SIZE//32, IMAGE_SIZE//32]
        # temp_pool_size = [s[0] for s in pool_size]

        for pi in range(self.num_pathways):
            avg_pool = nn.AvgPool3d(pool_size[pi], stride=1)
            self.add_module(f's{pi}_avg_pool', avg_pool)

        if dropout_rate > 0.0:
            self.rt_dropout = nn.Dropout(dropout_rate)
        self.rt_projection = nn.Linear(
                sum(dim_in), num_classes, bias=True)
        self.rt_act = nn.Softmax(dim=-1)

    def forward(self, feats_in, boxes):
        pool_out = []
        for pi in range(self.num_pathways):
            avg_pool = getattr(self, f's{pi}_avg_pool')
            out = avg_pool(feats_in[pi])
            pool_out.append(out)
        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "rt_dropout"):
            x = self.rt_dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.rt_projection(x)
        x = self.rt_act(x)
        result = {}
        result['x_final'] = x
        return result


class Head_roitune_sf_8x8(nn.Module):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(Head_roitune_sf_8x8, self).__init__()
        self.num_pathways = 2
        self._construct_head(cn, num_classes, dropout_rate)
        self.debug_outputs = debug_outputs

    def _construct_head(self, cn, num_classes, dropout_rate):
        # params
        model_nframes = cn.DATA.NUM_FRAMES
        POOL1 = SF_POOL1[cn.MODEL.ARCH]
        width_per_group = cn.RESNET.WIDTH_PER_GROUP
        xform_resolution = 7
        slowfast_alpha = cn.SLOWFAST.ALPHA

        dim_in = [
            width_per_group * 32,
            width_per_group * 32 // cn.SLOWFAST.BETA_INV,
        ]
        pool_size = [
            [model_nframes//slowfast_alpha//POOL1[0][0], 1, 1],
            [model_nframes//POOL1[1][0], 1, 1]]
        resolution = [[xform_resolution] * 2] * 2
        scale_factor = [32] * 2

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

        if dropout_rate > 0.0:
            self.rt_dropout = nn.Dropout(dropout_rate)
        self.rt_projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        self.rt_act = nn.Softmax(dim=-1)

    def forward(self, feats_in, bboxes0):
        pool_out = []
        for pi in range(self.num_pathways):
            t_pool = getattr(self, f's{pi}_tpool', None)
            out = t_pool(feats_in[pi])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)
            roi_align = getattr(self, f's{pi}_roi')
            out = roi_align(out, bboxes0)
            s_pool = getattr(self, f's{pi}_spool')
            out = s_pool(out)
            pool_out.append(out)
        # B C H W.
        x = torch.cat(pool_out, 1)

        # Perform dropout.
        if hasattr(self, "rt_dropout"):
            x = self.rt_dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.rt_projection(x)
        x = self.rt_act(x)
        result = {}
        result['x_final'] = x
        return result

class SF_8x8_custom_head(M_slowfast):
    def __init__(self, cn, head):
        super(M_slowfast, self).__init__()
        self.norm_module = get_norm(cn)
        self.enable_detection = cn.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cn)
        self.head = head

    def init_weights(self, init_std):
        init_helper.init_weights(self, init_std, False)

    def forward(self, x, bboxes0):
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
        x = self.head(x, bboxes0)
        return x


class Manager_model_checkpoints(object):
    def __init__(self, model, optimizer, model_id):
        self.model = model
        self.optimizer = optimizer
        self.model_id = model_id

    def load_model_initial(self, model):
        assert self.model_id in ['c2d_1x1', 'SLOWFAST_8x8_R50']
        CHECKPOINT_FILE_PATH = CHECKPOINTS_PREFIX/CHECKPOINTS[self.model_id]
        # Load model
        with vt_log.logging_disabled(logging.WARNING):
            sf_cu.load_checkpoint(
                CHECKPOINT_FILE_PATH, model, False, None,
                inflation=False, convert_from_caffe2=True,)

    def save_epoch(self, rundir, i_epoch):
        # model_{epoch} - "after epoch was finished"
        save_filepath = (Manager_checkpoint_name
                .get_checkpoint_path(rundir, i_epoch))
        states = {
            'i_epoch': i_epoch,
            'model_sdict': self.model.state_dict(),
            'optimizer_sdict': self.optimizer.state_dict(),
        }
        with small.QTimer() as qtr:
            torch.save(states, str(save_filepath))
        log.info(f'Saved model. Epoch {i_epoch}')
        log.debug('Saved to {}. Took {:.2f}s'.format(
            save_filepath, qtr.time))

    def restore_model_magic(
            self, checkpoint_path, starting_model=None,
            training_start_epoch=0):
        if checkpoint_path is not None:
            states = torch.load(checkpoint_path)
            self.model.load_state_dict(states['model_sdict'])
            self.optimizer.load_state_dict(states['optimizer_sdict'])
            start_epoch = states['i_epoch']
            start_epoch += 1
            log.info('Continuing training from checkpoint {}. '
                    'Epoch {} (ckpt + 1)'.format(checkpoint_path, start_epoch))
        else:
            start_epoch = training_start_epoch
            if starting_model is not None:
                states = torch.load(starting_model)
                self.model.load_state_dict(states['model_sdict'])
                log.info(('Starting new training, '
                    'initialized from model {}, at epoch {}').format(
                        starting_model, start_epoch))
            else:
                self.load_model_initial(self.model)
                log.info('Starting new training from epoch {}'.format(
                    start_epoch))
        return start_epoch

# def create_optimizer():
#     # Only params that are trained
#     params_to_update = {}
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             params_to_update[name] = param
#     paramlist = list(params_to_update.values())
#
#     lr = cf['solver.base_lr']
#     weight_decay = cf['solver.weight_decay']
#     opt_method = cf['solver.opt_method']
#     momentum = cf['solver.momentum']
#     if opt_method == 'adam':
#         optimizer = torch.optim.AdamW(paramlist,
#                 lr=lr, weight_decay=weight_decay)
#     elif opt_method == 'sgd':
#         optimizer = torch.optim.SGD(paramlist,
#                 lr=lr, momentum=momentum,
#                 weight_decay=weight_decay)
#     else:
#         raise RuntimeError()
#
#     man_ckpt = Manager_model_checkpoints(model, optimizer)
#     self.optimizer = optimizer
#     self.man_ckpt = man_ckpt

class Model_w_freezer(object):
    def __init__(self, cf, cn, n_inputs):
        # Build model
        self.cn = cn
        model = C2D_1x1_roitune(self.cn, n_inputs,
                cf['ll_dropout'], cf['debug_outputs'])
        freezer = Freezer(model, cf['freeze.level'],
                cf['freeze.freeze_bn'])
        freezer.set_finetune_level()
        self.model = model
        self.freezer = freezer

    def model_to_gpu(self):
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
        # Transfer the model to the current GPU device
        self.model = self.model.cuda(device=cur_device)

    def set_train(self):
        self.model.train()
        self.freezer.maybe_freeze_batchnorm()

    def set_eval(self):
        self.model.eval()

class Model_w_freezer_fullframe(object):
    def __init__(self, cf, cn, n_inputs):
        # Build model
        self.cn = cn
        model = C2D_1x1_fullframe(self.cn, n_inputs,
                cf['ll_dropout'], cf['debug_outputs'])
        freezer = Freezer(model, cf['freeze.level'],
                cf['freeze.freeze_bn'])
        freezer.set_finetune_level()
        self.model = model
        self.freezer = freezer

    def model_to_gpu(self):
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
        # Transfer the model to the current GPU device
        self.model = self.model.cuda(device=cur_device)

    def set_train(self):
        self.model.train()
        self.freezer.maybe_freeze_batchnorm()

    def set_eval(self):
        self.model.eval()


def add_roi_batch_indices(bboxes_t, counts=None):
    if counts is None:
        counts = np.ones(len(bboxes_t), dtype=np.int)
    batch_indices = np.repeat(np.arange(len(counts)), counts)
    batch_indices_t = torch.from_numpy(batch_indices).type(
            torch.DoubleTensor)[:, None]
    bboxes0_t = torch.cat((batch_indices_t, bboxes_t), axis=1)
    return bboxes0_t


# krgb stuff


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _config_preparations_c2d_1x1(cf_override):
    # / Configs
    # // CN for C2D_1x1 we are loading
    rel_yml_path = 'Kinetics/C2D_8x8_R50_IN1K.yaml'
    cn = basic_sf_cfg(rel_yml_path)
    merge_cf_into_cfgnode(cn, cf_override)
    # Set up last, since necessary for C2D_1x1
    cn.NUM_GPUS = 1
    cn.DATA.NUM_FRAMES = 1
    cn.DATA.SAMPLING_RATE = 1
    return cn

def _config_preparations_sf_8x8(cf_override):
    # / Configs
    # // CN for C2D_1x1 we are loading
    rel_yml_path = 'Kinetics/c2/SLOWFAST_8x8_R50.yaml'
    cn = basic_sf_cfg(rel_yml_path)
    merge_cf_into_cfgnode(cn, cf_override)
    # Set up last, since necessary for C2D_1x1
    cn.NUM_GPUS = 1
    return cn

class Manager_loader_krgb_sf8x8(object):

    class TDataset_over_krgb(torch.utils.data.Dataset):
        def __init__(self, array, bboxes, keyframes):
            self.array = array
            self.bboxes = bboxes
            self.keyframes = keyframes

        def __len__(self):
            return len(self.keyframes)

        def __getitem__(self, index):
            rgb = self.array[index]
            bbox = self.bboxes[index]
            keyframe = self.keyframes[index]
            keyframe['do_not_collate'] = True
            label = keyframe['action_id']
            # Recreate SlowFast sampling
            rgb = torch.from_numpy(rgb)
            packed_imgs = pack_pathway_output(rgb, True, 4, 0)
            return packed_imgs, bbox, label, keyframe

    def __init__(self, keyframes_rgb_fold, dataset,
            vgroup, norm_mean_cu, norm_std_cu):
        krgb_prefix = Path(keyframes_rgb_fold)
        with small.QTimer('Loading rgb.npy'):
            self.krgb_array = np.load(krgb_prefix/'rgb.npy')
        self.krgb_dict_outputs = \
                small.load_pkl(krgb_prefix/'dict_outputs.pkl')
        self.krgb_bboxes = np.vstack(self.krgb_dict_outputs['bboxes'])
        keyframes = create_keyframelist(dataset)
        # For SF8x8 we extracted only val keyframes
        subset_vids = vgroup['val']
        keyframes = [kf for kf in keyframes if kf['vid'] in subset_vids]
        self.keyframes = keyframes

        self.norm_mean_cu = norm_mean_cu
        self.norm_std_cu = norm_std_cu

    def _get_krgb_tdataset(self, vids: List[Vid_daly]):
        inds_kf = [i for i, kf in enumerate(self.keyframes)
                if kf['vid'] in vids]
        krgb_array = self.krgb_array[inds_kf]
        krgb_bboxes = self.krgb_bboxes[inds_kf]
        keyframes = [self.keyframes[i] for i in inds_kf]
        tdataset = Manager_loader_krgb_sf8x8.TDataset_over_krgb(
            krgb_array, krgb_bboxes, keyframes)
        return tdataset

    def get_eval_loader(self, vids, batch_size):
        tdataset = self._get_krgb_tdataset(vids)
        loader = torch.utils.data.DataLoader(tdataset,
                batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
                drop_last=False,
                collate_fn=sequence_batch_collate_v2)
        return loader, tdataset.keyframes

    def preprocess_data(self, data_input):
        Xts, bboxes_t, labels, keyframes = data_input
        Xts_f32c = [to_gpu_normalize_permute(
                x, self.norm_mean_cu, self.norm_std_cu) for x in Xts]
        bboxes0_t = add_roi_batch_indices(bboxes_t)
        bboxes0_c = bboxes0_t.type(torch.cuda.FloatTensor)
        labels_c = labels.cuda()
        return Xts_f32c, bboxes0_c, labels_c


class Lazy_Manager_krgb(object):
    def __init__(self, keyframes_rgb_fold, dataset,
            vgroup, norm_mean_cu, norm_std_cu,
            sset_eval, batch_size_eval):
        self.manager_args = [keyframes_rgb_fold, dataset,
                vgroup, norm_mean_cu, norm_std_cu]
        self.loader_args = [vgroup[sset_eval], batch_size_eval]

        self.man_lkrgb = None

    def actual_load(self):
        log.info('Loading (lazily) the very slow KRGB manager')
        self.man_lkrgb = Manager_loader_krgb_sf8x8(*self.manager_args)
        self.eval_krgb_loader, self.eval_krgb_keyframes = \
                self.man_lkrgb.get_eval_loader(*self.loader_args)

    def lazy_load(self):
        if self.man_lkrgb is None:
            self.actual_load()
        return self.man_lkrgb, self.eval_krgb_loader, self.eval_krgb_keyframes


class Manager_loader_krgb(object):

    class TDataset_over_krgb(torch.utils.data.Dataset):
        def __init__(self, array, bboxes, keyframes):
            self.array = array
            self.bboxes = bboxes
            self.keyframes = keyframes

        def __len__(self):
            return len(self.keyframes)

        def __getitem__(self, index):
            rgb = self.array[index]
            bbox = self.bboxes[index]
            keyframe = self.keyframes[index]
            keyframe['do_not_collate'] = True
            label = keyframe['action_id']
            return rgb, bbox, label, keyframe

    def __init__(self, keyframes_rgb_fold, dataset,
            norm_mean_cu, norm_std_cu):
        krgb_prefix = Path(keyframes_rgb_fold)
        with small.QTimer('Loading rgb.npy'):
            self.krgb_array = np.load(krgb_prefix/'rgb.npy')
        self.krgb_dict_outputs = \
                small.load_pkl(krgb_prefix/'dict_outputs.pkl')
        self.krgb_bboxes = np.vstack(self.krgb_dict_outputs['bboxes'])
        self.keyframes = create_keyframelist(dataset)

        self.norm_mean_cu = norm_mean_cu
        self.norm_std_cu = norm_std_cu

    def _get_krgb_tdataset(self, vids: List[Vid_daly]):
        inds_kf = [i for i, kf in enumerate(self.keyframes)
                if kf['vid'] in vids]
        krgb_array = self.krgb_array[inds_kf]
        krgb_bboxes = self.krgb_bboxes[inds_kf]
        keyframes = [self.keyframes[i] for i in inds_kf]
        tdataset = Manager_loader_krgb.TDataset_over_krgb(
            krgb_array, krgb_bboxes, keyframes)
        return tdataset

    def get_train_loader(self, vids, batch_size, train_sampler_rgen):
        tdataset = self._get_krgb_tdataset(vids)
        sampler = NumpyRandomSampler(tdataset, train_sampler_rgen)
        loader = torch.utils.data.DataLoader(tdataset,
                batch_size=batch_size,
                num_workers=0, pin_memory=True,
                sampler=sampler, shuffle=False,
                collate_fn=sequence_batch_collate_v2)
        return loader

    def get_eval_loader(self, vids, batch_size):
        tdataset = self._get_krgb_tdataset(vids)
        loader = torch.utils.data.DataLoader(tdataset,
                batch_size=batch_size, shuffle=False,
                num_workers=0, pin_memory=True,
                drop_last=False,
                collate_fn=sequence_batch_collate_v2)
        return loader, tdataset.keyframes

    def preprocess_data(self, data_input):
        rgbs, bboxes_t, labels, keyframes = data_input
        frame_list_f32c = [to_gpu_normalize_permute(
                rgbs, self.norm_mean_cu, self.norm_std_cu)]
        bboxes0_t = add_roi_batch_indices(bboxes_t)
        bboxes0_c = bboxes0_t.type(torch.cuda.FloatTensor)
        labels_c = labels.cuda()
        return frame_list_f32c, bboxes0_c, labels_c


class TDataset_over_lframes(torch.utils.data.Dataset):
    def __init__(self, cf, cn,
            labeled_frames: List[Frame_labeled], dataset):
        self.labeled_frames = labeled_frames
        self.dataset = dataset

        # Temporal sampling
        model_nframes = cn.DATA.NUM_FRAMES
        model_sample = cn.DATA.SAMPLING_RATE
        self.sampler_grid = Sampler_grid(model_nframes, model_sample)

        self.cn = cn
        self._is_slowfast = self.cn.MODEL.ARCH == 'slowfast'

        # Enable augmentations
        self.augment_scale = cf['train.augment.scale']
        self.augment_hflip = cf['train.augment.hflip']

    def __len__(self):
        return len(self.labeled_frames)

    def __getitem__(self, index):
        labeled_frame: Frame_labeled = self.labeled_frames[index]
        vid = labeled_frame['vid']
        i0 = labeled_frame['frame_ind']
        label = labeled_frame['label']

        video_path = str(self.dataset.videos_ocv[vid]['path'])

        # Construct labels
        # Sampling frames themselves
        nframes = self.dataset.videos_ocv[vid]['nframes']
        finds_to_sample = self.sampler_grid.apply(i0, nframes)
        with vt_cv.video_capture_open(video_path) as vcap:
            fl_u8_bgr = vt_cv.video_sample(vcap, finds_to_sample)

        imgs = _data_augment_nobox(
                self.cn, self.augment_scale, self.augment_hflip, fl_u8_bgr)

        # Pack pathways (we are going at CTHW here, TIME_DIM=1)
        packed_imgs = pack_pathway_output(imgs,
                self._is_slowfast, self.cn.SLOWFAST.ALPHA, TIME_DIM=1)

        meta = {
            'index': index,
            'label': label,
            'do_not_collate': True}
        return (packed_imgs, meta)


class TDataset_over_fgroups(torch.utils.data.Dataset):
    def __init__(self, cf, cn, frame_groups, dataset):
        self.frame_groups = frame_groups
        self.keys_vf = list(self.frame_groups.keys())
        self.dataset = dataset

        # Temporal sampling
        model_nframes = cn.DATA.NUM_FRAMES
        model_sample = cn.DATA.SAMPLING_RATE
        self.sampler_grid = Sampler_grid(model_nframes, model_sample)

        self.cn = cn
        self._is_slowfast = self.cn.MODEL.ARCH == 'slowfast'

        # Enable augmentations
        self.augment_scale = cf['train.augment.scale']
        self.augment_hflip = cf['train.augment.hflip']

    def __len__(self):
        return len(self.frame_groups)

    def __getitem__(self, index):
        key_vf = self.keys_vf[index]
        frame_group = self.frame_groups[key_vf]
        vid = frame_group[0]['vid']
        i0 = frame_group[0]['frame_ind']
        assert key_vf == (vid, i0)

        video_path = str(self.dataset.videos_ocv[vid]['path'])

        # Construct labels
        labels = np.array([f['label'] for f in frame_group])
        # Sampling frames themselves
        nframes = self.dataset.videos_ocv[vid]['nframes']
        finds_to_sample = self.sampler_grid.apply(i0, nframes)
        with vt_cv.video_capture_open(video_path) as vcap:
            fl_u8_bgr = vt_cv.video_sample(vcap, finds_to_sample)
        # Sampling boxes
        boxes_ltrd = np.vstack([f['box'] for f in frame_group])
        orig_boxes_ltrd = boxes_ltrd.copy()

        imgs, boxes = _data_augment_with_box(
            self.cn, self.augment_scale,
            self.augment_hflip,
            fl_u8_bgr, boxes_ltrd,
        )

        # Pack pathways (we are going at CTHW here, TIME_DIM=1)
        packed_imgs = pack_pathway_output(imgs,
                self._is_slowfast, self.cn.SLOWFAST.ALPHA, TIME_DIM=1)

        meta = {
            'labels': labels,
            'index': index,
            'ckey': key_vf,
            'bboxes': boxes,  # LTRD format
            'orig_boxes_ltrd': orig_boxes_ltrd,
            'do_not_collate': True}
        return (packed_imgs, meta)


def _train_epoch(
        model_wf, train_loader, optimizer, loss_fn,
        cn_optim, inputs_converter, i_epoch, ibatch_actions):
    # Train part
    model_wf.set_train()

    l_avg = snippets.misc.Averager()
    l_wavg = snippets.misc.WindowAverager(10)

    for i_batch, (data_input) in enumerate(train_loader):
        # preprocess data, transfer to GPU
        inputs, boxes, labels = inputs_converter(data_input)

        # Update learning rate
        data_size = len(train_loader)
        lr = tsf_optim.get_lr_at_epoch(cn_optim,
                i_epoch + float(i_batch) / data_size)
        set_lr(optimizer, lr)

        result = model_wf.model(inputs, boxes)
        preds = result['x_final']

        # Compute loss
        loss = loss_fn(preds, labels)
        # check nan Loss.
        sf_misc.check_nan_losses(loss)
        # Perform the backward pass.
        optimizer.zero_grad()
        loss.backward()
        # Update the parameters.
        optimizer.step()

        # Loss update
        l_avg.update(loss.item())
        l_wavg.update(loss.item())

        # Interiv evaluations
        for sslice, func in ibatch_actions:
            if snippets.check_step_sslice(i_batch, sslice):
                func(locals())


def _evaluate_krgb_perf(model, eval_loader, eval_keyframes,
        tubes_dwein_eval, tubes_dgt_eval, dataset,
        inputs_converter, cut_off_bg, print_timers=False):
    eval_timers = snippets.TicToc(
            ['softmaxes', 'acc_roc', 'cheat_app'])
    # Obtain softmaxes
    eval_timers.tic('softmaxes')
    all_softmaxes_ = []
    for i_batch, (data_input) in enumerate(eval_loader):
        frame_list_f32c, bboxes0_c, labels_c = \
            inputs_converter(data_input)
        with torch.no_grad():
            result = model(frame_list_f32c, bboxes0_c)
        pred = result['x_final']
        pred_np = pred.cpu().numpy()
        all_softmaxes_.append(pred_np)
    all_softmaxes = np.vstack(all_softmaxes_)
    if cut_off_bg:
        all_softmaxes_nobg = all_softmaxes[:, :-1]
    else:
        all_softmaxes_nobg = all_softmaxes
    eval_timers.toc('softmaxes')
    # Evaluate and print
    eval_timers.tic('acc_roc')
    gt_ids = np.array(
            [k['action_id'] for k in eval_keyframes])
    preds = np.argmax(all_softmaxes_nobg, axis=1)
    kf_acc = accuracy_score(gt_ids, preds)
    eval_timers.toc('acc_roc')
    eval_timers.tic('cheat_app')
    av_stubes_cheat: AV_dict[T_dwein_scored] = \
        cheating_tube_scoring(
            all_softmaxes_nobg, eval_keyframes,
            tubes_dwein_eval, dataset)
    df_ap_cheat, df_recall_cheat = \
        quick_tube_eval(av_stubes_cheat, tubes_dgt_eval)
    eval_timers.toc('cheat_app')

    df_ap_cheat = (df_ap_cheat*100).round(2)
    cheating_apline = '/'.join(
            df_ap_cheat.loc['all'].values.astype(str))
    log.info(' '.join((
        'K. Acc: {:.2f};'.format(kf_acc*100),
        'AP (cheating tubes): {}'.format(cheating_apline)
    )))
    if print_timers:
        log.info('Eval_timers: {}'.format(eval_timers.time_str))


def _preset_defaults(cfg):
    cfg.set_defaults_yaml("""
    seed: 42
    inputs:
        tubes_dwein: ~
        keyframes_rgb: ~
        ckpt: ~
    split_assignment: !def ['train/val', ['train/val', 'trainval/test']]
    freeze:
        level: -1
        freeze_bn: False
    debug_outputs: False
    ll_dropout: 0.5
    train:
        start_epoch: 0
        batch_save_interval_seconds: 120
        batch_size:
            train: 32
            eval: 64
        tubes:
            top_n_matches: ~
            stride: 4
            frame_dist: -1
            add_keyframes: True
        num_workers: 8
        augment:
            scale: False
            hflip: False
            color: False
    period:
        i_batch:
            loss_log: '0::10'
            eval_krgb: '::'
        i_epoch:
            eval_krgb: '0::1'
    CN:
        SOLVER:
          BASE_LR: 0.0375
          LR_POLICY: steps_with_relative_lrs
          LRS: [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
          STEPS: [0, 41, 49]
          MAX_EPOCH: 57
          MOMENTUM: 0.9
          WEIGHT_DECAY: 1e-4
          WARMUP_EPOCHS: 4.0
          WARMUP_START_LR: 0.0001
          OPTIMIZING_METHOD: sgd
    """)


class Isaver_train_epoch(snippets.isaver.Isaver_base):
    def __init__(
            self, folder, total,
            f_init, f_forward,
            i_epoch, model, optimizer,
            interval_iters=None,
            interval_seconds=120,  # every 2 minutes by default
                ):
        super(Isaver_train_epoch, self).__init__(folder, total)
        self.f_init = f_init
        self.f_forward = f_forward
        self.i_epoch = i_epoch
        self.model = model
        self.optimizer = optimizer
        self._interval_iters = interval_iters
        self._interval_seconds = interval_seconds
        self._history_size = 2

    def _get_filenames(self, i_batch) -> Dict[str, Path]:
        base_filenames = {
            'finished': self._fmt_finished.format(i_batch, self._total)}
        # base_filenames['pkl'] = Path(
        #         base_filenames['finished']).with_suffix('.pkl')
        base_filenames['ckpt'] = Path(
                base_filenames['finished']).with_suffix('.pth.tar')
        filenames = {k: self._folder/v
                for k, v in base_filenames.items()}
        return filenames

    def _restore(self):
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        start_i, ifiles = max(intermediate_files.items(),
                default=(-1, None))
        if ifiles is not None:
            ckpt_path = ifiles['ckpt']
            states = torch.load(ckpt_path)
            self.model.load_state_dict(states['model_sdict'])
            self.optimizer.load_state_dict(states['optimizer_sdict'])
            assert start_i == states['i_batch']
            assert self.i_epoch == states['i_epoch']
            log.info('Restore model at [{}, {}] from {}'.format(
                self.i_epoch, start_i, ckpt_path))
        return start_i

    def _save(self, i_batch):
        ifiles = self._get_filenames(i_batch)
        ckpt_path = ifiles['ckpt']

        # Save checkpoint
        states = {
            'i_epoch': self.i_epoch,
            'i_batch': i_batch,
            'model_sdict': self.model.state_dict(),
            'optimizer_sdict': self.optimizer.state_dict(),
        }
        torch.save(states, str(ckpt_path))

        ifiles['finished'].touch()
        log.info(f'Saved intermediate files at [{self.i_epoch}, {i_batch}]')
        self._purge_intermediate_files()

    def run(self):
        i_last = self._restore()
        countra = snippets.Counter_repeated_action(
                seconds=self._interval_seconds,
                iters=self._interval_iters)
        i_start = i_last + 1

        loader = self.f_init(i_start)
        # pbar = tqdm(loader, total=len(loader))
        pbar = tqdm(loader, total=self._total, initial=i_start)
        for i_batch, data_input in enumerate(pbar, start=i_start):
            self.f_forward(i_batch, self._total, data_input)
            if countra.check(i_batch):
                self._save(i_batch)
                log.debug(snippets.tqdm_str(pbar))
                countra.tic(i_batch)


def _debug_krgb_vis():
    rgbs_, bboxes_t_, labels_, keyframes_ = data_input
    iii = 3
    for iii in range(len(labels_)):
        img = inputs[0].cpu()[iii][:, 0]
        boxes = bboxes_t_[[iii]]
        labels = labels_[[iii]]
        data_std = cn.DATA.STD
        data_mean = cn.DATA.MEAN

        actnames = dataset.action_names + ['background']
        Y = img.numpy().transpose([1, 2, 0])*data_std+data_mean
        Y = (Y*255).clip(0, 255).astype(np.uint8)[..., ::-1]
        Y = np.ascontiguousarray(Y)
        for i, box_ltrd in enumerate(boxes):
            label = actnames[labels[i]]
            snippets.misc.cv_put_box_with_text(
                    Y, box_ltrd, text=label)
        cv2.imshow("test", Y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _debug_vis_frameload():
    pass
    # Y = (imgs[0].transpose([1, 2, 0])*255).clip(0, 255).astype(np.uint8)

    # Y = imgs[0].numpy().transpose([1, 2, 0])*data_std+data_mean

    # Y = imgs[0].astype(np.uint8)

    # Y = imgs[:, 0].numpy().transpose([1, 2, 0])*data_std+data_mean
    # Y = (Y*255).clip(0, 255).astype(np.uint8)[..., ::-1]

    # Yo = fl_u8_bgr[0]
    # cv2.imshow("test_o", Yo)
    # cv2.imshow("test", Y)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # actnames = self.dataset.action_names + ['background']
    # Y = imgs[:, 0].numpy().transpose([1, 2, 0])*data_std+data_mean
    # Y = (Y*255).clip(0, 255).astype(np.uint8)[..., ::-1]
    # Y = np.ascontiguousarray(Y)
    # for i, box_ltrd in enumerate(boxes):
    #     label = actnames[labels[i]]
    #     snippets.misc.cv_put_box_with_text(
    #             Y, box_ltrd, text=label)
    # Yo = fl_u8_bgr[0].copy()
    # for i, box_ltrd in enumerate(orig_boxes_ltrd):
    #     label = actnames[labels[i]]
    #     snippets.misc.cv_put_box_with_text(
    #             Yo, box_ltrd, text=label)
    # cv2.imshow("test", Y)
    # cv2.imshow("test_orig", Yo)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def _debug_finetune_vis():
    iii = 3
    for iii in range(len(metas)):
        img = frame_list[0][iii][:, 0]
        boxes = metas[iii]['bboxes']
        labels = metas[iii]['labels']
        data_std = cn.DATA.STD
        data_mean = cn.DATA.MEAN

        actnames = dataset.action_names + ['background']
        Y = img.numpy().transpose([1, 2, 0])*data_std+data_mean
        Y = (Y*255).clip(0, 255).astype(np.uint8)[..., ::-1]
        Y = np.ascontiguousarray(Y)
        for i, box_ltrd in enumerate(boxes):
            label = actnames[labels[i]]
            snippets.misc.cv_put_box_with_text(
                    Y, box_ltrd, text=label)
        cv2.imshow("test", Y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def _debug_full_tube_eval_vis():
    for iii in range(10):
        Y = Xts[0][iii, 0].numpy()[..., ::-1]
        Y = np.ascontiguousarray(Y)
        boxes = metas[iii]['bboxes_tldr']
        for i, box_ltrd in enumerate(boxes):
            snippets.misc.cv_put_box_with_text(
                    Y, box_ltrd, text=f'{i}',
                    rec_thickness=1,
                    text_position='right_up')
        cv2.imshow("test", Y)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class BSampler_prepared(torch.utils.data.Sampler):
    def __init__(self, samples):
        self.samples = samples

    def __iter__(self):
        for sample in self.samples:
            yield sample

    def __len__(self):
        return len(self.samples)


def finetube_perf_fulltube_evaluate(
        x_final,
        connections_f,
        tubes_dwein_eval: Dict[I_dwein, T_dwein],
        tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided],
        tubes_dgt_eval: Dict[I_dgt, T_dgt],
        dwti_to_label_eval: Dict[I_dwein, int],
        dataset: Dataset_daly_ocv,
        # stats for fulltube eval
        f_detect_mode: Literal['roipooled', 'fullframe'],
        f_nms: float,
        f_field_nms: str,
        f_field_det: str,
        ):
    iou_thresholds = [.3, .5, .7]
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(tubes_dgt_eval)
    acc_flattube_synt: float
    if f_detect_mode == 'roipooled':
        assert len(x_final) == len(connections_f)
        # Aggregation of per-frame scores
        dwti_preds: Dict[I_dwein, Dict[int, Dict]] = {}
        for cons, outputs_ in zip(connections_f.values(), x_final):
            frame_ind = cons['frame_ind']
            for (box, source, output) in zip(
                    cons['boxes'], cons['dwti_sources'], outputs_):
                pred = {'box': box, 'output': output}
                dwti_preds.setdefault(source, {})[frame_ind] = pred
        # Assignment of final score to each DWTI
        tube_softmaxes_eval: Dict[I_dwein, np.ndarray] = {}
        for dwti, preds in dwti_preds.items():
            preds_agg_ = []
            for frame_ind, pred in preds.items():
                preds_agg_.append(pred['output'])
            pred_agg = np.vstack(preds_agg_)
            tube_softmaxes_eval[dwti] = pred_agg
        # We assume input_dims == 11
        assert x_final[0].shape[-1] == 11
        tube_softmaxes_eval_bg = tube_softmaxes_eval
        tube_softmaxes_eval_nobg = {k: v[:, :-1]
                for k, v in tube_softmaxes_eval.items()}
        acc_flattube_synt = compute_flattube_syntlabel_acc(
                tube_softmaxes_eval_bg, dwti_to_label_eval)
        # Assign scores to tubes
        av_stubes_with_scores = assign_scores_to_dwt_roipooled(
                tubes_dwein_eval, tubes_dwein_prov,
                tube_softmaxes_eval_nobg, dataset)
    elif f_detect_mode =='fullframe':
        # We have wrongly attempted to be smart, fix this
        x_final = np.vstack(x_final)
        # Aggregate frame scores
        frame_scores: Dict[Tuple[Vid_daly, int], np.ndarray] = {}
        for cons, outputs_ in zip(connections_f.values(), x_final):
            vid = cons['vid']
            frame_ind = cons['frame_ind']
            frame_scores[(vid, frame_ind)] = outputs_
        assert len(frame_scores) == len(connections_f)
        # Flat accuracy not possible
        acc_flattube_synt = np.NAN
        # Assign scores to tubes
        av_stubes_with_scores = assign_scores_to_dwt_fullframe(
                tubes_dwein_eval, tubes_dwein_prov,
                frame_scores, dataset)
    else:
        raise RuntimeError()

    # Full evaluation
    av_stubes: Any = copy.deepcopy(av_stubes_with_scores)
    av_stubes = assign_scorefield(av_stubes, f_field_nms)
    av_stubes = av_stubes_above_score(av_stubes, 0.0)
    av_stubes = compute_nms_for_av_stubes(av_stubes, f_nms)
    av_stubes = assign_scorefield(av_stubes, f_field_det)

    df_ap_full = compute_ap_for_avtubes_as_df(
        av_gt_tubes, av_stubes, iou_thresholds, False, False)

    _df_ap_full = (df_ap_full*100).round(2)
    _apline = '/'.join(_df_ap_full.loc['all'].values.astype(str))
    log.info('AP (full tubes):\n{}'.format(_df_ap_full))
    log.info('Flattube synthetic acc: {:.2f}'.format(acc_flattube_synt*100))
    log.info('Full tube AP357: {}'.format(_apline))

# interacting with data
def _data_imgs_postprocess(imgs, cn):
    # Convert image to CHW keeping BGR order.
    imgs = [sf_cv2_transform.HWC2CHW(img) for img in imgs]
    # Image [0, 255] -> [0, 1].
    imgs = [img / 255.0 for img in imgs]
    imgs = [
        np.ascontiguousarray(
            img.reshape((3, imgs[0].shape[1], imgs[0].shape[2]))
        ).astype(np.float32)
        for img in imgs
    ]

    # Normalize now
    data_mean = cn.DATA.MEAN
    data_std = cn.DATA.STD
    imgs = [
        sf_cv2_transform.color_normalization(
            img,
            np.array(data_mean, dtype=np.float32),
            np.array(data_std, dtype=np.float32),
        )
        for img in imgs
    ]

    # Concat list of images to single ndarray.
    imgs = np.concatenate(
        [np.expand_dims(img, axis=1) for img in imgs], axis=1
    )

    # To RGB
    imgs = imgs[::-1, ...]

    imgs = np.ascontiguousarray(imgs)
    imgs = torch.from_numpy(imgs)
    return imgs


def _data_augment_with_box(
        cn, augment_scale, augment_hflip,
        fl_u8_bgr, boxes_ltrd):

    # # Flip to RGB
    # fl_u8 = [np.flip(x, -1) for x in fl_u8_bgr]
    # Video is in HWC, BGR format
    # Boxes should be in LTRD format

    # I am following slowfast/datasets/ava_dataset.py
    # (_images_and_boxes_preprocessing_cv2)

    imgs, boxes = fl_u8_bgr, [boxes_ltrd]
    height, width, _ = imgs[0].shape

    jmin, jmax = cn.DATA.TRAIN_JITTER_SCALES
    crop_size = cn.DATA.TRAIN_CROP_SIZE
    if augment_scale:
        imgs, boxes = sf_cv2_transform.random_short_side_scale_jitter_list(
                imgs, min_size=jmin, max_size=jmax, boxes=boxes)
        imgs, boxes = sf_cv2_transform.random_crop_list(
                imgs, crop_size, order="HWC", boxes=boxes)
    else:
        # Centercrop
        imgs = [sf_cv2_transform.scale(crop_size, img) for img in imgs]
        boxes = [sf_cv2_transform.scale_boxes(
                crop_size, boxes[0], height, width)]
        imgs, boxes = sf_cv2_transform.spatial_shift_crop_list(
            crop_size, imgs, 1, boxes=boxes)

    if augment_hflip:
        imgs, boxes = sf_cv2_transform.horizontal_flip_list(
            0.5, imgs, order="HWC", boxes=boxes)

    imgs = _data_imgs_postprocess(imgs, cn)
    boxes = sf_cv2_transform.clip_boxes_to_image(
        boxes[0], imgs[0].shape[1], imgs[0].shape[2]
    )
    return imgs, boxes


def _data_augment_nobox(
        cn, augment_scale, augment_hflip,
        fl_u8_bgr):

    imgs = fl_u8_bgr
    height, width, _ = imgs[0].shape

    jmin, jmax = cn.DATA.TRAIN_JITTER_SCALES
    crop_size = cn.DATA.TRAIN_CROP_SIZE

    if augment_scale:
        imgs, _ = sf_cv2_transform.random_short_side_scale_jitter_list(
                imgs, min_size=jmin, max_size=jmax)
        imgs, _ = sf_cv2_transform.random_crop_list(
                imgs, crop_size, order="HWC")
    else:
        imgs = [sf_cv2_transform.scale(crop_size, img) for img in imgs]
        imgs, _ = sf_cv2_transform.spatial_shift_crop_list(
            crop_size, imgs, 1)

    if augment_hflip:
        imgs, _ = sf_cv2_transform.horizontal_flip_list(
            0.5, imgs, order="HWC")
    imgs = _data_imgs_postprocess(imgs, cn)
    return imgs


# Labeled frames


def _prepare_permute_lframes(
        cf, cn, dataset, batch_size_train,
        labeled_frames: List[Frame_labeled],
        ts_rgen,
        ):
    # Permute
    order = ts_rgen.permutation(len(labeled_frames))
    labeled_frames_permuted = [labeled_frames[i] for i in order]
    # Dataset over permuted frame groups
    tdataset = TDataset_over_lframes(cf, cn, labeled_frames_permuted, dataset)
    # Perform batching ourselves
    idx_batches = snippets.misc.leqn_split(np.arange(
        len(labeled_frames_permuted)), batch_size_train, 'sharp')
    return tdataset, idx_batches


def _lframes_forward(data_input, model):
    frame_list, metas, = data_input

    labels_np = np.array([m['label'] for m in metas])
    labels_t = torch.from_numpy(labels_np)
    labels_c = labels_t.cuda()

    inputs = [x.type(torch.cuda.FloatTensor) for x in frame_list]

    result = model(inputs, None)
    preds = result['x_final']
    return labels_c, preds


# Labeled boxes


def _prepare_permute_lboxes(
        cf, cn, dataset, batch_size_train,
        labeled_boxes: List[Box_labeled],
        ts_rgen,
        ):
    # Group into frame_groups
    frame_groups: Dict[Tuple[Vid_daly, int], List[Box_labeled]] = {}
    for lbox in labeled_boxes:
        vid = lbox['vid']
        frame_ind = lbox['frame_ind']
        frame_groups.setdefault((vid, frame_ind), []).append(lbox)
    # Permute
    fg_list = list(frame_groups.items())
    fg_order = ts_rgen.permutation(len(fg_list))
    fg_list = [fg_list[i] for i in fg_order]
    frame_groups_permuted = dict(fg_list)
    # Dataset over permuted frame groups
    tdataset = TDataset_over_fgroups(cf, cn, frame_groups_permuted, dataset)
    # Perform batching ourselves
    idx_batches = snippets.misc.leqn_split(np.arange(
        len(frame_groups_permuted)), batch_size_train, 'sharp')
    return tdataset, idx_batches

def _lboxes_forward(data_input, model):
    # preprocess data, transfer to GPU
    frame_list, metas, = data_input

    # bbox transformations
    bboxes_np = [m['bboxes'] for m in metas]
    counts = np.array([len(x) for x in bboxes_np])
    batch_indices = np.repeat(np.arange(len(counts)), counts)
    bboxes0 = np.c_[batch_indices, np.vstack(bboxes_np)]
    bboxes0 = torch.from_numpy(bboxes0)
    bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)

    # labels
    labels_np = [m['labels'] for m in metas]
    labels_np = np.hstack(labels_np)
    labels_t = torch.from_numpy(labels_np)
    labels_c = labels_t.cuda()

    inputs = [x.type(torch.cuda.FloatTensor) for x in frame_list]
    boxes = bboxes0_c
    labels = labels_c

    result = model(inputs, boxes)
    preds = result['x_final']
    return labels, preds


def _ftube_extract(
        connections_f, model_wf, extract_fold,
        cf, cn, dataset):
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    sampler_grid = Sampler_grid(
            cn.DATA.NUM_FRAMES, cn.DATA.SAMPLING_RATE)
    frameloader_vsf = Frameloader_video_slowfast(
            False, cn.SLOWFAST.ALPHA, cn.DATA.CROP_SIZE, 'ltrd')

    # Means
    norm_mean_cu = np_to_gpu(cn.DATA.MEAN)
    norm_std_cu = np_to_gpu(cn.DATA.STD)

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
            x, norm_mean_cu, norm_std_cu) for x in Xts]
        # bbox transformations
        bboxes_np = [m['bboxes_tldr'] for m in metas]
        counts = np.array([len(x) for x in bboxes_np])
        batch_indices = np.repeat(np.arange(len(counts)), counts)
        bboxes0 = np.c_[batch_indices, np.vstack(bboxes_np)]
        bboxes0 = torch.from_numpy(bboxes0)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)

        with torch.no_grad():
            result = model_wf.model.forward(Xts_f32c, bboxes0_c)
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

    disaver_fold = small.mkdir(extract_fold/'disaver')
    total = len(connections_f)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
        save_interval_seconds=cf['batch_save_interval_seconds'],
        log_interval=30)

    model_wf.set_eval()
    outputs = disaver.run()

    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        key_outputs = [oo for o in outputs for oo in o[k]]
        dict_outputs[k] = key_outputs
    return dict_outputs

# EXPERIMENTS

def finetune_preextracted_krgb(workfolder, cfg_dict, add_args):
    """
    Will finetune the c2d_1x1 model on preeextracted krgb frames only
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=['CN.'])
    Ncfg_daly.set_defcfg_v2(cfg)
    _preset_defaults(cfg)
    cfg.set_defaults_yaml("""
    CN:
        SOLVER:
          BASE_LR: 0.001
    period:
        i_batch:
            loss_log: '0::10'
            eval_krgb: '::95'
    n_outputs: !def [10, [10, 11]]
    """, allow_overwrite=True)
    cf = cfg.parse()
    cn = _config_preparations_c2d_1x1(cfg.without_prefix('CN.'))

    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    torch.cuda.manual_seed(initial_seed)
    ts_rgen = np.random.default_rng(initial_seed)

    # / Data
    # General DALY level preparation
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    vgroup: Dict[str, List[Vid_daly]] = \
            Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    # Means
    norm_mean_cu = np_to_gpu(cn.DATA.MEAN)
    norm_std_cu = np_to_gpu(cn.DATA.STD)
    # Sset
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]
    vids_eval = vgroup[sset_eval]

    # Model
    model_wf = Model_w_freezer(cf, cn, cf['n_outputs'])
    CUT_OFF_BG = cf['n_outputs'] == 11
    optimizer = tsf_optim.construct_optimizer(model_wf.model, cn)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model_wf.model.init_weights(0.01)
    model_wf.model_to_gpu()

    # / Training setup
    max_epoch = cn.SOLVER.MAX_EPOCH
    man_lkrgb = Manager_loader_krgb(
            cf['inputs.keyframes_rgb'], dataset,
            norm_mean_cu, norm_std_cu)
    man_ckpt = Manager_model_checkpoints(model_wf.model, optimizer, 'c2d_1x1')

    # Restore previous run
    rundir = small.mkdir(out/'rundir')
    checkpoint_path = (Manager_checkpoint_name.find_last_checkpoint(rundir))
    if '--new' in add_args:
        Manager_checkpoint_name.rename_old_rundir(rundir)
        checkpoint_path = None
    start_epoch = (man_ckpt.restore_model_magic(checkpoint_path))

    eval_krgb_loader, eval_krgb_keyframes = man_lkrgb.get_eval_loader(
        vids_eval, cf['train.batch_size.eval'])

    # Training
    for i_epoch in range(start_epoch, max_epoch):
        # Reset seed to i_epoch + seed
        torch.manual_seed(initial_seed+i_epoch)
        torch.cuda.manual_seed(initial_seed+i_epoch)
        ts_rgen = np.random.default_rng(initial_seed+i_epoch)

        # Train part
        model_wf.set_train()
        train_loader = man_lkrgb.get_train_loader(
                vgroup[sset_train],
                cf['train.batch_size.train'], ts_rgen)
        inputs_converter = man_lkrgb.preprocess_data

        wavg_loss = snippets.misc.WindowAverager(10)

        for i_batch, (data_input) in enumerate(train_loader):
            # preprocess data, transfer to GPU
            inputs, boxes, labels = inputs_converter(data_input)

            # Update learning rate
            data_size = len(train_loader)
            lr = tsf_optim.get_lr_at_epoch(cn,
                    i_epoch + float(i_batch) / data_size)
            set_lr(optimizer, lr)

            result = model_wf.model(inputs, boxes)
            preds = result['x_final']

            # Compute loss
            loss = loss_fn(preds, labels)
            # check nan Loss.
            sf_misc.check_nan_losses(loss)
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()

            # Loss update
            wavg_loss.update(loss.item())

            if check_step(i_batch, cf['period.i_batch.loss_log']):
                log.info(f'[{i_epoch}, {i_batch}/{data_size}]'
                    f' {lr=} loss={wavg_loss}')
            if check_step(i_batch, cf['period.i_batch.eval_krgb']):
                log.info(f'Perf at [{i_epoch}, {i_batch}]')
                model_wf.set_eval()
                _evaluate_krgb_perf(model_wf.model, eval_krgb_loader,
                    eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                    dataset, man_lkrgb.preprocess_data, cut_off_bg=CUT_OFF_BG)
                model_wf.set_train()

        # Save part
        man_ckpt.save_epoch(rundir, i_epoch)
        # Eval part
        model_wf.set_eval()
        log.info(f'Perf at [{i_epoch}]')
        _evaluate_krgb_perf(model_wf.model, eval_krgb_loader,
            eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
            dataset, man_lkrgb.preprocess_data, cut_off_bg=CUT_OFF_BG)


def finetune(workfolder, cfg_dict, add_args):
    """
    Will finetune the c2d_1x1 model directly on video frames
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=['CN.'])
    Ncfg_daly.set_defcfg_v2(cfg)
    _preset_defaults(cfg)
    cfg.set_defaults_yaml("""
    detect_mode: !def ['roipooled', ['fullframe', 'roipooled']]
    """)
    cf = cfg.parse()
    cn = _config_preparations_c2d_1x1(cfg.without_prefix('CN.'))

    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    torch.cuda.manual_seed(initial_seed)
    ts_rgen = np.random.default_rng(initial_seed)

    # Data
    # General DALY level preparation
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    vgroup: Dict[str, List[Vid_daly]] = \
            Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    # Means
    norm_mean_cu = np_to_gpu(cn.DATA.MEAN)
    norm_std_cu = np_to_gpu(cn.DATA.STD)
    # Sset
    tubes_dwein_train = tubes_dwein_d[sset_train]
    tubes_dgt_train = tubes_dgt_d[sset_train]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    detect_mode = cf['detect_mode']
    stride = cf['train.tubes.stride']
    top_n_matches = cf['train.tubes.top_n_matches']
    max_distance = cf['train.tubes.frame_dist']
    batch_size_train = cf['train.batch_size.train']

    if detect_mode == 'fullframe':
        labeled_frames: List[Frame_labeled] = \
            prepare_label_fullframes_for_training(
                tubes_dgt_train, dataset, stride, max_distance)
        output_dims = 10
        model_wf = Model_w_freezer_fullframe(cf, cn, output_dims)
    elif detect_mode == 'roipooled':
        add_keyframes = cf['train.tubes.add_keyframes']
        keyframes = create_keyframelist(dataset)
        keyframes_train = [kf for kf in keyframes
                if kf['vid'] in vgroup[sset_train]]
        labeled_boxes: List[Box_labeled] = \
          prepare_label_roiboxes_for_training(
            tubes_dgt_train, dataset, stride, max_distance,
            tubes_dwein_train, keyframes_train, top_n_matches,
            add_keyframes)
        output_dims = 11
        model_wf = Model_w_freezer(cf, cn, output_dims)
    else:
        raise RuntimeError()

    cut_off_bg = output_dims == 11

    # Model
    optimizer = tsf_optim.construct_optimizer(model_wf.model, cn)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model_wf.model.init_weights(0.01)
    model_wf.model_to_gpu()

    # / Training setup
    max_epoch = cn.SOLVER.MAX_EPOCH
    man_lkrgb = Manager_loader_krgb(
            cf['inputs.keyframes_rgb'], dataset,
            norm_mean_cu, norm_std_cu)

    eval_krgb_loader, eval_krgb_keyframes = man_lkrgb.get_eval_loader(
        vgroup[sset_eval], cf['train.batch_size.eval'])

    man_ckpt = Manager_model_checkpoints(model_wf.model, optimizer, 'c2d_1x1')

    # Restore previous run
    rundir = small.mkdir(out/'rundir')
    checkpoint_path = (Manager_checkpoint_name
            .find_last_checkpoint(rundir))
    if '--new' in add_args:
        Manager_checkpoint_name.rename_old_rundir(rundir)
        checkpoint_path = None
    start_epoch = man_ckpt.restore_model_magic(checkpoint_path,
            cf['inputs.ckpt'], cf['train.start_epoch'])

    batch_size_train = cf['train.batch_size.train']
    NUM_WORKERS = cf['train.num_workers']
    # Training
    for i_epoch in range(start_epoch, max_epoch):
        log.info(f'New epoch {i_epoch}')
        fqtimer = snippets.misc.FQTimer()

        folder_epoch = small.mkdir(rundir/f'TRAIN/{i_epoch:03d}')

        # Reset seed to i_epoch + seed
        torch.manual_seed(initial_seed+i_epoch)
        torch.cuda.manual_seed(initial_seed+i_epoch)
        ts_rgen = np.random.default_rng(initial_seed+i_epoch)

        if detect_mode == 'fullframe':
            tdataset, idx_batches = _prepare_permute_lframes(
                cf, cn, dataset, batch_size_train, labeled_frames, ts_rgen)
        elif detect_mode == 'roipooled':
            tdataset, idx_batches = _prepare_permute_lboxes(
                cf, cn, dataset, batch_size_train, labeled_boxes, ts_rgen)
        else:
            raise RuntimeError()

        wavg_loss = snippets.misc.WindowAverager(10)

        # Loader
        def init_dataloader(i_start):
            remaining_idx_batches = idx_batches[i_start:]
            bsampler = BSampler_prepared(remaining_idx_batches)
            train_loader = torch.utils.data.DataLoader(tdataset,
                batch_sampler=bsampler,
                num_workers=NUM_WORKERS,
                collate_fn=sequence_batch_collate_v2)
            return train_loader

        def batch_forward(i_batch, total_batches, data_input):
            model_wf.set_train()
            # Update learning rate
            lr = tsf_optim.get_lr_at_epoch(cn,
                    i_epoch + float(i_batch) / total_batches)
            set_lr(optimizer, lr)

            if detect_mode == 'fullframe':
                labels, preds = _lframes_forward(data_input, model_wf.model)
            elif detect_mode == 'roipooled':
                labels, preds = _lboxes_forward(data_input, model_wf.model)
            else:
                raise RuntimeError()

            # Compute loss
            loss = loss_fn(preds, labels)
            # check nan Loss.
            sf_misc.check_nan_losses(loss)
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()

            # Loss update
            wavg_loss.update(loss.item())

            if check_step(i_batch, cf['period.i_batch.loss_log']):
                log.info(f'[{i_epoch}, {i_batch}/{total_batches}]'
                    f' {lr=} loss={wavg_loss}')
            if check_step(i_batch, cf['period.i_batch.eval_krgb']):
                log.info(f'Perf at [{i_epoch}, {i_batch}]')
                model_wf.set_eval()
                _evaluate_krgb_perf(model_wf.model, eval_krgb_loader,
                    eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                    dataset, man_lkrgb.preprocess_data, cut_off_bg=cut_off_bg)
                model_wf.set_train()

        isaver = Isaver_train_epoch(
                folder_epoch, len(idx_batches),
                init_dataloader, batch_forward,
                i_epoch, model_wf.model, optimizer,
                interval_seconds=cf['train.batch_save_interval_seconds'])
        isaver.run()

        # Save part
        man_ckpt.save_epoch(rundir, i_epoch)

        # Remove temporary helpers
        shutil.rmtree(folder_epoch)
        fqtimer.release(f'Epoch {i_epoch} computations')

        # Eval part
        if check_step(i_epoch, cf['period.i_epoch.eval_krgb']):
            fqtimer = snippets.misc.FQTimer()
            log.info(f'Perf at [{i_epoch}]')
            model_wf.set_eval()
            _evaluate_krgb_perf(model_wf.model, eval_krgb_loader,
                eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                dataset, man_lkrgb.preprocess_data, cut_off_bg=cut_off_bg)
            fqtimer.release(f'KRGB Evaluation at epoch {i_epoch}')


def finetune_sf8x8(workfolder, cfg_dict, add_args):
    """
    Will finetune the c2d_1x1 model directly on video frames
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=['CN.'])
    Ncfg_daly.set_defcfg_v2(cfg)
    _preset_defaults(cfg)
    cfg.set_defaults_yaml("""
    detect_mode: !def ['roipooled', ['fullframe', 'roipooled']]
    """)
    cf = cfg.parse()
    cn = _config_preparations_sf_8x8(cfg.without_prefix('CN.'))

    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    torch.cuda.manual_seed(initial_seed)
    ts_rgen = np.random.default_rng(initial_seed)

    # Data
    # General DALY level preparation
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    vgroup: Dict[str, List[Vid_daly]] = \
            Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    # Means
    norm_mean_cu = np_to_gpu(cn.DATA.MEAN)
    norm_std_cu = np_to_gpu(cn.DATA.STD)
    # Sset
    tubes_dwein_train = tubes_dwein_d[sset_train]
    tubes_dgt_train = tubes_dgt_d[sset_train]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    detect_mode = cf['detect_mode']
    stride = cf['train.tubes.stride']
    top_n_matches = cf['train.tubes.top_n_matches']
    max_distance = cf['train.tubes.frame_dist']
    batch_size_train = cf['train.batch_size.train']

    if detect_mode == 'fullframe':
        labeled_frames: List[Frame_labeled] = \
            prepare_label_fullframes_for_training(
                tubes_dgt_train, dataset, stride, max_distance)
        output_dims = 10
        head = Head_fullframe_sf_8x8(cn, output_dims,
                cf['ll_dropout'], cf['debug_outputs'])
    elif detect_mode == 'roipooled':
        add_keyframes = cf['train.tubes.add_keyframes']
        keyframes = create_keyframelist(dataset)
        keyframes_train = [kf for kf in keyframes
                if kf['vid'] in vgroup[sset_train]]
        labeled_boxes: List[Box_labeled] = \
          prepare_label_roiboxes_for_training(
            tubes_dgt_train, dataset, stride, max_distance,
            tubes_dwein_train, keyframes_train, top_n_matches,
            add_keyframes)
        output_dims = 11
        head = Head_roitune_sf_8x8(cn, output_dims,
                cf['ll_dropout'], cf['debug_outputs'])
    else:
        raise RuntimeError()

    model = SF_8x8_custom_head(cn, head)

    cut_off_bg = output_dims == 11

    # Model
    optimizer = tsf_optim.construct_optimizer(model, cn)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model.init_weights(0.01)
    # move to gpu
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    # / Training setup
    max_epoch = cn.SOLVER.MAX_EPOCH

    lmanager = Lazy_Manager_krgb(
            cf['inputs.keyframes_rgb'], dataset, vgroup,
            norm_mean_cu, norm_std_cu, sset_eval,
            cf['train.batch_size.eval'])

    man_ckpt = Manager_model_checkpoints(model, optimizer, 'SLOWFAST_8x8_R50')

    # Restore previous run
    rundir = small.mkdir(out/'rundir')
    checkpoint_path = (Manager_checkpoint_name
            .find_last_checkpoint(rundir))
    if '--new' in add_args:
        Manager_checkpoint_name.rename_old_rundir(rundir)
        checkpoint_path = None
    start_epoch = man_ckpt.restore_model_magic(checkpoint_path,
            cf['inputs.ckpt'], cf['train.start_epoch'])

    batch_size_train = cf['train.batch_size.train']
    NUM_WORKERS = cf['train.num_workers']
    # Training
    for i_epoch in range(start_epoch, max_epoch):
        log.info(f'New epoch {i_epoch}')
        fqtimer = snippets.misc.FQTimer()

        folder_epoch = small.mkdir(rundir/f'TRAIN/{i_epoch:03d}')

        # Reset seed to i_epoch + seed
        torch.manual_seed(initial_seed+i_epoch)
        torch.cuda.manual_seed(initial_seed+i_epoch)
        ts_rgen = np.random.default_rng(initial_seed+i_epoch)

        if detect_mode == 'fullframe':
            tdataset, idx_batches = _prepare_permute_lframes(
                cf, cn, dataset, batch_size_train, labeled_frames, ts_rgen)
        elif detect_mode == 'roipooled':
            tdataset, idx_batches = _prepare_permute_lboxes(
                cf, cn, dataset, batch_size_train, labeled_boxes, ts_rgen)
        else:
            raise RuntimeError()
        #
        # idx_batches = idx_batches[:5]

        wavg_loss = snippets.misc.WindowAverager(10)

        # Loader
        def init_dataloader(i_start):
            remaining_idx_batches = idx_batches[i_start:]
            bsampler = BSampler_prepared(remaining_idx_batches)
            train_loader = torch.utils.data.DataLoader(tdataset,
                batch_sampler=bsampler,
                num_workers=NUM_WORKERS,
                collate_fn=sequence_batch_collate_v2)
            return train_loader

        def batch_forward(i_batch, total_batches, data_input):
            model.train()
            # Update learning rate
            lr = tsf_optim.get_lr_at_epoch(cn,
                    i_epoch + float(i_batch) / total_batches)
            set_lr(optimizer, lr)

            if detect_mode == 'fullframe':
                labels, preds = _lframes_forward(data_input, model)
            elif detect_mode == 'roipooled':
                labels, preds = _lboxes_forward(data_input, model)
            else:
                raise RuntimeError()

            # Compute loss
            loss = loss_fn(preds, labels)
            # check nan Loss.
            sf_misc.check_nan_losses(loss)
            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()

            # Loss update
            wavg_loss.update(loss.item())

            if check_step(i_batch, cf['period.i_batch.loss_log']):
                log.info(f'[{i_epoch}, {i_batch}/{total_batches}]'
                    f' {lr=} loss={wavg_loss}')
            if check_step(i_batch, cf['period.i_batch.eval_krgb']):
                log.info(f'Perf at [{i_epoch}, {i_batch}]')
                model.eval()
                (man_lkrgb, eval_krgb_loader, eval_krgb_keyframes) = \
                        lmanager.lazy_load()
                _evaluate_krgb_perf(model, eval_krgb_loader,
                    eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                    dataset, man_lkrgb.preprocess_data, cut_off_bg=cut_off_bg)
                model.train()

        isaver = Isaver_train_epoch(
                folder_epoch, len(idx_batches),
                init_dataloader, batch_forward,
                i_epoch, model, optimizer,
                interval_seconds=cf['train.batch_save_interval_seconds'])
        isaver.run()

        # Save part
        man_ckpt.save_epoch(rundir, i_epoch)

        # Remove temporary helpers
        shutil.rmtree(folder_epoch)
        fqtimer.release(f'Epoch {i_epoch} computations')

        # Eval part
        if check_step(i_epoch, cf['period.i_epoch.eval_krgb']):
            fqtimer = snippets.misc.FQTimer()
            log.info(f'Perf at [{i_epoch}]')
            model.eval()
            (man_lkrgb, eval_krgb_loader, eval_krgb_keyframes) = \
                    lmanager.lazy_load()
            _evaluate_krgb_perf(model, eval_krgb_loader,
                eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                dataset, man_lkrgb.preprocess_data, cut_off_bg=cut_off_bg)
            fqtimer.release(f'KRGB Evaluation at epoch {i_epoch}')


def full_tube_eval(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=[''])
    Ncfg_daly.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    seed: 42
    inputs:
        tubes_dwein: ~
        keyframes_rgb: ~
        ckpt:
            fold: ~
            epoch: ~
            path: ~
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
    freeze:
        level: -1
        freeze_bn: False
    debug_outputs: False
    ll_dropout: 0.5
    tubes:
        stride: 4
    batch_save_interval_seconds: 120
    compute_split:
        chunk: 0
        total: 1
    detect_mode: !def ['roipooled', ['fullframe', 'roipooled']]
    eval:
        full_tubes:
            nms: 0.3
            field_nms: 'box_det_score'  # hscore
            field_det: 'box_det_score'  # hscore*frame_cls_score
    """)
    cf = cfg.parse()
    cn = _config_preparations_c2d_1x1(cfg.without_prefix('CN.'))

    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    torch.cuda.manual_seed(initial_seed)

    # Data
    # General DALY level preparation
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    vgroup: Dict[str, List[Vid_daly]] = \
            Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided] = \
            small.load_pkl(cf['inputs.tubes_dwein'])
    # Sset
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # / Create appropriate model
    detect_mode = cf['detect_mode']
    if detect_mode == 'roipooled':
        output_dims = 11
        model_wf = Model_w_freezer(cf, cn, output_dims)
    elif detect_mode == 'fullframe':
        output_dims = 10
        model_wf = Model_w_freezer_fullframe(cf, cn, output_dims)
    else:
        raise RuntimeError()
    model_wf.model_to_gpu()

    # / Restore checkpoint
    if cf['inputs.ckpt.path']:
        ckpt_path = cf['inputs.ckpt.path']
    elif cf['inputs.ckpt.fold']:
        epoch = cf['inputs.ckpt.epoch']
        modelname = Manager_checkpoint_name.ckpt_format.format(epoch)
        ckpt_path = str(Path(cf['inputs.ckpt.fold'])/modelname)
    else:
        raise RuntimeError()
    states = torch.load(ckpt_path)
    i_epoch = states['i_epoch']
    model_wf.model.load_state_dict(states['model_sdict'])

    # / Prepare conections, determine split (if any) and folder
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
        sample_daly_frames_from_instances(dataset, cf['tubes.stride'])
    connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    connections_f = group_tubes_on_frame_level(
            tubes_dwein_eval, frames_to_cover)
    # Here we'll run our connection split
    if cf['compute_split.total'] > 1:
        cc, ct = (cf['compute_split.chunk'], cf['compute_split.total'])
        if '--chunk' in add_args:
            cc = int(add_args[add_args.index('--chunk')+1])
            log.info('Due to shell argument chunk set {} -> {}'.format(
                cf['compute_split.chunk'], cc))
        compute_connections_f = perform_connections_split(
                connections_f, cc, ct, False)
    else:
        compute_connections_f = connections_f
        cc, ct = 0, 1

    extract_fold = out/f'chunk_{cc}_of_{ct}'
    dict_outputs = _ftube_extract(
        connections_f, model_wf, extract_fold, cf, cn, dataset)
    small.save_pkl(extract_fold/'dict_outputs.pkl', dict_outputs)
    small.save_pkl(extract_fold/'connections_f.pkl', compute_connections_f)

    attempt_eval = (cc, ct == 0, 1)
    if attempt_eval:
        assert ct == 1
        log.info('We can eval right away')
        _, dwti_to_label_eval = qload_synthetic_tube_labels(
            tubes_dgt_eval, tubes_dwein_eval, dataset)
        log.info(f'Perf at [{i_epoch}]')
        finetube_perf_fulltube_evaluate(
            dict_outputs['x_final'], connections_f,
            tubes_dwein_eval, tubes_dwein_prov, tubes_dgt_eval,
            dwti_to_label_eval, dataset, detect_mode,
            cf['eval.full_tubes.nms'],
            cf['eval.full_tubes.field_nms'],
            cf['eval.full_tubes.field_det'])


def combine_split_full_tube_eval(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=[''])
    Ncfg_daly.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    seed: 42
    inputs:
        tubes_dwein: ~
        rpath_splits: ~
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
    tubes:
        stride: 4
    mode: !def ['roi', ['roi', 'fullframe']]
    """)
    cf = cfg.parse()

    # Data
    # General DALY level preparation
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    vgroup: Dict[str, List[Vid_daly]] = \
            Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided] = \
            small.load_pkl(cf['inputs.tubes_dwein'])

    # Sset
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # Prepare
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
        sample_daly_frames_from_instances(dataset, cf['tubes.stride'])
    connections_f_: Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    connections_f_ = group_tubes_on_frame_level(
            tubes_dwein_eval, frames_to_cover)

    # search relativefolder
    import dervo.experiment
    fold_rpath = cf['inputs.rpath_splits']
    fold = (dervo.experiment.EXPERIMENT_PATH/fold_rpath).resolve()
    runs = list(glob(f'{fold}/**/dict_outputs.pkl', recursive=True))
    level = 1
    dfdict = {}
    for f in runs:
        rpath = os.path.relpath(f, fold)
        name = '.'.join(rpath.split('/')[:level])
        dfdict[name] = Path(f).parent
    log.info('{} chunks found:\n{}'.format(
        len(dfdict), pprint.pformat(list(dfdict.keys()))))

    # Chunk merge
    dict_outputs = {}
    connections_f = {}
    for name, path in dfdict.items():
        path = Path(path)
        local_outputs = small.load_pkl(path/'dict_outputs.pkl')
        for k, v in local_outputs.items():
            dict_outputs.setdefault(k, []).extend(v)
        connections_f.update(small.load_pkl(path/'connections_f.pkl'))
    # Check consistency
    if connections_f.keys() != connections_f_.keys():
        log.error('Loaded connections inconsistent with expected ones')

    # Now we can eval
    if cf['mode'] == 'roi':
        _, dwti_to_label_eval = qload_synthetic_tube_labels(
            tubes_dgt_eval, tubes_dwein_eval, dataset)
        full_tube_perf_eval(dict_outputs['x_final'], connections_f,
            dataset, dwti_to_label_eval,
            tubes_dgt_eval, tubes_dwein_eval)
    elif cf['mode'] == 'fullframe':
        full_tube_full_frame_perf_eval(
            dict_outputs['x_final'], connections_f, tubes_dwein_prov,
            dataset, tubes_dwein_eval, tubes_dgt_eval)
    else:
        raise RuntimeError()


def merge_evaluate_full_and_roi(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=[''])
    Ncfg_daly.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    seed: 42
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
    inputs:
        tubes_dwein: ~
        dict_outputs:
            roi: ~
            full: ~
        connections_f:
            roi: ~
            full: ~
    nms_scorefield: 'hscore'
    """)
    cf = cfg.parse()
    # Data
    # General DALY level preparation
    dataset = Ncfg_daly.get_dataset(cf)
    vgroup = Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided] = \
            small.load_pkl(cf['inputs.tubes_dwein'])
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # / ROI
    x_final_roi = small.load_pkl(
        cf['inputs.dict_outputs.roi'])['x_final']
    connections_f_roi = small.load_pkl(
        cf['inputs.connections_f.roi'])
    # / FULL
    x_final_full = small.load_pkl(
        cf['inputs.dict_outputs.full'])['x_final']
    connections_f_full = small.load_pkl(
        cf['inputs.connections_f.full'])

    # Actual evaluation
    iou_thresholds = [.3, .5, .7]
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(tubes_dgt_eval)

    # / ROI evaluations
    # Aggregation of per-frame scores
    dwti_preds: Dict[I_dwein, Dict[int, Dict]] = {}
    for cons, outputs_ in zip(connections_f_roi.values(), x_final_roi):
        frame_ind = cons['frame_ind']
        for (box, source, output) in zip(
                cons['boxes'], cons['dwti_sources'], outputs_):
            pred = {'box': box, 'output': output}
            dwti_preds.setdefault(source, {})[frame_ind] = pred
    # Assignment of final score to each DWTI
    tube_softmaxes_eval: Dict[I_dwein, np.ndarray] = {}
    for dwti, preds in dwti_preds.items():
        preds_agg_ = []
        for frame_ind, pred in preds.items():
            preds_agg_.append(pred['output'])
        pred_agg = np.vstack(preds_agg_)
        tube_softmaxes_eval[dwti] = pred_agg
    # We assume input_dims == 11
    assert x_final_roi[0].shape[-1] == 11
    tube_softmaxes_eval_nobg = {k: v[:, :-1]
            for k, v in tube_softmaxes_eval.items()}

    # / FULL evaluations
    # We have wrongly attempted to be smart, fix this
    x_final_full = np.vstack(x_final_full)
    # Aggregate frame scores
    frame_scores: Dict[Tuple[Vid_daly, int], np.ndarray] = {}
    for cons, outputs_ in zip(connections_f_full.values(), x_final_full):
        vid = cons['vid']
        frame_ind = cons['frame_ind']
        frame_scores[(vid, frame_ind)] = outputs_
    assert len(frame_scores) == len(connections_f_full)

    tubes_dwein = tubes_dwein_eval
    av_stubes_with_scores: AV_dict[Dict] = {}
    for dwt_index, tube in tubes_dwein.items():
        (vid, bunch_id, tube_id) = dwt_index
        # Human score from dwein tubes
        hscores = tubes_dwein_prov[dwt_index]['hscores']
        iscores = tubes_dwein_prov[dwt_index]['iscores']
        # Scores over roi
        softmaxes = tube_softmaxes_eval_nobg[dwt_index]
        scores = softmaxes.mean(axis=0)
        # Aggregated frame score
        fscores_for_tube_ = []
        for frame_ind in tube['frame_inds']:
            fscore = frame_scores.get((vid, frame_ind))
            if fscore is not None:
                fscores_for_tube_.append(fscore)
        fscores_for_tube = np.vstack(fscores_for_tube_)
        for ia, (action_name, score) in enumerate(
                zip(dataset.action_names, scores)):
            stube = cast(Dict, tube.copy())
            stube['hscore'] = hscores.mean()
            stube['iscore'] = np.nanmean(iscores)
            stube['box_det_score'] = score
            stube['box_nonbg_score'] = scores.sum()
            stube['frame_cls_score'] = fscores_for_tube.mean(0)[ia]
            stube['hscore*frame_cls_score'] = \
                    stube['hscore'] * stube['frame_cls_score']
            stube['mean(box_det_score+frame_cls_score'] = \
                    (stube['box_det_score'] + stube['frame_cls_score'])/2
            stube['mean(box_det_score, hscore*frame_cls_score)'] = \
                    (stube['box_det_score'] + stube['hscore*frame_cls_score'])/2
            (av_stubes_with_scores
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))

    av_stubes: Any = copy.deepcopy(av_stubes_with_scores)
    nms_scorefield = cf['nms_scorefield']
    av_stubes = assign_scorefield(av_stubes, nms_scorefield)
    av_stubes = av_stubes_above_score(av_stubes, 0.0)
    av_stubes = compute_nms_for_av_stubes(av_stubes, 0.3)
    av_stubes = assign_scorefield(
            av_stubes, 'mean(box_det_score, hscore*frame_cls_score)')

    df_ap_full = compute_ap_for_avtubes_as_df(
        av_gt_tubes, av_stubes, iou_thresholds, False, False)
    log.info(df_ap_full*100)
    apline = '/'.join((df_ap_full*100).round(2).loc['all'].values.astype(str))
    log.info('AP357: {}'.format(apline))
