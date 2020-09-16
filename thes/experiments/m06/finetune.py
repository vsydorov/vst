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
    Dict, Any, List, Optional, Tuple, TypedDict, Set, cast)
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
    cheating_tube_scoring, quick_tube_eval)

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
    Dataloader_isaver,
    TDataset_over_connections)
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


class SlowFast_roitune(M_slowfast):
    def __init__(self, cn, num_classes):
        super(M_slowfast, self).__init__()
        self.norm_module = get_norm(cn)
        self.enable_detection = cn.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(cn)
        self._construct_roitune(cn, num_classes)
        init_helper.init_weights(
            self, cn.MODEL.FC_INIT_STD, cn.RESNET.ZERO_INIT_FINAL_BN
        )
        # Weight init
        last_layer_generator = torch.Generator(2147483647)
        self.rt_projection.weight.data.normal_(
            mean=0.0, std=0.01, generator=last_layer_generator)
        self.rt_projection.weight.data.zero_()

    def _construct_roitune(self, cn, num_classes):
        # DETECTION.ROI_XFORM_RESOLUTION
        xform_resolution = 7
        resolution = [[xform_resolution] * 2] * 2
        scale_factor = [32] * 2
        # since extraction mode is roi
        _POOL_SIZE = [[1, 1, 1], [1, 1, 1]]
        model_nframes = cn.DATA.NUM_FRAMES
        slowfast_alpha = cn.SLOWFAST.ALPHA
        head_pool_size = [
            [model_nframes//slowfast_alpha//_POOL_SIZE[0][0], 1, 1],
            [model_nframes//_POOL_SIZE[1][0], 1, 1]]
        dropout_rate = 0.5
        width_per_group = cn.RESNET.WIDTH_PER_GROUP
        dim_in=[
            width_per_group * 32,
            width_per_group * 32 // cn.SLOWFAST.BETA_INV,
        ]

        for pi in range(self.num_pathways):
            tpool = nn.AvgPool3d(
                    [head_pool_size[pi][0], 1, 1], stride=1)
            self.add_module(f'rt_s{pi}_tpool', tpool)
            roi_align = ROIAlign(
                    resolution[pi],
                    spatial_scale=1.0/scale_factor[pi],
                    sampling_ratio=0,
                    aligned=True)
            self.add_module(f'rt_s{pi}_roi', roi_align)
            spool = nn.MaxPool2d(resolution[pi], stride=1)
            self.add_module(f'rt_s{pi}_spool', spool)

        if dropout_rate > 0.0:
            self.rt_dropout = nn.Dropout(dropout_rate)
        self.rt_projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        self.rt_act = nn.Softmax(dim=-1)

    def _headless_forward(self, x):
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

    def _roitune_forward(self, x, bboxes0):
        # / Roi_Pooling
        pool_out = []
        for pi in range(self.num_pathways):
            t_pool = getattr(self, f'rt_s{pi}_tpool')
            out = t_pool(x[pi])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)

            # Roi, assuming 1 box per 1 batch_ind
            roi_align = getattr(self, f'rt_s{pi}_roi')
            out = roi_align(out, bboxes0)

            s_pool = getattr(self, f'rt_s{pi}_spool')
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
        return x

    def forward(self, x, bboxes0):
        x = self._headless_forward(x)
        x = self._roitune_forward(x, bboxes0)
        return x


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


class Head_roitune(nn.Module):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(Head_roitune, self).__init__()
        self._construct_roitune(cn, num_classes, dropout_rate)
        self.debug_outputs = debug_outputs

    def _construct_roitune(self, cn, num_classes, dropout_rate):
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


class C2D_1x1_roitune(M_resnet):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(M_resnet, self).__init__()
        self.norm_module = get_norm(cn)
        self.enable_detection = cn.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cn)
        self.head = Head_roitune(cn, num_classes,
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


class Head_fullframe(nn.Module):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(Head_fullframe, self).__init__()
        self._construct_roitune(cn, num_classes, dropout_rate)
        self.debug_outputs = debug_outputs

    def _construct_roitune(self, cn, num_classes, dropout_rate):
        # this is us following "resnet" archi
        POOL_SIZE = SF_POOL1[cn.MODEL.ARCH]
        pool_size_head = [
            cn.DATA.NUM_FRAMES // POOL_SIZE[0][0],
            cn.DATA.CROP_SIZE // 32 // POOL_SIZE[0][1],
            cn.DATA.CROP_SIZE // 32 // POOL_SIZE[0][2]]
        # BUT, we are doing c2d_1x1, hence
        pool_size_head[0] = 1
        self.pool_size_head = pool_size_head
        self.dim_in = [cn.RESNET.WIDTH_PER_GROUP * 32]

        pi = 0
        avg_pool = nn.AvgPool3d(pool_size_head, stride=1)
        self.add_module(f's{pi}_avg_pool', avg_pool)

        if dropout_rate > 0.0:
            self.rt_dropout = nn.Dropout(dropout_rate)
        self.rt_projection = nn.Linear(
                sum(self.dim_in), num_classes, bias=True)
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


class C2D_1x1_fullframe(M_resnet):
    def __init__(self, cn, num_classes,
            dropout_rate, debug_outputs):
        super(M_resnet, self).__init__()
        self.norm_module = get_norm(cn)
        self.enable_detection = cn.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cn)
        self.head = Head_fullframe(cn, num_classes,
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


class Manager_model_checkpoints(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    @staticmethod
    def load_model_initial(model):
        CHECKPOINTS_PREFIX = Path('/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_zoo_checkpoints/')
        rel = 'kin400_video_nonlocal/c2d_baseline_8x8_IN_pretrain_400k.pkl'
        CHECKPOINT_FILE_PATH = CHECKPOINTS_PREFIX/rel

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


def _config_preparations(cf_override):
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

    def get_krgb_tdataset(self, vids: List[Vid_daly]):
        inds_kf = [i for i, kf in enumerate(self.keyframes)
                if kf['vid'] in vids]
        krgb_array = self.krgb_array[inds_kf]
        krgb_bboxes = self.krgb_bboxes[inds_kf]
        keyframes = [self.keyframes[i] for i in inds_kf]
        tdataset = Manager_loader_krgb.TDataset_over_krgb(
            krgb_array, krgb_bboxes, keyframes)
        return tdataset

    def get_train_loader(self, vids, batch_size, train_sampler_rgen):
        tdataset = self.get_krgb_tdataset(vids)
        sampler = NumpyRandomSampler(tdataset, train_sampler_rgen)
        loader = torch.utils.data.DataLoader(tdataset,
                batch_size=batch_size,
                num_workers=0, pin_memory=True,
                sampler=sampler, shuffle=False,
                collate_fn=sequence_batch_collate_v2)
        return loader

    def get_eval_loader(self, vids, batch_size):
        tdataset = self.get_krgb_tdataset(vids)
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


def vis_frameload():
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
        self._is_slowfast = False

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

        # Pack pathways
        packed_imgs = pack_pathway_output(imgs,
                self._is_slowfast, self.cn.SLOWFAST.ALPHA)

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
        self._is_slowfast = False

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

        # Pack pathways
        packed_imgs = pack_pathway_output(imgs,
                self._is_slowfast, self.cn.SLOWFAST.ALPHA)

        meta = {
            'labels': labels,
            'index': index,
            'ckey': key_vf,
            'bboxes': boxes,  # LTRD format
            'orig_boxes_ltrd': orig_boxes_ltrd,
            'do_not_collate': True}
        return (packed_imgs, meta)

class Manager_loader_boxcons_improved(object):
    def __init__(self, tubes_dgt_d, tubes_dwein_d,
            sset_train, dataset, cn,
            keyframes_train,
            stride, top_n_matches,
            norm_mean_cu, norm_std_cu):
        self.dataset = dataset
        self.norm_mean_cu = norm_mean_cu
        self.norm_std_cu = norm_std_cu
        self.keyframes_train = keyframes_train

        tubes_dwein_train = tubes_dwein_d[sset_train]
        tubes_dgt_train = tubes_dgt_d[sset_train]

        # // Associate tubes
        matched_dwts: Dict[I_dgt, Dict[I_dwein, float]] = \
            get_dwein_overlaps_per_dgt(tubes_dgt_train, tubes_dwein_train)
        fg_meta, bg_meta = select_fg_bg_tubes(matched_dwts, top_n_matches)
        log.info('Selected {} FG and {} BG tubes from a total of {}'.format(
            len(fg_meta), len(bg_meta), len(tubes_dwein_train)))
        # Merge fg/bg
        tube_metas = {}
        tube_metas.update(fg_meta)
        tube_metas.update(bg_meta)
        # Break into frames, sort by distance
        self.dist_boxes_train = group_dwein_frames_wrt_kf_distance(
            dataset, stride, tubes_dwein_train, tube_metas)

    def get_labeled_boxes(self, max_distance, add_keyframes):
        # fg/bf boxes -> labels
        labeled_boxes = []
        for i, boxes in self.dist_boxes_train.items():
            if i > max_distance:
                break
            for box in boxes:
                (vid, bunch_id, tube_id) = box['dwti']
                if box['kind'] == 'fg':
                    (vid, action_name, ins_id) = box['dgti']
                    label = self.dataset.action_names.index(action_name)
                else:
                    label = len(self.dataset.action_names)
                lbox = {
                    'vid': vid,
                    'frame_ind': box['frame_ind'],
                    'box': box['box'],
                    'label': label}
                labeled_boxes.append(lbox)
        if add_keyframes:
            # Merge keyframes too
            for kf in self.keyframes_train:
                action_name = kf['action_name']
                label = self.dataset.action_names.index(action_name)
                lbox = {'vid': kf['vid'],
                        'frame_ind': kf['frame0'],
                        'box': kf['bbox'],
                        'label': label}
                labeled_boxes.append(lbox)
        return labeled_boxes

    @staticmethod
    def get_frame_groups(labeled_boxes):
        # Group into frame_groups
        frame_groups: Dict[Tuple[Vid_daly, int], List] = {}
        for lbox in labeled_boxes:
            vid = lbox['vid']
            frame_ind = lbox['frame_ind']
            frame_groups.setdefault((vid, frame_ind), []).append(lbox)
        return frame_groups

    def preprocess_data(self, data_input):
        frame_list, metas, = data_input
        frame_list_f32c = [
            to_gpu_normalize_permute(
                x, self.norm_mean_cu,
                self.norm_std_cu) for x in frame_list]

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
        return frame_list_f32c, bboxes0_c, labels_c


class Manager_loader_fullframes(object):
    """
    Based on Manager_loader_boxcons_improved
    """
    def __init__(self, tubes_dgt_train, dataset,
            keyframes_train,
            norm_mean_cu, norm_std_cu):
        self.tubes_dgt_train = tubes_dgt_train
        self.dataset = dataset
        self.keyframes_train = keyframes_train
        self.norm_mean_cu = norm_mean_cu
        self.norm_std_cu = norm_std_cu

    def get_labeled_frames(self):
        labeled_frames = []
        # Merge keyframes too
        for kf in self.keyframes_train:
            action_name = kf['action_name']
            label = self.dataset.action_names.index(action_name)
            lbox = {'vid': kf['vid'],
                    'frame_ind': kf['frame0'],
                    'box': kf['bbox'],
                    'label': label}
            labeled_frames.append(lbox)
        return labeled_frames

    def preprocess_data(self, data_input):
        frame_list, metas, = data_input
        frame_list_f32c = [
            to_gpu_normalize_permute(
                x, self.norm_mean_cu,
                self.norm_std_cu) for x in frame_list]

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
        return frame_list_f32c, bboxes0_c, labels_c


class Manager_loader_boxcons(object):
    def __init__(self, tubes_dgt_d, tubes_dwein_d,
            sset_train, dataset, cn, cf,
            norm_mean_cu, norm_std_cu):
        self.dataset = dataset
        self.norm_mean_cu = norm_mean_cu
        self.norm_std_cu = norm_std_cu

        # Prepare labeled tubes
        dwti_to_label_train: Dict[I_dgt, int]
        cls_labels, dwti_to_label_train = qload_synthetic_tube_labels(
                tubes_dgt_d[sset_train], tubes_dwein_d[sset_train], dataset)
        self.cls_vf_train: Dict[Tuple[Vid_daly, int], Bc_dwti_labeled] = \
            self._prepare_labeled_connections(
                cf, dataset, tubes_dwein_d[sset_train], dwti_to_label_train)
        # Prepare sampler
        model_nframes = cn.DATA.NUM_FRAMES
        model_sample = cn.DATA.SAMPLING_RATE
        slowfast_alpha = cn.SLOWFAST.ALPHA
        self.sampler_grid = Sampler_grid(model_nframes, model_sample)
        self.frameloader_vsf = Frameloader_video_slowfast(
                False, slowfast_alpha, 256)

    @staticmethod
    def _prepare_labeled_connections(
            cf, dataset, tubes_dwein_train, dwti_to_label_train):
        # Frames to cover: keyframes and every 4th frame
        frames_to_cover: Dict[Vid_daly, np.ndarray] = \
            sample_daly_frames_from_instances(
                    dataset, cf['frame_coverage.subsample'])
        cs_vf_train_unlabeled: \
                Dict[Tuple[Vid_daly, int], Box_connections_dwti]
        cs_vf_train_unlabeled = group_tubes_on_frame_level(
                tubes_dwein_train, frames_to_cover)
        # Add labels
        cls_vf_train: Dict[Tuple[Vid_daly, int], Bc_dwti_labeled] = {}
        for (vid, f), bc_dwti in cs_vf_train_unlabeled.items():
            good_i = []
            labels = []
            for i, dwti in enumerate(bc_dwti['dwti_sources']):
                if dwti in dwti_to_label_train:
                    label = dwti_to_label_train[dwti]
                    good_i.append(i)
                    labels.append(label)
            if len(good_i):
                vid = bc_dwti['vid']
                frame_ind = bc_dwti['frame_ind']
                boxes = [bc_dwti['boxes'][i] for i in good_i]
                dwti_sources = [bc_dwti['dwti_sources'][i] for i in good_i]
                bc_dwti_labeled: Bc_dwti_labeled = {
                        'vid': vid,
                        'frame_ind': frame_ind,
                        'dwti_sources': dwti_sources,
                        'boxes': boxes,
                        'labels': np.array(labels)}
                cls_vf_train[(vid, f)] = bc_dwti_labeled
        return cls_vf_train

    def get_train_loader(self, batch_size, rgen):
        NUM_WORKERS = 8
        td_bc_labeled = TDataset_over_box_connections_w_labels(
            self.cls_vf_train, self.dataset,
            self.sampler_grid, self.frameloader_vsf)
        sampler = NumpyRandomSampler(td_bc_labeled, rgen)
        loader = torch.utils.data.DataLoader(td_bc_labeled,
            batch_size=batch_size, num_workers=NUM_WORKERS,
            sampler=sampler, shuffle=False,
            pin_memory=True,
            collate_fn=sequence_batch_collate_v2)
        return loader

    def preprocess_data(self, data_input):
        frame_list, metas, = data_input
        frame_list_f32c = [
            to_gpu_normalize_permute(
                x, self.norm_mean_cu,
                self.norm_std_cu) for x in frame_list]

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
        return frame_list_f32c, bboxes0_c, labels_c


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


def _evaluate_krgb_perf(model_wf, eval_loader, eval_keyframes,
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
            result = model_wf.model(frame_list_f32c, bboxes0_c)
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


def full_tube_perf_eval(outputs, connections_f,
        dataset, dwti_to_label_eval,
        tubes_dgt_eval, tubes_dwein_eval):
    assert len(outputs) == len(connections_f)
    # Aggregation of per-frame scores
    dwti_preds: Dict[I_dwein, Dict[int, Dict]] = {}
    for cons, outputs_ in zip(connections_f.values(), outputs):
        frame_ind = cons['frame_ind']
        for (box, source, output) in zip(
                cons['boxes'], cons['dwti_sources'], outputs_):
            pred = {
                'box': box,
                'output': output}
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
    tube_softmaxes_eval_bg = tube_softmaxes_eval
    tube_softmaxes_eval_nobg = {k: v[:, :-1]
            for k, v in tube_softmaxes_eval.items()}

    acc_flattube_synt = compute_flattube_syntlabel_acc(
            tube_softmaxes_eval_bg, dwti_to_label_eval)
    av_stubes_eval: AV_dict[T_dwein_scored] = \
        quick_assign_scores_to_dwein_tubes(
            tubes_dwein_eval, tube_softmaxes_eval_nobg, dataset)
    df_ap_full, df_recall_full = quick_tube_eval(av_stubes_eval, tubes_dgt_eval)

    _df_ap_full = (df_ap_full*100).round(2)
    _apline = '/'.join(_df_ap_full.loc['all'].values.astype(str))
    log.info('AP (full tubes):\n{}'.format(_df_ap_full))
    log.info('Flattube synthetic acc: {:.2f}'.format(acc_flattube_synt*100))
    log.info('Full tube AP357: {}'.format(_apline))

def full_tube_full_frame_perf_eval(
        x_final_broken, connections_f,
        tubes_dwein_prov, dataset, tubes_dwein_eval, tubes_dgt_eval):
    # // fullframe evaluation
    # We have wrongly attempted to be smart and unbatch out connections
    x_final = np.vstack(x_final_broken)
    # Aggregate frame scores
    frame_scores: Dict[Tuple[Vid_daly, int], np.ndarray] = {}
    for cons, outputs_ in zip(connections_f.values(), x_final):
        vid = cons['vid']
        frame_ind = cons['frame_ind']
        frame_scores[(vid, frame_ind)] = outputs_
    assert len(frame_scores) == len(connections_f)

    av_stubes_eval_augm: AV_dict[Dict] = {}
    for dwt_index, tube in tubes_dwein_eval.items():
        (vid, bunch_id, tube_id) = dwt_index
        # Human score from dwein tubes
        hscores = tubes_dwein_prov[dwt_index]['hscores']
        # Aggregated frame score
        fscores_for_tube = []
        for frame_ind in tube['frame_inds']:
            fscore = frame_scores.get((vid, frame_ind))
            if fscore is not None:
                fscores_for_tube.append(fscore)
        fscores_for_tube = np.vstack(fscores_for_tube)
        for ia, action_name in enumerate(dataset.action_names):
            stube = cast(T_dwein_scored, tube.copy())
            stube['hscore'] = hscores.mean()
            stube['fscore'] = fscores_for_tube.mean(0)[ia]
            stube['hfscore'] = stube['hscore'] * stube['fscore']
            (av_stubes_eval_augm
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))

    def assign_scorefield(av_stubes, score_field):
        for a, v_stubes in av_stubes.items():
            for v, stubes in v_stubes.items():
                for stube in stubes:
                    stube['score'] = stube[score_field]

    iou_thresholds = [.3, .5, .7]
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(tubes_dgt_eval)
    # Full evaluation
    av_stubes = copy.deepcopy(av_stubes_eval_augm)
    assign_scorefield(av_stubes, 'hscore')
    av_stubes = av_stubes_above_score(av_stubes, 0.0)
    av_stubes = compute_nms_for_av_stubes(av_stubes, 0.3)

    assign_scorefield(av_stubes, 'hfscore')
    df_ap_hfscore = compute_ap_for_avtubes_as_df(
        av_gt_tubes, av_stubes, iou_thresholds, False, False)

    _df_ap_full = (df_ap_hfscore*100).round(2)
    _apline = '/'.join(_df_ap_full.loc['all'].values.astype(str))
    log.info('AP (full tubes):\n{}'.format(_df_ap_full))
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


def _lframes_forward(data_input, model_wf):
    frame_list, metas, = data_input

    labels_np = np.array([m['label'] for m in metas])
    labels_t = torch.from_numpy(labels_np)
    labels_c = labels_t.cuda()

    inputs = [x.type(torch.cuda.FloatTensor) for x in frame_list]

    result = model_wf.model(inputs, None)
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

def _lboxes_forward(data_input, model_wf):
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

    result = model_wf.model(inputs, boxes)
    preds = result['x_final']
    return labels, preds

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
    cn = _config_preparations(cfg.without_prefix('CN.'))

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
    man_ckpt = Manager_model_checkpoints(model_wf.model, optimizer)

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
                _evaluate_krgb_perf(model_wf, eval_krgb_loader,
                    eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                    dataset, man_lkrgb.preprocess_data, cut_off_bg=CUT_OFF_BG)
                model_wf.set_train()

        # Save part
        man_ckpt.save_epoch(rundir, i_epoch)
        # Eval part
        model_wf.set_eval()
        log.info(f'Perf at [{i_epoch}]')
        _evaluate_krgb_perf(model_wf, eval_krgb_loader,
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
    cn = _config_preparations(cfg.without_prefix('CN.'))

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

    man_ckpt = Manager_model_checkpoints(model_wf.model, optimizer)

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
                labels, preds = _lframes_forward(data_input, model_wf)
            elif detect_mode == 'roipooled':
                labels, preds = _lboxes_forward(data_input, model_wf)
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
                _evaluate_krgb_perf(model_wf, eval_krgb_loader,
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
            _evaluate_krgb_perf(model_wf, eval_krgb_loader,
                eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                dataset, man_lkrgb.preprocess_data, cut_off_bg=cut_off_bg)
            fqtimer.release(f'KRGB Evaluation at epoch {i_epoch}')


def full_tube_eval(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=['CN.', 'train.', 'period.'])
    Ncfg_daly.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    seed: 42
    inputs:
        tubes_dwein: ~
        keyframes_rgb: ~
        ckpt: ~
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
        enabled: False
        chunk: !def {default: 0, evalcheck: "VALUE >= 0"}
        total: 1
    mode: !def ['roi', ['roi', 'fullframe']]
    """)
    cf = cfg.parse()
    cn = _config_preparations(cfg.without_prefix('CN.'))

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
    # Means
    norm_mean_cu = np_to_gpu(cn.DATA.MEAN)
    norm_std_cu = np_to_gpu(cn.DATA.STD)
    # Sset
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # Model
    if cf['mode'] == 'roi':
        n_outputs = 11
        model_wf = Model_w_freezer(cf, cn, n_outputs)
    elif cf['mode'] == 'fullframe':
        n_outputs = 10
        model_wf = Model_w_freezer_fullframe(cf, cn, n_outputs)
    else:
        raise RuntimeError()

    model_wf.model_to_gpu()
    # Restore checkpoint
    states = torch.load(cf['inputs.ckpt'])
    model_wf.model.load_state_dict(states['model_sdict'])

    # Prepare
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
        sample_daly_frames_from_instances(dataset, cf['tubes.stride'])
    connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    connections_f = group_tubes_on_frame_level(
            tubes_dwein_eval, frames_to_cover)
    # Here we'll run our connection split
    if cf['compute_split.enabled']:
        cc, ct = (cf['compute_split.chunk'], cf['compute_split.total'])
        connections_f = perform_connections_split(connections_f, cc, ct, False)

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    sampler_grid = Sampler_grid(cn.DATA.NUM_FRAMES, cn.DATA.SAMPLING_RATE)
    frameloader_vsf = Frameloader_video_slowfast(
            False, cn.SLOWFAST.ALPHA, cn.DATA.CROP_SIZE, 'ltrd')

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

    disaver_fold = small.mkdir(out/'disaver')
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

    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    small.save_pkl(out/'connections_f.pkl', connections_f)

    if not cf['compute_split.enabled']:
        log.info('We can eval right away')
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


def finetune_on_fullframes(workfolder, cfg_dict, add_args):
    """
    Will finetune the c2d_1x1 model directly on video frames, sans boxes
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=['CN.'])
    Ncfg_daly.set_defcfg_v2(cfg)
    _preset_defaults(cfg)
    cf = cfg.parse()
    cn = _config_preparations(cfg.without_prefix('CN.'))

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
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # Model
    model_wf = Model_w_freezer_fullframe(cf, cn, 10)
    optimizer = tsf_optim.construct_optimizer(model_wf.model, cn)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model_wf.model.init_weights(0.01)
    model_wf.model_to_gpu()

    # / Training setup
    max_epoch = cn.SOLVER.MAX_EPOCH
    man_lkrgb = Manager_loader_krgb(
            cf['inputs.keyframes_rgb'], dataset,
            norm_mean_cu, norm_std_cu)

    keyframes = create_keyframelist(dataset)
    keyframes_train = [kf for kf in keyframes
            if kf['vid'] in vgroup[sset_train]]

    manli_full = Manager_loader_fullframes(
        tubes_dgt_d[sset_train], dataset,
        keyframes_train, norm_mean_cu, norm_std_cu)

    eval_krgb_loader, eval_krgb_keyframes = man_lkrgb.get_eval_loader(
        vgroup[sset_eval], cf['train.batch_size.eval'])

    man_ckpt = Manager_model_checkpoints(model_wf.model, optimizer)

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
    frame_dist = cf['train.tubes.frame_dist']
    add_keyframes = cf['train.tubes.add_keyframes']
    NUM_WORKERS = cf['train.num_workers']

    # Eval check after restore
    i_epoch = start_epoch-1
    if checkpoint_path and check_step(i_epoch, cf['period.i_epoch.eval_krgb']):
        fqtimer = snippets.misc.FQTimer()
        log.info(f'Perf at [{i_epoch}]')
        model_wf.set_eval()
        _evaluate_krgb_perf(model_wf, eval_krgb_loader,
            eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
            dataset, man_lkrgb.preprocess_data, cut_off_bg=False)
        fqtimer.release(f'KRGB Evaluation at epoch {i_epoch}')

    # Training
    for i_epoch in range(start_epoch, max_epoch):
        log.info(f'New epoch {i_epoch}')
        fqtimer = snippets.misc.FQTimer()

        folder_epoch = small.mkdir(rundir/f'TRAIN/{i_epoch:03d}')

        # Reset seed to i_epoch + seed
        torch.manual_seed(initial_seed+i_epoch)
        torch.cuda.manual_seed(initial_seed+i_epoch)
        ts_rgen = np.random.default_rng(initial_seed+i_epoch)

        labeled_frames = manli_full.get_labeled_frames()
        # Permute
        prm_order = ts_rgen.permutation(len(labeled_frames))
        labeled_frames_permuted = [labeled_frames[i] for i in prm_order]
        # Fake the frame_groups
        frame_groups_permuted: Dict[Tuple[Vid_daly, int], List] = {}
        for lfp in labeled_frames_permuted:
            vid = lfp['vid']
            frame_ind = lfp['frame_ind']
            frame_groups_permuted[(vid, frame_ind)] = [lfp]
        # Dataset over permuted frame groups
        td = TDataset_over_fgroups(
                cf, cn, frame_groups_permuted, manli_full.dataset)
        # Perform batching ourselves
        idx_batches = snippets.misc.leqn_split(np.arange(
            len(frame_groups_permuted)), batch_size_train, 'sharp')

        wavg_loss = snippets.misc.WindowAverager(10)

        # Loader
        def init_dataloader(i_start):
            remaining_idx_batches = idx_batches[i_start:]
            bsampler = BSampler_prepared(remaining_idx_batches)
            train_loader = torch.utils.data.DataLoader(td,
                batch_sampler=bsampler,
                num_workers=NUM_WORKERS,
                collate_fn=sequence_batch_collate_v2)
            return train_loader

        def batch_forward(i_batch, total_batches, data_input):
            model_wf.set_train()
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

            # Update learning rate
            lr = tsf_optim.get_lr_at_epoch(cn,
                    i_epoch + float(i_batch) / total_batches)
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
                log.info(f'[{i_epoch}, {i_batch}/{total_batches}]'
                    f' {lr=} loss={wavg_loss}')
            if check_step(i_batch, cf['period.i_batch.eval_krgb']):
                log.info(f'Perf at [{i_epoch}, {i_batch}]')
                model_wf.set_eval()
                _evaluate_krgb_perf(model_wf, eval_krgb_loader,
                    eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                    dataset, man_lkrgb.preprocess_data, cut_off_bg=False)
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
            _evaluate_krgb_perf(model_wf, eval_krgb_loader,
                eval_krgb_keyframes, tubes_dwein_eval, tubes_dgt_eval,
                dataset, man_lkrgb.preprocess_data, cut_off_bg=False)
            fqtimer.release(f'KRGB Evaluation at epoch {i_epoch}')
