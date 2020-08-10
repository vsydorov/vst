import copy
import string
import random
import shutil
import pprint
import yacs
import re
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from types import MethodType
from typing import (  # NOQA
    Dict, Any, List, Optional, Tuple, TypedDict, Set)
from sklearn.metrics import (
    accuracy_score, roc_auc_score)

import torch
import torch.nn as nn
import torch.utils.data
from detectron2.layers import ROIAlign
import slowfast.models
import slowfast.utils.checkpoint as cu

from slowfast.models.video_model_builder import SlowFast as M_slowfast
from slowfast.models.video_model_builder import ResNet as M_resnet
from slowfast.models.batchnorm_helper import get_norm
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.video_model_builder import _POOL1 as SF_POOL1
import slowfast.utils.misc as sf_misc

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv
from vsydorov_tools import log as vt_log

from thes.data.dataset.daly import (
    Ncfg_daly, get_daly_keyframes_to_cover,
    load_gt_and_wein_tubes, create_keyframelist)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.tubes.types import (
    Box_connections_dwti,
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    get_daly_gt_tubes, push_into_avdict,
    AV_dict, loadconvert_tubes_dwein,
    av_stubes_above_score)
from thes.data.tubes.routines import (
    group_tubes_on_frame_level,
    score_ftubes_via_objaction_overlap_aggregation,
    create_kinda_objaction_struct,
    qload_synthetic_tube_labels
    )

from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)
from thes.evaluation.meta import (
    keyframe_cls_scores, cheating_tube_scoring, quick_tube_eval)

from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.slowfast.cfg import (basic_sf_cfg)
from thes.tools import snippets
from thes.tools.video import (
    tfm_video_resize_threaded, tfm_video_center_crop)
from thes.pytorch import (
    sequence_batch_collate_v2, np_to_gpu,
    to_gpu_normalize_permute, Sampler_grid,
    Frameloader_video_slowfast, NumpyRandomSampler,
    merge_cf_into_cfgnode)

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
            'bboxes_tldr': np.stack(prepared_bboxes, axis=0),
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


class Manager_checkpoint_name(object):
    ckpt_re = r'model_at_epoch_(?P<i_epoch>\d*).pth.tar'
    ckpt_format = 'model_at_epoch_{:03d}.pth.tar'

    @classmethod
    def get_checkpoint_path(self, rundir, i_epoch) -> Path:
        save_filepath = rundir/self.ckpt_format.format(i_epoch)
        return save_filepath

    @classmethod
    def find_checkpoints(self, rundir):
        checkpoints = {}
        for subfolder_item in rundir.iterdir():
            search = re.search(self.ckpt_re, subfolder_item.name)
            if search:
                i_epoch = int(search.groupdict()['i_epoch'])
                checkpoints[i_epoch] = subfolder_item
        return checkpoints

    @classmethod
    def find_last_checkpoint(self, rundir):
        checkpoints = self.find_checkpoints(rundir)
        if len(checkpoints):
            checkpoint_path = max(checkpoints.items())[1]
        else:
            checkpoint_path = None
        return checkpoint_path


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
            cu.load_checkpoint(
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
            self, checkpoint_path, training_start_epoch=0):
        if checkpoint_path is not None:
            states = torch.load(checkpoint_path)
            self.model.load_state_dict(states['model_sdict'])
            self.optimizer.load_state_dict(states['optimizer_sdict'])
            start_epoch = states['i_epoch']
            start_epoch += 1
            log.info('Continuing training from checkpoint {}. '
                    'Epoch {} (ckpt + 1)'.format(checkpoint_path, start_epoch))
        else:
            self.load_model_initial(self.model)
            start_epoch = training_start_epoch
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


def add_roi_batch_indices(bboxes_t, counts=None):
    if counts is None:
        counts = np.ones(len(bboxes_t), dtype=np.int)
    batch_indices = np.repeat(np.arange(len(counts)), counts)
    batch_indices_t = torch.from_numpy(batch_indices).type(
            torch.DoubleTensor)[:, None]
    bboxes0_t = torch.cat((batch_indices_t, bboxes_t), axis=1)
    return bboxes0_t


# krgb stuff


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


def _prepare_labeled_connections(
        cf, dataset, tubes_dwein_train, dwti_to_label_train):
    # Frames to cover: keyframes and every 4th frame
    vids = list(dataset.videos_ocv.keys())
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
            get_daly_keyframes_to_cover(dataset, vids,
                    cf['frame_coverage.keyframes'],
                    cf['frame_coverage.subsample'])
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


class Manager_loader_krgb(object):
    def __init__(self, keyframes_rgb_fold, dataset,
            norm_mean_cu, norm_std_cu):
        krgb_prefix = Path(keyframes_rgb_fold)
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
        tdataset = TDataset_over_krgb(
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
            _prepare_labeled_connections(
                cf, dataset, tubes_dwein_d[sset_train], dwti_to_label_train)
        # Prepare sampler
        model_nframes = cn.DATA.NUM_FRAMES
        model_sample = cn.DATA.SAMPLING_RATE
        slowfast_alpha = cn.SLOWFAST.ALPHA
        self.sampler_grid = Sampler_grid(model_nframes, model_sample)
        self.frameloader_vsf = Frameloader_video_slowfast(
                False, slowfast_alpha, 256)

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
        bboxes_np = [m['bboxes_tldr'] for m in metas]
        counts = np.array([len(x) for x in bboxes_np])
        batch_indices = np.repeat(np.arange(len(counts)), counts)
        bboxes0 = np.c_[batch_indices, np.vstack(bboxes_np)]
        bboxes0 = torch.from_numpy(bboxes0)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)

        # labels
        labels_np = [m['labels'] for m in metas]
        labels_np = np.hstack(labels_np)
        labels_t = torch.from_numpy(labels_np)
        labels_cuda = labels_t.cuda()
        return frame_list_f32c, bboxes0_c, labels_cuda

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
        inputs_converter, cut_off_bg):
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
    log.info('Eval_timers: {}'.format(eval_timers.time_str))


def _preset_defaults(cfg):
    cfg.set_defaults_yaml("""
    seed: 42
    inputs:
        tubes_dwein: ~
        keyframes_rgb: ~
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
    freeze:
        level: -1
        freeze_bn: False
    debug_outputs: False
    ll_dropout: 0.5
    train:
        batch_size:
            train: 32
            eval: 64
    period:
        i_batch:
            loss_log: '::10'
            eval_krgb:
                trainset: '0::10'
                evalset: '0,50::100'
        i_epoch:
            log: '::1'
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

def f_loss_log(local_vars):
    i_epoch = local_vars['i_epoch']
    i_batch = local_vars['i_batch']
    data_size = local_vars['data_size']
    lr = local_vars['lr']
    l_avg = local_vars['l_avg']
    l_wavg = local_vars['l_wavg']
    log.info(f'[{i_epoch}, {i_batch}/{data_size}] {lr=} '
            f'loss(all/10/last): {l_avg.avg}/{l_wavg.avg}/{l_avg.last}')

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
            loss_log: '::10'
    """, allow_overwrite=True)
    cf = cfg.parse()
    cn = _config_preparations(cfg.without_prefix('CN.'))

    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    torch.cuda.manual_seed(initial_seed)
    train_sampler_rgen = np.random.default_rng(initial_seed)

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
    model_wf = Model_w_freezer(cf, cn, 10)
    optimizer = tsf_optim.construct_optimizer(model_wf.model, cn)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model_wf.model.init_weights(0.01)
    model_wf.model_to_gpu()

    # / Training setup
    max_epoch = cn.SOLVER.MAX_EPOCH
    man_lkrgb = Manager_loader_krgb(
            cf['inputs.keyframes_rgb'], dataset,
            norm_mean_cu, norm_std_cu)
    ibatch_actions = (
        (cf['period.i_batch.loss_log'], f_loss_log),
    )
    man_ckpt = Manager_model_checkpoints(model_wf.model, optimizer)

    # Restore previous run
    rundir = small.mkdir(out/'rundir')
    if '--new' in add_args:
        checkpoint_path = None
        if len(list(rundir.iterdir())) > 0:
            timestamp = datetime.fromtimestamp(
                    rundir.stat().st_mtime).strftime('%Y-%m-%d_%H:%M:%S')
            str_rnd = ''.join(random.choices(string.ascii_uppercase, k=3))
            new_foldname = f'old_{timestamp}_{str_rnd}'
            log.info(f'Existing experiment moved to {new_foldname}')
            shutil.move(rundir, rundir.parent/new_foldname)
            small.mkdir(rundir)
    else:
        checkpoint_path = (Manager_checkpoint_name.
                find_last_checkpoint(rundir))
    start_epoch = (man_ckpt.restore_model_magic(checkpoint_path))

    # Training
    for i_epoch in range(start_epoch, max_epoch):
        # Train part
        model_wf.set_train()
        train_loader = man_lkrgb.get_train_loader(
                vgroup[sset_train],
                cf['train.batch_size.train'], train_sampler_rgen)
        _train_epoch(model_wf, train_loader, optimizer,
                loss_fn, cn, man_lkrgb.preprocess_data, i_epoch,
                ibatch_actions)
        # Save part
        man_ckpt.save_epoch(rundir, i_epoch)
        # Eval part
        eval_krgb_loader, eval_krgb_keyframes = man_lkrgb.get_eval_loader(
            vids_eval, cf['train.batch_size.eval'])
        model_wf.set_eval()
        log.info(f'Perf at {i_epoch=}')
        _evaluate_krgb_perf(model_wf, eval_krgb_loader, eval_krgb_keyframes,
                tubes_dwein_eval, tubes_dgt_eval, dataset,
                man_lkrgb.preprocess_data, cut_off_bg=False)


def finetune_on_tubefeats(workfolder, cfg_dict, add_args):
    """
    Will finetune the c2d_1x1 model directly on video frames
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict,
            allowed_wo_defaults=['CN.'])
    Ncfg_daly.set_defcfg_v2(cfg)
    _preset_defaults(cfg)
    cfg.set_defaults_yaml("""
    frame_coverage:
        keyframes: True
        subsample: 4
    """)
    cf = cfg.parse()
    cn = _config_preparations(cfg.without_prefix('CN.'))

    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    torch.cuda.manual_seed(initial_seed)
    train_sampler_rgen = np.random.default_rng(initial_seed)

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
    model_wf = Model_w_freezer(cf, cn, 11)
    optimizer = tsf_optim.construct_optimizer(model_wf.model, cn)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model_wf.model.init_weights(0.01)
    model_wf.model_to_gpu()

    # / Training setup
    max_epoch = cn.SOLVER.MAX_EPOCH
    man_lkrgb = Manager_loader_krgb(
            cf['inputs.keyframes_rgb'], dataset,
            norm_mean_cu, norm_std_cu)
    man_lbox = Manager_loader_boxcons(
        tubes_dgt_d, tubes_dwein_d,
        sset_train, dataset, cn, cf,
        norm_mean_cu, norm_std_cu)

    def f_ibatch_eval_krgb_evalset(local_vars):
        i_epoch = local_vars['i_epoch']
        i_batch = local_vars['i_batch']
        model_wf.set_eval()
        eval_krgb_loader, eval_krgb_keyframes = man_lkrgb.get_eval_loader(
            vgroup[sset_eval], cf['train.batch_size.eval'])
        log.info(f'Perf at {i_epoch=}/{i_batch=}')
        _evaluate_krgb_perf(model_wf, eval_krgb_loader, eval_krgb_keyframes,
                tubes_dwein_eval, tubes_dgt_eval, dataset,
                man_lkrgb.preprocess_data, cut_off_bg=True)
    ibatch_actions = (
        (cf['period.i_batch.loss_log'], f_loss_log),
        (cf['period.i_batch.eval_krgb.evalset'],
            f_ibatch_eval_krgb_evalset))
    man_ckpt = Manager_model_checkpoints(model_wf.model, optimizer)

    # Restore previous run
    rundir = small.mkdir(out/'rundir')
    if '--new' in add_args:
        checkpoint_path = None
    else:
        checkpoint_path = (Manager_checkpoint_name.
                find_last_checkpoint(rundir))
    start_epoch = (man_ckpt.restore_model_magic(checkpoint_path))

    # Training
    for i_epoch in range(start_epoch, max_epoch):
        # Train part
        model_wf.set_train()
        train_loader = man_lbox.get_train_loader(
                cf['train.batch_size.train'], train_sampler_rgen)
        _train_epoch(model_wf, train_loader, optimizer,
                loss_fn, cn, man_lbox.preprocess_data, i_epoch,
                ibatch_actions)
