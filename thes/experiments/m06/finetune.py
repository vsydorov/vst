import copy
import numpy as np
import logging
from pathlib import Path
from types import MethodType
from typing import (  # NOQA
    Dict, Any, List, Optional, Tuple, TypedDict, Set)

import torch
import torch.nn as nn
import torch.utils.data
from vsydorov_tools import log as vt_log
from detectron2.layers import ROIAlign
import slowfast.models
import slowfast.utils.checkpoint as cu

from slowfast.models.video_model_builder import SlowFast as M_slowfast
from slowfast.models.video_model_builder import ResNet as M_resnet
from slowfast.models.batchnorm_helper import get_norm
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.video_model_builder import _POOL1 as SF_POOL1


from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv

from thes.data.dataset.daly import (
    Ncfg_daly, get_daly_keyframes_to_cover,
    load_gt_and_wein_tubes)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.tubes.types import (
    Box_connections_dwti,
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    get_daly_gt_tubes, push_into_avdict,
    AV_dict, loadconvert_tubes_dwein)
from thes.data.tubes.routines import (
    group_tubes_on_frame_level, qload_synthetic_tube_labels)
from thes.slowfast.cfg import (basic_sf_cfg)
from thes.tools import snippets
from thes.tools.video import (
    tfm_video_resize_threaded, tfm_video_center_crop)
from thes.pytorch import (
    sequence_batch_collate_v2, np_to_gpu,
    to_gpu_normalize_permute, Sampler_grid,
    Frameloader_video_slowfast)

log = logging.getLogger(__name__)


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


class TD_over_cl(torch.utils.data.Dataset):
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
    def __init__(self, sf_cfg, num_classes):
        super(M_slowfast, self).__init__()
        self.norm_module = get_norm(sf_cfg)
        self.enable_detection = sf_cfg.DETECTION.ENABLE
        self.num_pathways = 2
        self._construct_network(sf_cfg)
        self._construct_roitune(sf_cfg, num_classes)
        init_helper.init_weights(
            self, sf_cfg.MODEL.FC_INIT_STD, sf_cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_roitune(self, sf_cfg, num_classes):
        # DETECTION.ROI_XFORM_RESOLUTION
        xform_resolution = 7
        resolution = [[xform_resolution] * 2] * 2
        scale_factor = [32] * 2
        # since extraction mode is roi
        _POOL_SIZE = [[1, 1, 1], [1, 1, 1]]
        model_nframes = sf_cfg.DATA.NUM_FRAMES
        slowfast_alpha = sf_cfg.SLOWFAST.ALPHA
        head_pool_size = [
            [model_nframes//slowfast_alpha//_POOL_SIZE[0][0], 1, 1],
            [model_nframes//_POOL_SIZE[1][0], 1, 1]]
        dropout_rate = 0.5
        width_per_group = sf_cfg.RESNET.WIDTH_PER_GROUP
        dim_in=[
            width_per_group * 32,
            width_per_group * 32 // sf_cfg.SLOWFAST.BETA_INV,
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


class C2D_1x1_roitune(M_resnet):
    def __init__(self, sf_cfg, num_classes):
        super(M_resnet, self).__init__()
        self.norm_module = get_norm(sf_cfg)
        self.enable_detection = sf_cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(sf_cfg)
        self._construct_roitune(sf_cfg, num_classes)
        init_helper.init_weights(
            self, sf_cfg.MODEL.FC_INIT_STD,
            sf_cfg.RESNET.ZERO_INIT_FINAL_BN)

    def _construct_roitune(self, sf_cfg, num_classes):
        # params
        xform_resolution = 7
        resolution = [[xform_resolution] * 2]
        scale_factor = [32]
        dim_in = [sf_cfg.RESNET.WIDTH_PER_GROUP * 32]
        dropout_rate = 0.5

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

    def _headless_forward(self, x):
        # hforward_resnet_nopool
        # slowfast/models/video_model_builder.py/ResNet.forward
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        return x

    def _roitune_forward(self, feats_in, bboxes0):
        out = feats_in[0]
        assert out.shape[2] == 1
        out = torch.squeeze(out, 2)
        out = self.s0_roi(out, bboxes0)
        out = self.s0_spool(out)

        # B C H W
        x = out.view(out.shape[0], -1)

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


def build_slowfast_roitune(num_classes):
    rel_yml_path = 'Kinetics/c2/SLOWFAST_4x16_R50.yaml'
    sf_cfg = basic_sf_cfg(rel_yml_path)
    sf_cfg.NUM_GPUS = 1

    # slowfast.models.build_model
    model = SlowFast_roitune(sf_cfg, num_classes)
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)
    return model, sf_cfg

def _prepare_keyframe_cls_labels():
    dwti_to_label_train: Dict[I_dgt, int]
    cls_labels, dwti_to_label_train = qload_synthetic_tube_labels(
            tubes_dgt_d[sset_train], tubes_dwein_d[sset_train], dataset)

    # Frames to cover: keyframes and every 4th frame
    vids = list(dataset.videos_ocv.keys())
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
            get_daly_keyframes_to_cover(dataset, vids,
                    cf['frame_coverage.keyframes'],
                    cf['frame_coverage.subsample'])
    cs_vf_train: \
            Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    cs_vf_train = group_tubes_on_frame_level(
            tubes_dwein_d[sset_train], frames_to_cover)
    # Add labels
    cls_vf_train: Dict[Tuple[Vid_daly, int], Bc_dwti_labeled] = {}
    for (vid, f), bc_dwti in cs_vf_train.items():
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

def _build_model():
    # Build model
    rel_yml_path = 'Kinetics/C2D_8x8_R50_IN1K.yaml'
    sf_cfg = basic_sf_cfg(rel_yml_path)
    sf_cfg.NUM_GPUS = 1
    sf_cfg.DATA.NUM_FRAMES = 1
    sf_cfg.DATA.SAMPLING_RATE = 1

    # / slowfast.models.build_model
    model = C2D_1x1_roitune(sf_cfg, 11)
    # Determine the GPU used by the current process
    cur_device = torch.cuda.current_device()
    # Transfer the model to the current GPU device
    model = model.cuda(device=cur_device)

    model_nframes = sf_cfg.DATA.NUM_FRAMES
    model_sample = sf_cfg.DATA.SAMPLING_RATE
    slowfast_alpha = sf_cfg.SLOWFAST.ALPHA

    sampler_grid = Sampler_grid(model_nframes, model_sample)
    frameloader_vsf = Frameloader_video_slowfast(
            False, slowfast_alpha, 256)

    norm_mean = sf_cfg.DATA.MEAN
    norm_std = sf_cfg.DATA.STD
    norm_mean_t = np_to_gpu(norm_mean)
    norm_std_t = np_to_gpu(norm_std)


def _load_model_initial():
    # Load model
    # CHECKPOINT_FILE_PATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_zoo_checkpoints/kinetics400/SLOWFAST_4x16_R50.pkl'
    CHECKPOINTS_PREFIX = Path('/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_zoo_checkpoints/')
    rel = 'kin400_video_nonlocal/c2d_baseline_8x8_IN_pretrain_400k.pkl'
    CHECKPOINT_FILE_PATH = CHECKPOINTS_PREFIX/rel

    # Load model
    model.eval()
    with vt_log.logging_disabled(logging.WARNING):
        cu.load_checkpoint(
            CHECKPOINT_FILE_PATH, model, False, None,
            inflation=False, convert_from_caffe2=True,)


# EXPERIMENTS

def tubefeats_train_model(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    cfg.set_deftype("""
    seed: [42, int]
    inputs:
        tubes_dwein: [~, str]
    frame_coverage:
        keyframes: [True, bool]
        subsample: [4, int]
    split_assignment: ['train/val', ['train/val', 'trainval/test']]
    """)
    cfg.set_defaults("""
    train:
        lr: 1.0e-5
        weight_decay: 5.0e-2
        n_epochs: 120
        tubes_per_batch: 500
        frames_per_tube: 2
        period:
            log: '::1'
            eval:
                trainset: '::'
                evalset: '0::20'
    """)
    cf = cfg.parse()
    initial_seed = cf['seed']

    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    vgroup = Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)

    # Training
    torch.manual_seed(initial_seed)
    rgen = np.random.default_rng(initial_seed)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])
    period_log = cf['train.period.log']
    n_epochs = cf['train.n_epochs']
    tubes_per_batch = cf['train.tubes_per_batch']
    frames_per_tube = cf['train.frames_per_tube']
    model.train()

    BATCH_SIZE = 12
    NUM_WORKERS = 8
    for epoch in range(n_epochs):
        tdataset_cl = TD_over_cl(cls_vf_train, dataset,
                sampler_grid, frameloader_vsf)
        loader = torch.utils.data.DataLoader(tdataset_cl,
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=sequence_batch_collate_v2)
        for i_batch, (data_input) in enumerate(loader):
            frame_list, metas, = data_input
            frame_list_f32c = [
                to_gpu_normalize_permute(
                    x, norm_mean_t, norm_std_t) for x in frame_list]
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

            # pred
            pred_train = model(frame_list_f32c, bboxes0_c)

            # loss
            loss = loss_fn(pred_train, labels_cuda)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            Nb = len(loader)
            log.info(f'{epoch=}, {i_batch=}/{Nb}: loss {loss.item()}')
        if snippets.check_step_sslice(epoch, period_log):
            model.eval()
            # kacc_train = _quick_kf_eval(tkfeats_train, model, True)*100
            # kacc_eval = _quick_kf_eval(tkfeats_eval, model, True)*100
            model.train()
            # log.info(f'{epoch}: {loss.item()} '
            #         f'{kacc_train=:.2f} {kacc_eval=:.2f}')
            log.info(f'{epoch}: {loss.item()} ')


def finetune_over_extracted_frames(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    cfg.set_deftype("""
    seed: [42, int]
    inputs:
        tubes_dwein: [~, str]
        keyframes_rgb: [~, str]
    split_assignment: ['train/val', ['train/val', 'trainval/test']]
    """)
    cf = cfg.parse()

    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
