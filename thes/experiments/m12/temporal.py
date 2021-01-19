import random
import logging
from typing import (  # NOQA
    Dict, Any, List, Optional, Tuple, TypedDict, Set)

from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data

import slowfast.utils.checkpoint as sf_cu
import slowfast.utils.weight_init_helper as init_helper
import slowfast.utils.misc as sf_misc
from slowfast.models.video_model_builder import ResNet as M_resnet
from slowfast.models.batchnorm_helper import get_norm
from slowfast.models.video_model_builder import _POOL1 as SF_POOL1
from slowfast.datasets import cv2_transform as sf_cv2_transform

import vst

from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.dataset.daly import (
    Ncfg_daly, load_gt_and_wein_tubes,
    prepare_label_fullframes_for_training,
    Frame_labeled, get_keyframe_ranges
    )
from thes.feat_extract import (
    Ncfg_extractor)
from thes.slowfast.cfg import (basic_sf_cfg)
from thes.pytorch import (
    sequence_batch_collate_v2, Sampler_grid, Frameloader_video_slowfast,
    TDataset_over_keyframes, TDataset_over_connections,
    Dataloader_isaver, to_gpu_normalize_permute, np_to_gpu,
    merge_cf_into_cfgnode,
    CHECKPOINTS_PREFIX, CHECKPOINTS, REL_YAML_PATHS,
    pack_pathway_output)
from thes.slowfast import optimizer as tsf_optim
from thes.training import (
    Manager_checkpoint_name)
from thes.tools import video as tvideo

log = logging.getLogger(__name__)


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


class Manager_model_checkpoints(object):
    def __init__(self, model, optimizer, model_id):
        self.model = model
        self.optimizer = optimizer
        self.model_id = model_id

    def load_model_initial(self, model):
        assert self.model_id in ['c2d_1x1', 'SLOWFAST_8x8_R50']
        CHECKPOINT_FILE_PATH = CHECKPOINTS_PREFIX/CHECKPOINTS[self.model_id]
        # Load model
        # with vt_log.logging_disabled(logging.WARNING):
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
        with vst.QTimer() as qtr:
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


def get_device():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.cuda.current_device()
    return device

def enforce_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    rgen = np.random.default_rng(seed)
    return rgen


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


class TDataset_over_frames(torch.utils.data.Dataset):

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
        # Sampling frames themselves
        nframes = self.dataset.videos_ocv[vid]['nframes']
        finds_to_sample = self.sampler_grid.apply(i0, nframes)
        with tvideo.video_capture_open(video_path) as vcap:
            fl_u8_bgr = tvideo.video_sample(vcap, finds_to_sample)

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


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# Experiments


def train_frame_classifier(workfolder, cfg_dict, add_args):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    seed: 42
    inputs:
        tubes_dwein: ~
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
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
    period:
        i_batch:
            loss_log: '0::10'
            eval_krgb: '::'
        i_epoch:
            eval_krgb: '0::1'
    train:
        num_workers: 8
        augment:
            scale: False
            hflip: False
    """)
    cf = cfg.parse()
    cn = _config_preparations_c2d_1x1(cfg.without_prefix('CN.'))

    initial_seed = cf['seed']
    enforce_all_seeds(initial_seed)

    # prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    vgroup = Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    vids_train, vids_eval = vgroup[sset_train], vgroup[sset_eval]
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    tubes_dgt_train = tubes_dgt_d[sset_train]
    # Means
    norm_mean_cu = np_to_gpu(cn.DATA.MEAN)
    norm_std_cu = np_to_gpu(cn.DATA.STD)

    # Model
    model = C2D_1x1_fullframe(cn, 11, 0.5, False)
    optimizer = tsf_optim.construct_optimizer(model, cn)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    model.init_weights(0.01)
    device = get_device()
    model.to(device)

    # Training setup
    max_epoch = cn.SOLVER.MAX_EPOCH
    NUM_WORKERS = cf['train.num_workers']
    man_ckpt = Manager_model_checkpoints(model, optimizer, 'c2d_1x1')

    # Restore previous run
    rundir = vst.mkdir(out/'rundir')
    checkpoint_path = (Manager_checkpoint_name.find_last_checkpoint(rundir))
    start_epoch = (man_ckpt.restore_model_magic(checkpoint_path))

    # Positives
    stride = 1
    max_distance = np.inf
    labeled_frames: List[Frame_labeled] = \
        prepare_label_fullframes_for_training(
            tubes_dgt_train, dataset, stride, max_distance)

    # Get all negative frames from videos
    negative_frames = []
    for vid in vids_train:
        v = dataset.videos_ocv[vid]
        instance_franges = get_keyframe_ranges(v, include_diff=True)
        good_frames = np.arange(0, v['nframes'], stride)
        for s, e, kf in instance_franges:
            bad_frames = np.arange(s, e)
            good_frames = np.setdiff1d(good_frames, bad_frames)
            for frame_ind in good_frames:
                negative_frame = {
                        'vid': vid, 'frame_ind': frame_ind, 'label': 10}
                negative_frames.append(negative_frame)

    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    pass

    # Training
    for i_epoch in range(start_epoch, max_epoch):
        rgen = enforce_all_seeds(initial_seed+i_epoch)

        # Sample negative frames
        sample_ids = rgen.choice(len(negative_frames), size=len(labeled_frames))
        sampled_negative_frames = [negative_frames[i] for i in sample_ids]
        all_frames = labeled_frames + sampled_negative_frames
        random.shuffle(all_frames)

        tdataset = TDataset_over_frames(cf, cn, labeled_frames, dataset)

        train_loader = torch.utils.data.DataLoader(tdataset,
            num_workers=NUM_WORKERS,
            collate_fn=sequence_batch_collate_v2)

        pbar = tqdm(train_loader, total=len(tdataset))
        total_batches = len(tdataset)

        avg_loss = vst.Averager()

        for i_batch, data_input in enumerate(pbar):
            model.train()
            # Update learning rate
            lr = tsf_optim.get_lr_at_epoch(cn,
                    i_epoch + float(i_batch) / total_batches)
            set_lr(optimizer, lr)

            frame_list, metas, = data_input
            labels_np = np.array([m['label'] for m in metas])
            labels_t = torch.from_numpy(labels_np)
            labels_c = labels_t.cuda()

            inputs = [x.type(torch.cuda.FloatTensor) for x in frame_list]

            result = model(inputs, None)
            preds = result['x_final']

            # Compute loss
            loss = loss_fn(preds, labels_c)
            # check nan Loss.
            sf_misc.check_nan_losses(loss)

            # Perform the backward pass.
            optimizer.zero_grad()
            loss.backward()
            # Update the parameters.
            optimizer.step()
            # Loss update
            avg_loss.update(loss.item())

            if vst.check_step(i_batch, cf['period.i_batch.loss_log']):
                log.info(f'[{i_epoch}, {i_batch}/{total_batches}]'
                    f' {lr=} loss={avg_loss}')
