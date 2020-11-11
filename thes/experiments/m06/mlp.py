import copy
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import (  # NOQA
        Dict, Any, Optional, List, cast,
        Tuple, TypedDict, Iterable, Literal)

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from vsydorov_tools import small

from thes.data.dataset.daly import (
    Ncfg_daly, load_gt_and_wein_tubes,
    group_dwein_frames_wrt_kf_distance,
    sample_daly_frames_from_instances,
    Frame_labeled, Box_labeled,
    prepare_label_fullframes_for_training,
    prepare_label_roiboxes_for_training
)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)

from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    Frametube_scored,
    AV_dict, Box_connections_dwti,
    Tube_daly_wein_as_provided,
    av_stubes_above_score, push_into_avdict,
    get_daly_gt_tubes, dtindex_filter_split
)
from thes.data.tubes.routines import (
    get_dwein_overlaps_per_dgt,
    select_fg_bg_tubes,
    qload_synthetic_tube_labels,
    compute_flattube_syntlabel_acc,
    quick_assign_scores_to_dwein_tubes
)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)

from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)
from thes.evaluation.meta import (
    cheating_tube_scoring, quick_tube_eval,
    assign_scorefield, assign_scores_to_dwt_roipooled,
    assign_scores_to_dwt_fullframe)

from thes.tools import snippets
from thes.tools.snippets import check_step_sslice as check_step
from thes.pytorch import (
    sequence_batch_collate_v2, NumpyRandomSampler)
from thes.training import (
    Manager_checkpoint_name)
from thes.mlp import (
    Ncfg_kfeats, E_tkfeats, create_preextracted_feats_manager,
    Manager_feats_tubes_dwein_roipooled,
    Manager_feats_tubes_dwein_full,
    define_mlp_model, mlp_perf_kf_evaluate,
    mlp_perf_fulltube_evaluate, mlp_perf_display,
    quick_accuracy_over_kfeat)

log = logging.getLogger(__name__)


class Train_sampler(object):
    def train_step(self, model, optimizer, loss_fn, i_epoch):
        raise NotImplementedError()


class Full_train_sampler(Train_sampler):
    def __init__(self, tkfeats_train):
        self.tkfeats_train = tkfeats_train

    def train_step(self, model, optimizer, loss_fn, i_epoch):
        pred_train = model(self.tkfeats_train['X'])['x_final']
        loss = loss_fn(pred_train, self.tkfeats_train['Y'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        return loss


class Partial_Train_sampler(Train_sampler):

    class TD_over_feats(torch.utils.data.Dataset):
        def __init__(self, feats, labels, keyframes):
            self.feats = feats
            self.labels = labels
            self.keyframes = keyframes

        def __len__(self):
            return len(self.feats)

        def __getitem__(self, index):
            feat = self.feats[index]
            label = self.labels[index]
            keyframe = self.keyframes[index]
            keyframe['do_not_collate'] = True
            return feat, label, keyframe

    def __init__(self, tkfeats_train, initial_seed,
            TRAIN_BATCH_SIZE, period_ibatch_loss_log):
        td_sstrain = Partial_Train_sampler.TD_over_feats(
                tkfeats_train['X'],
                tkfeats_train['Y'],
                tkfeats_train['kf'])
        train_sampler_rgen = np.random.default_rng(initial_seed)
        sampler = NumpyRandomSampler(td_sstrain, train_sampler_rgen)

        self.train_krgb_loader = torch.utils.data.DataLoader(td_sstrain,
                batch_size=TRAIN_BATCH_SIZE,
                num_workers=0, pin_memory=True,
                sampler=sampler, shuffle=False,
                collate_fn=sequence_batch_collate_v2)
        self.period_ibatch_loss_log = period_ibatch_loss_log

    def train_step(self, model, optimizer, loss_fn, i_epoch):
        l_avg = snippets.misc.Averager()
        for i_batch, (data_input) in enumerate(self.train_krgb_loader):
            feats, labels, keyframes = data_input
            result = model(feats)
            pred_train = result['x_final']
            loss = loss_fn(pred_train, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            l_avg.update(loss.item())
            if snippets.check_step_sslice(
                    i_batch, self.period_ibatch_loss_log):
                Nb = len(self.train_krgb_loader)
                log.info(f'{i_epoch=}, {i_batch=}/{Nb}: loss {loss.item()}')
        return l_avg.avg

# Management of dwein roipooled features

class Box_labeled_linked(Box_labeled):
    bi: Optional[int]


def _link_lboxes_to_exfeats(
        labeled_boxes: List[Box_labeled],
        man_feats_dwt: Manager_feats_tubes_dwein_roipooled
        ) -> List[Box_labeled_linked]:
    labeled_linked_boxes = []
    for lbox in labeled_boxes:
        lbox = cast(Box_labeled_linked, copy.deepcopy(lbox))
        if lbox['dwti'] is not None:
            bi = man_feats_dwt.dwti_f_bi[lbox['dwti']][lbox['frame_ind']]
        else:
            bi = None
        lbox['bi'] = bi
        labeled_linked_boxes.append(lbox)
    return labeled_linked_boxes


class TDataset_over_labeled_linked_boxes(torch.utils.data.Dataset):
    def __init__(self,
            labeled_linked_boxes: List[Box_labeled_linked],
            man_feats_dwt: Manager_feats_tubes_dwein_roipooled,
            KFX: np.ndarray):
        # Group into frame_groups
        frame_groups: Dict[Tuple[Vid_daly, int], List[Box_labeled_linked]] = {}
        for lbox in labeled_linked_boxes:
            vid = lbox['vid']
            frame_ind = lbox['frame_ind']
            frame_groups.setdefault((vid, frame_ind), []).append(lbox)
        self.frame_groups = frame_groups
        self.keys_vf = list(frame_groups.keys())
        self.BIG = man_feats_dwt.BIG
        self.scaler = man_feats_dwt.scaler
        self.KFX = KFX

    def __getitem__(self, index):
        key_vf = self.keys_vf[index]
        boxes: List[Box_labeled_linked] = self.frame_groups[key_vf]

        # Features
        bi_presence = np.array([b.get('bi') is not None for b in boxes])
        if bi_presence.all():
            bis = np.array([b['bi'] for b in boxes])
            feats = self.BIG[bis]
            feats = feats.astype(np.float32)
        else:
            feats = np.zeros((len(boxes),
                self.BIG.shape[-1]), dtype=np.float32)
            for i, box in enumerate(boxes):
                if box.get('bi') is not None:
                    feats[i] = self.BIG[box['bi']]
                elif box.get('kfi') is not None:
                    feats[i] = self.KFX[box['kfi']]
                else:
                    raise RuntimeError()

        if self.scaler is not None:
            feats = self.scaler.transform(feats)

        # Labels
        labels = np.array([b['label'] for b in boxes])

        meta = {'feats': feats, 'labels': labels, 'do_not_collate': True}
        return (meta,)

    def __len__(self):
        return len(self.frame_groups)


# Management of dwein fullframe features

class Frame_labeled_linked(Frame_labeled):
    exfeat_ind: int


def _link_lframes_to_exfeats(
        labeled_frames: List[Frame_labeled],
        man_feats_dwt: Manager_feats_tubes_dwein_full
        ) -> List[Frame_labeled_linked]:
    # Assign link to extracted feature
    vf_to_exfeat_ind = {}
    for i, vf in enumerate(list(man_feats_dwt.connections_f.keys())):
        vf_to_exfeat_ind[vf] = i
    errors = 0
    ll_frames: List[Frame_labeled_linked] = []
    for lframe in labeled_frames:
        vf = (lframe['vid'], lframe['frame_ind'])
        lframe = cast(Frame_labeled_linked, copy.deepcopy(lframe))
        exfeat_ind = vf_to_exfeat_ind.get(vf)
        if exfeat_ind is None:
            errors += 1
            continue
        lframe['exfeat_ind'] = exfeat_ind
        ll_frames.append(lframe)
    log.info(f'Got {errors=} when linking frames to exfeats')
    return ll_frames


class TDataset_over_labeled_linked_frames(torch.utils.data.Dataset):
    def __init__(self,
            labeled_linked_frames: List[Frame_labeled_linked],
            man_feats_dwt: Manager_feats_tubes_dwein_full):
        self.labeled_linked_frames = labeled_linked_frames
        self.fullframe_feats = man_feats_dwt.fullframe_feats
        self.scaler = man_feats_dwt.scaler

    def __getitem__(self, index):
        llframe = self.labeled_linked_frames[index]

        feat = self.fullframe_feats[llframe['exfeat_ind']]
        label = np.array(llframe['label'])

        meta = {'feats': feat[None],
                'labels': label[None],
                'do_not_collate': True}
        return (meta,)

    def __len__(self):
        return len(self.labeled_linked_frames)


class Checkpointer(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

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
        log.debug(f'Saved model. Epoch {i_epoch}')
        log.debug('Saved to {}. Took {:.2f}s'.format(
            save_filepath, qtr.time))

    def restore_model_magic(
            self, checkpoint_path,
            starting_model=None, training_start_epoch=0):
        if checkpoint_path is not None:
            # Continue training
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
                log.info(('Starting new training, '
                    'empty model, at epoch {}').format(start_epoch))
        return start_epoch


def _get_trainloader_rnd_sampler(tdataset, batch_size, rgen):
    NUM_WORKERS = 0
    sampler = NumpyRandomSampler(tdataset, rgen)
    loader = torch.utils.data.DataLoader(tdataset,
        batch_size=batch_size, num_workers=NUM_WORKERS,
        sampler=sampler, shuffle=False,
        pin_memory=True,
        collate_fn=sequence_batch_collate_v2)
    return loader


def _kffeats_train_mlp_single_run(
        cf, fold_trmodels, i,
        initial_seed, man_feats_dwt,
        tkfeats_train: E_tkfeats,
        tkfeats_eval: E_tkfeats,
        tubes_dwein_eval, tubes_dgt_eval,
        tubes_dwein_prov, dataset, sset_eval):

    torch.manual_seed(initial_seed)
    period_log = cf['train.period.log']
    period_eval_evalset = cf['train.period.eval']

    output_dims = cf['net.n_outputs']

    n_epochs = cf['train.n_epochs']
    D_in = tkfeats_train['X'].shape[-1]
    model = define_mlp_model(cf, D_in, output_dims)
    model.init_weights(0.01, initial_seed)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])

    trsampler: Train_sampler
    if cf['train.batch_mode'] == 'full':
        trsampler = Full_train_sampler(tkfeats_train)
    elif cf['train.batch_mode'] == 'partial':
        train_batch_size = cf['train.batch_mode_partial.train_batch_size']
        period_ibatch_loss_log = cf['train.period.ibatch_loss']
        trsampler = Partial_Train_sampler(
                tkfeats_train, initial_seed,
                train_batch_size, period_ibatch_loss_log)
    else:
        raise NotImplementedError()

    _, dwti_to_label_eval = qload_synthetic_tube_labels(
        tubes_dgt_eval, tubes_dwein_eval, dataset)
    kf_cut_last = output_dims == 11
    # fullframe eval stats

    def evaluate(i_epoch):
        model.eval()
        result = mlp_perf_kf_evaluate(
                model, tkfeats_train, tkfeats_eval,
                tubes_dwein_eval, tubes_dgt_eval,
                dataset, output_dims)
        if cf['eval.full_tubes.enabled']:
            result_fulltube = mlp_perf_fulltube_evaluate(
                    model, man_feats_dwt,
                    tubes_dwein_eval, tubes_dwein_prov,
                    tubes_dgt_eval, dwti_to_label_eval,
                    dataset, output_dims,
                    cf['eval.full_tubes.detect_mode'],
                    cf['eval.full_tubes.nms'],
                    cf['eval.full_tubes.field_nms'],
                    cf['eval.full_tubes.field_det'])
            result.update(result_fulltube)
        model.train()
        log.info(f'Perf at {i_epoch=}')
        mlp_perf_display(result, sset_eval)
        return result

    for i_epoch in range(n_epochs):
        avg_loss = trsampler.train_step(model, optimizer, loss_fn, i_epoch)
        if snippets.check_step_sslice(i_epoch, period_log):
            log.info(f'Loss at {i_epoch=}: {avg_loss}')
            model.eval()
            kacc_train = quick_accuracy_over_kfeat(
                    tkfeats_train, model, kf_cut_last)*100
            kacc_eval = quick_accuracy_over_kfeat(
                    tkfeats_eval, model, kf_cut_last)*100
            model.train()
            log.info(f'Qperf at {i_epoch=}: '
                    f'{kacc_train=:.2f} {kacc_eval=:.2f}')
        if snippets.check_step_sslice(i_epoch, period_eval_evalset):
            result = evaluate(i_epoch)

    # Save the final model
    save_filepath = str(fold_trmodels/f'finished_{i}.ckpt')
    states = {
            'i_epoch': i_epoch,
            'model_sdict': model.state_dict(),
    }
    torch.save(states, str(save_filepath))
    # Eval
    result = evaluate(i_epoch)
    return result


# Experiments


def kffeats_train_mlp(workfolder, cfg_dict, add_args):
    """
    Train a classification MLP on keyframes only. Simple to do.
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict)
    Ncfg_daly.set_defcfg_v2(cfg)
    Ncfg_kfeats.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    inputs:
        tubes_dwein: ~
        tubes_dwein_feats:
            fold: ~
            kind: !def ['roipooled', ['fullframe', 'roipooled']]
    seed: 42
    split_assignment: !def ['train/val', ['train/val', 'trainval/test']]
    data_scaler: !def ['keyframes', ['keyframes', 'no']]
    net:
        kind: !def ['layer1', ['layer0', 'layer1']]
        layer1:
            H: 32
        ll_dropout: 0.5
        n_outputs: !def [10, [10, 11]]
    train:
        lr: 1.0e-5
        weight_decay: 5.0e-2
        n_epochs: 2001
        period:
            log: '0::500'
            eval: '0::500'
            ibatch_loss: '::'  # only relevant for partial
        batch_mode: !def ['full', ['full', 'partial']]
        batch_mode_partial:
            train_batch_size: 32
    eval:
        full_tubes:
            enabled: False
            detect_mode: !def ['roipooled', ['fullframe', 'roipooled']]
            nms: 0.3
            field_nms: 'box_det_score'  # hscore
            field_det: 'box_det_score'  # hscore*frame_cls_score
    n_trials: 1
    """)
    cf = cfg.parse()
    # params
    initial_seed = cf['seed']
    n_trials = cf['n_trials']
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # Inputs
    dataset = Ncfg_daly.get_dataset(cf)
    vgroup = Ncfg_daly.get_vids(cf, dataset)
    tkfeats_d, scaler = Ncfg_kfeats.load_scale(cf, vgroup)
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided] = \
            small.load_pkl(cf['inputs.tubes_dwein'])
    detect_mode = cf['eval.full_tubes.detect_mode']
    if cf['eval.full_tubes.enabled']:
        man_feats_dwt = create_preextracted_feats_manager(
                cf, scaler, detect_mode)
    else:
        man_feats_dwt = None

    # / Torch section
    tkfeats_train = tkfeats_d[sset_train]
    tkfeats_eval = tkfeats_d[sset_eval]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    log.info(f'Train/eval splits: {sset_train} {sset_eval=}')
    # Folder for trained models
    fold_trmodels = small.mkdir(out/'trained_models')

    def experiment(i):
        log.info(f'Experiment {i}')
        result = _kffeats_train_mlp_single_run(
            cf, fold_trmodels, i,
            initial_seed+i, man_feats_dwt,
            tkfeats_train, tkfeats_eval,
            tubes_dwein_eval, tubes_dgt_eval,
            tubes_dwein_prov, dataset, sset_eval)
        return result

    if n_trials == 1:
        experiment(0)
        return

    isaver = snippets.Isaver_simple(
            small.mkdir(out/'isaver_ntrials'),
            range(n_trials), experiment)
    trial_results = isaver.run()

    df_keys = ['df_recall_cheat', 'df_ap_cheat']
    scalar_keys = ['kacc_train', 'kacc_eval']
    if 'df_ap_full' in trial_results[0]:
        df_keys.append('df_ap_full')
    if 'df_ap_full' in trial_results[0]:
        df_keys.append('df_recall_full')
    if 'acc_flattube_synt' in trial_results[0]:
        scalar_keys.append('acc_flattube_synt')
    avg_result = {}
    for k in scalar_keys:
        avg_result[k] = np.mean([tr[k] for tr in trial_results])
    for k in df_keys:
        to_avg = [tr[k] for tr in trial_results]
        df = pd.concat(to_avg,
                keys=range(len(to_avg)), axis=1).mean(axis=1, level=1)
        avg_result[k] = df
    log.info('Results for average over {} trials'.format(n_trials))
    mlp_perf_display(avg_result, sset_eval)


def tubefeats_dist_train_mlp(workfolder, cfg_dict, add_args):
    """
    Training of MLP trainfeats in the same way as we finetune stuff
    """
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict)
    Ncfg_daly.set_defcfg_v2(cfg)
    Ncfg_kfeats.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    inputs:
        tubes_dwein: ~
        tubes_dwein_feats:
            fold: ~
            kind: !def ['roipooled', ['fullframe', 'roipooled']]
        ckpt: ~
    seed: 42
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
    data_scaler: !def ['no', ['keyframes', 'no']]
    net:
        kind: !def ['layer1', ['layer0', 'layer1']]
        layer1:
            H: 32
        ll_dropout: 0.5
        n_outputs: !def [~, [10, 11]]
    train:
        lr: 1.0e-05
        weight_decay: 5.0e-2
        start_epoch: 0
        n_epochs: 120
        batch_size: 32
        tubes:
            top_n_matches: 1
            stride: 4
            frame_dist: 16
            add_keyframes: True
    period:
        i_batch:
            loss_log: '::'
        i_epoch:
            loss_log: '0::1'
            q_eval: '::'
            full_eval: '0,1,2,3,4::5'
    eval:
        full_tubes:
            enabled: True
            detect_mode: !def ['roipooled', ['fullframe', 'roipooled']]
            nms: 0.3
            field_nms: 'box_det_score'  # hscore
            field_det: 'box_det_score'  # hscore*frame_cls_score
    """)
    cf = cfg.parse()
    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    # Data
    # General DALY level preparation
    dataset = Ncfg_daly.get_dataset(cf)
    vgroup = Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # keyframe feats
    tkfeats_d, scaler = Ncfg_kfeats.load_scale(cf, vgroup)
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided] = \
            small.load_pkl(cf['inputs.tubes_dwein'])

    # synthetic tube labels
    _, dwti_to_label_eval = qload_synthetic_tube_labels(
            tubes_dgt_d[sset_eval], tubes_dwein_d[sset_eval], dataset)

    # Ssset
    tkfeats_train = tkfeats_d[sset_train]
    tkfeats_eval = tkfeats_d[sset_eval]
    tubes_dwein_train = tubes_dwein_d[sset_train]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_train = tubes_dgt_d[sset_train]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # Interacting with big
    assert cf['eval.full_tubes.enabled'], 'We train on them anyway'
    top_n_matches = cf['train.tubes.top_n_matches']
    stride = cf['train.tubes.stride']
    detect_mode = cf['inputs.tubes_dwein_feats.kind']
    man_feats_dwt = create_preextracted_feats_manager(
            cf, scaler, detect_mode)

    max_distance = cf['train.tubes.frame_dist']
    output_dims = cf['net.n_outputs']

    if detect_mode == 'fullframe':
        # fullframes
        labeled_frames: List[Frame_labeled] = \
            prepare_label_fullframes_for_training(
                tubes_dgt_train, dataset, stride, max_distance)
        # associate to extracted feats
        labeled_linked_frames: List[Frame_labeled] = \
            _link_lframes_to_exfeats(labeled_frames, man_feats_dwt)
        tdataset = TDataset_over_labeled_linked_frames(
                labeled_linked_frames, man_feats_dwt)
        assert output_dims == 10
        D_in = man_feats_dwt.fullframe_feats.shape[-1]
    elif detect_mode == 'roipooled':
        # roitubes
        keyframes_train = tkfeats_train['kf']
        keyframe_feats_train = tkfeats_train['X'].numpy()
        labeled_boxes: List[Box_labeled] = \
          prepare_label_roiboxes_for_training(
            tubes_dgt_train, dataset, stride, max_distance,
            tubes_dwein_train, keyframes_train, top_n_matches,
            cf['train.tubes.add_keyframes'])
        # associate roiboxes to extracted feats
        labeled_linked_boxes: List[Box_labeled_linked] = \
                _link_lboxes_to_exfeats(labeled_boxes, man_feats_dwt)
        tdataset = TDataset_over_labeled_linked_boxes(
            labeled_linked_boxes, man_feats_dwt, keyframe_feats_train)
        assert output_dims == 11
        D_in = man_feats_dwt.BIG.shape[-1]
    else:
        raise RuntimeError()

    # Model
    model = define_mlp_model(cf, D_in, output_dims)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])

    ckpt = Checkpointer(model, optimizer)

    # Restore previous run
    rundir = small.mkdir(out/'rundir')
    checkpoint_path = (Manager_checkpoint_name.find_last_checkpoint(rundir))
    if '--new' in add_args:
        Manager_checkpoint_name.rename_old_rundir(rundir)
        checkpoint_path = None
    start_epoch = (ckpt.restore_model_magic(checkpoint_path,
        cf['inputs.ckpt'], cf['train.start_epoch']))

    # Training
    n_epochs = cf['train.n_epochs']
    for i_epoch in range(start_epoch, n_epochs):
        log.info(f'Started epoch {i_epoch=}')
        model.train()
        l_avg = snippets.misc.Averager()
        avg_bs = snippets.misc.Averager()
        # Loader reproducible even if we restore
        rgen = np.random.default_rng(initial_seed+i_epoch)
        loader = _get_trainloader_rnd_sampler(tdataset,
                cf['train.batch_size'], rgen)
        for i_batch, (data_input) in enumerate(loader):
            # inputs converter
            (meta,) = data_input
            flat_labels = np.hstack([m['labels'] for m in meta])
            flat_feats = np.vstack([m['feats'] for m in meta])

            flat_labels_t = torch.from_numpy(flat_labels)
            flat_feats_t = torch.from_numpy(flat_feats)

            result = model(flat_feats_t)
            pred_train = result['x_final']
            loss = loss_fn(pred_train, flat_labels_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            avg_bs.update(len(flat_labels))
            l_avg.update(loss.item())

            if check_step(i_batch, cf['period.i_batch.loss_log']):
                Nb = len(loader)
                loss_str = (f'loss(all/last):{l_avg.avg:.4f}/{l_avg.last:.4f}')
                log.info(f'{i_epoch=}, {i_batch=}/{Nb}; {loss_str}')
        log.info('Epoch stats: avg_batchsize {}, loader_size {} '.format(
            avg_bs.avg, len(loader)))
        ckpt.save_epoch(rundir, i_epoch)
        if check_step(i_epoch, cf['period.i_epoch.loss_log']):
            log.info(f'Loss at {i_epoch=}: {l_avg.avg}')
        if check_step(i_epoch, cf['period.i_epoch.q_eval']):
            model.eval()
            kacc_train = quick_accuracy_over_kfeat(
                    tkfeats_train, model, True)*100
            kacc_eval = quick_accuracy_over_kfeat(
                    tkfeats_eval, model, True)*100
            model.train()
            log.info(f'Qperf at {i_epoch=}: '
                    f'{kacc_train=:.2f} {kacc_eval=:.2f}')
        if check_step(i_epoch, cf['period.i_epoch.full_eval']):
            model.eval()
            result = mlp_perf_kf_evaluate(
                    model, tkfeats_train, tkfeats_eval,
                    tubes_dwein_eval, tubes_dgt_eval,
                    dataset, output_dims)
            result_fulltube = mlp_perf_fulltube_evaluate(
                    model, man_feats_dwt,
                    tubes_dwein_eval, tubes_dwein_prov,
                    tubes_dgt_eval, dwti_to_label_eval,
                    dataset, output_dims,
                    cf['eval.full_tubes.detect_mode'],
                    cf['eval.full_tubes.nms'],
                    cf['eval.full_tubes.field_nms'],
                    cf['eval.full_tubes.field_det'])
            result.update(result_fulltube)
            model.train()
            log.info(f'Evalset perf at {i_epoch=}')
            mlp_perf_display(result, sset_eval)


def fancy_evaluate(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict)
    Ncfg_daly.set_defcfg_v2(cfg)
    Ncfg_kfeats.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    inputs:
        tubes_dwein: ~
        tubes_dwein_feats:
            fold: ~
            kind: !def ['roipooled', ['fullframe', 'roipooled']]
        ckpt: ~
    seed: 42
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
    data_scaler: !def ['no', ['keyframes', 'no']]
    net:
        kind: !def ['layer1', ['layer0', 'layer1']]
        layer1:
            H: 32
        ll_dropout: 0.5
        n_outputs: !def [~, [10, 11]]
    eval:
        full_tubes:
            enabled: True
            detect_mode: !def ['roipooled', ['fullframe', 'roipooled']]
            nms: 0.3
            field_nms: 'box_det_score'  # hscore
            field_det: 'box_det_score'  # hscore*frame_cls_score
    """)
    cf = cfg.parse()
    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    # Data
    # General DALY level preparation
    dataset = Ncfg_daly.get_dataset(cf)
    vgroup = Ncfg_daly.get_vids(cf, dataset)
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # keyframe feats
    tkfeats_d, scaler = Ncfg_kfeats.load_scale(cf, vgroup)
    # wein tubes
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided] = \
            small.load_pkl(cf['inputs.tubes_dwein'])

    # synthetic tube labels
    _, dwti_to_label_eval = qload_synthetic_tube_labels(
            tubes_dgt_d[sset_eval], tubes_dwein_d[sset_eval], dataset)
    # Ssset
    tkfeats_train = tkfeats_d[sset_train]
    tkfeats_eval = tkfeats_d[sset_eval]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # Interacting with big
    assert cf['eval.full_tubes.enabled'], 'We train on them anyway'
    detect_mode = cf['inputs.tubes_dwein_feats.kind']
    man_feats_dwt = create_preextracted_feats_manager(
            cf, scaler, detect_mode)
    output_dims = cf['net.n_outputs']

    if detect_mode == 'fullframe':
        assert output_dims == 10
        D_in = man_feats_dwt.fullframe_feats.shape[-1]
    elif detect_mode == 'roipooled':
        assert output_dims == 11
        D_in = man_feats_dwt.BIG.shape[-1]
    else:
        raise RuntimeError()

    # Model
    model = define_mlp_model(cf, D_in, output_dims)
    states = torch.load(cf['inputs.ckpt'])
    i_epoch = states['i_epoch']
    model.load_state_dict(states['model_sdict'])

    model.eval()
    result = mlp_perf_kf_evaluate(
            model, tkfeats_train, tkfeats_eval,
            tubes_dwein_eval, tubes_dgt_eval,
            dataset, output_dims)
    result_fulltube = mlp_perf_fulltube_evaluate(
            model, man_feats_dwt,
            tubes_dwein_eval, tubes_dwein_prov,
            tubes_dgt_eval, dwti_to_label_eval,
            dataset, output_dims,
            cf['eval.full_tubes.detect_mode'],
            cf['eval.full_tubes.nms'],
            cf['eval.full_tubes.field_nms'],
            cf['eval.full_tubes.field_det'])
    result.update(result_fulltube)
    model.train()
    log.info(f'Evalset perf at {i_epoch=}')
    mlp_perf_display(result, sset_eval)


def merge_evaluate(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig_v2(cfg_dict)
    Ncfg_daly.set_defcfg_v2(cfg)
    cfg.set_defaults_yaml("""
    inputs:
        tubes_dwein: ~
        keyframes:
            roi: ~
            full: ~
        tubes_dwein_feats:
            roi: ~
            full: ~
        ckpt:
            roi: ~
            full: ~
    net:
        kind: 'layer1'
        layer1:
            H: 32
        ll_dropout: 0.5
    seed: 42
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
    nms_scorefield: 'hscore'
    """)
    cf = cfg.parse()
    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
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

    # / keyframe feats
    def to_torch(kfeats):
        tkfeats = kfeats.copy()
        tkfeats['X'] = torch.from_numpy(tkfeats['X'])
        tkfeats['Y'] = torch.from_numpy(tkfeats['Y'])
        return tkfeats
    # // ROI
    keyframes_featfold = Path(cf['inputs.keyframes.roi'])
    keyframes = small.load_pkl(keyframes_featfold/'keyframes.pkl')
    outputs = small.load_pkl(
            keyframes_featfold/'dict_outputs.pkl')['roipooled']
    kfeats_eval_roi = to_torch(Ncfg_kfeats.split_off_D(
            outputs, keyframes, vgroup[sset_eval]))
    # // FULL
    keyframes_featfold = Path(cf['inputs.keyframes.full'])
    keyframes = small.load_pkl(keyframes_featfold/'keyframes.pkl')
    outputs = small.load_pkl(
            keyframes_featfold/'dict_outputs.pkl')['fullframe']
    kfeats_eval_full = to_torch(Ncfg_kfeats.split_off_D(
            outputs, keyframes, vgroup[sset_eval]))

    # synthetic tube labels
    _, dwti_to_label_eval = qload_synthetic_tube_labels(
            tubes_dgt_d[sset_eval], tubes_dwein_d[sset_eval], dataset)
    # Ssset
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # Interacting with big
    man_feats_dwt_roi = Manager_feats_tubes_dwein_roipooled(
            cf['inputs.tubes_dwein_feats.roi'], None)
    man_feats_dwt_full = Manager_feats_tubes_dwein_full(
            cf['inputs.tubes_dwein_feats.full'], None)
    D_in = man_feats_dwt_full.fullframe_feats.shape[-1]

    model_roi = define_mlp_model(cf, D_in, 11)
    states = torch.load(cf['inputs.ckpt.roi'])
    i_epoch = states['i_epoch']
    model_roi.load_state_dict(states['model_sdict'])
    model_roi.eval()

    model_full = define_mlp_model(cf, D_in, 10)
    states = torch.load(cf['inputs.ckpt.full'])
    model_full.load_state_dict(states['model_sdict'])
    model_full.eval()

    # Actual evaluation
    iou_thresholds = [.3, .5, .7]
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(tubes_dgt_eval)

    # ROI evaluations
    tube_softmaxes_eval: Dict[I_dwein, np.ndarray] = {}
    with torch.no_grad():
        for dwti in tubes_dwein_eval.keys():
            dwti_feats = man_feats_dwt_roi.get_all_tube_feats(dwti)
            preds_softmax = model_roi(dwti_feats)['x_final']
            tube_softmaxes_eval[dwti] = preds_softmax.numpy()
    tube_softmaxes_eval_nobg = {k: v[:, :-1]
            for k, v in tube_softmaxes_eval.items()}

    # FULL evaluations
    connections_f = man_feats_dwt_full.connections_f
    fullframe_feats = man_feats_dwt_full.fullframe_feats
    with torch.no_grad():
        t_fullframe_feats = torch.from_numpy(fullframe_feats)
        x_final = model_full(t_fullframe_feats)['x_final'].numpy()
    # Aggregate frame scores
    frame_scores: Dict[Tuple[Vid_daly, int], np.ndarray] = {}
    for cons, outputs_ in zip(connections_f.values(), x_final):
        vid = cons['vid']
        frame_ind = cons['frame_ind']
        frame_scores[(vid, frame_ind)] = outputs_

    # Universal detector experiments
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
    # av_stubes = assign_scorefield(av_stubes, 'box_det_score')  # roi, 72.73
    # av_stubes = assign_scorefield(av_stubes, 'hscore*frame_cls_score')  # full, 71.35
    # av_stubes = assign_scorefield(av_stubes, 'mean(box_det_score,frame_cls_score)')  # 72.54
    av_stubes = assign_scorefield(av_stubes, 'mean(box_det_score, hscore*frame_cls_score)')  # 74.72

    df_ap_full = compute_ap_for_avtubes_as_df(
        av_gt_tubes, av_stubes, iou_thresholds, False, False)
    log.info(df_ap_full*100)
    apline = '/'.join((df_ap_full*100).round(2).loc['all'].values.astype(str))
    log.info('AP357: {}'.format(apline))
