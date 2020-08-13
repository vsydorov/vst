import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.preprocessing import (StandardScaler)
from sklearn.metrics import (accuracy_score,)
from typing import (  # NOQA
        Dict, Any, Optional, List, cast, Tuple, TypedDict, Iterable)

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from vsydorov_tools import small

from thes.data.dataset.daly import (
    Ncfg_daly, load_gt_and_wein_tubes,
    group_dwein_frames_wrt_kf_distance)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    AV_dict, Box_connections_dwti)
from thes.data.tubes.routines import (
    qload_synthetic_tube_labels)
from thes.data.tubes.routines import (
    temporal_ious_where_positive, spatial_tube_iou_v3,
    get_dwein_overlaps_per_dgt, select_fg_bg_tubes)
from thes.evaluation.meta import (
    keyframe_cls_scores, cheating_tube_scoring, quick_tube_eval)
from thes.tools import snippets
from thes.pytorch import (
    sequence_batch_collate_v2, NumpyRandomSampler)
from thes.training import (
    Manager_checkpoint_name)

log = logging.getLogger(__name__)


# Envelopes for keyframe evaluation


class E_kfeats(TypedDict):
    X: np.ndarray
    Y: np.ndarray
    kf: dict


class E_tkfeats(TypedDict):
    X: torch.Tensor
    Y: torch.Tensor
    kf: dict


class Ncfg_kfeats:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        inputs:
            keyframes:
                fold: [~, str]
                featname: [~, ~]
        """)

    def set_defcfg_v2(cfg):
        cfg.set_defaults_yaml("""
        inputs:
            keyframes:
                fold: ~
                featname: ~
        """)

    @staticmethod
    def split_off(X, linked_vids, good_vids):
        if isinstance(X, np.ndarray):
            isin = np.in1d(linked_vids, good_vids)
            result = X[isin]
        elif isinstance(X, list):
            result = [x for x, v in zip(X, linked_vids) if v in good_vids]
        else:
            raise RuntimeError()
        return result

    @staticmethod
    def split_off_D(outputs, keyframes, vids) -> E_kfeats:
        global_kf_vids = [kf['vid'] for kf in keyframes]
        X = Ncfg_kfeats.split_off(outputs, global_kf_vids, vids)
        kf = Ncfg_kfeats.split_off(keyframes, global_kf_vids, vids)
        Y = np.array([kf['action_id'] for kf in kf])
        d: E_kfeats = {'X': X, 'kf': kf, 'Y': Y}
        return d

    @staticmethod
    def load_scale(cf, vgroup):
        # Produce keyframe datasets realquick
        featname = cf['inputs.keyframes.featname']
        keyframes_featfold = Path(cf['inputs.keyframes.fold'])
        keyframes = small.load_pkl(keyframes_featfold/'keyframes.pkl')
        outputs = small.load_pkl(
                keyframes_featfold/'dict_outputs.pkl')[featname]

        kfeats_d: Dict[str, E_kfeats] = {}
        for sset, vids in vgroup.items():
            kfeats_d[sset] = Ncfg_kfeats.split_off_D(
                    outputs, keyframes, vids)
        # Scale
        if cf['data_scaler'] == 'keyframes':
            scaler = Ncfg_kfeats.fitscale_kfeats(kfeats_d)
        elif cf['data_scaler'] == 'no':
            scaler = None
        else:
            raise RuntimeError()

        # To torch
        tkfeats_d: Dict[str, E_tkfeats] = Ncfg_kfeats.to_torch(kfeats_d)
        return tkfeats_d, scaler

    @staticmethod
    def fitscale_kfeats(kfeats_d):
        # Optional standard scaling on trianval
        scaler = StandardScaler()
        scaler.fit(kfeats_d['trainval']['X'])
        for sset, kfeats in kfeats_d.items():
            kfeats['X'] = scaler.transform(kfeats['X'])
        return scaler

    @staticmethod
    def to_torch(kfeats_d: Dict[str, E_kfeats]) -> Dict[str, E_tkfeats]:
        tkfeats_d = {}
        for sset, kfeats in kfeats_d.items():
            tkfeats = kfeats.copy()
            tkfeats['X'] = torch.from_numpy(tkfeats['X'])
            tkfeats['Y'] = torch.from_numpy(tkfeats['Y'])
            tkfeats_d[sset] = tkfeats
        return tkfeats_d


class Net_mlp_zerolayer(nn.Module):
    """
    This one will match outputs with the last layer finetuning in
    SoftMax-derived convnent
    """
    def __init__(
            self, D_in, D_out,
            dropout_rate, debug_outputs=False):
        super().__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.rt_projection = nn.Linear(D_in, D_out, bias=True)
        self.rt_act = nn.Softmax(dim=-1)

    def init_weights(self, init_std, seed):
        """Will match initialization of ll finetuned net"""
        ll_generator = torch.Generator()
        ll_generator = ll_generator.manual_seed(67280421310679+seed)

        self.rt_projection.weight.data.normal_(
                mean=0.0, std=init_std, generator=ll_generator)
        self.rt_projection.bias.data.zero_()

    def forward(self, x, debug_outputs=False):
        result = {}
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if debug_outputs:
            result['x_after_dropout'] = x
        x = self.rt_projection(x)
        if debug_outputs:
            result['x_after_proj'] = x
        if not self.training:
            x = self.rt_act(x)
        result['x_final'] = x
        return result


class Net_mlp_onelayer(nn.Module):
    def __init__(self, D_in, D_out, H, dropout_rate):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.rt_act = nn.Softmax(dim=-1)

    def init_weights(self, init_std, seed):
        ll_generator = torch.Generator()
        ll_generator = ll_generator.manual_seed(67280421310679+seed)
        for l in [self.linear1, self.linear2]:
            l.weight.data.normal_(
                    mean=0.0, std=init_std, generator=ll_generator)
            l.bias.data.zero_()

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.linear2(x)
        if not self.training:
            x = self.rt_act(x)
        result = {}
        result['x_final'] = x
        return result


class Data_access_big(object):
    def __init__(self, BIG, dwti_to_inds_big, scaler):
        self.BIG = BIG
        self.dwti_to_inds_big = dwti_to_inds_big
        self.scaler = scaler

    def get(self, model, dwti):
        inds_big = self.dwti_to_inds_big[dwti]
        feats = self.BIG[inds_big]
        feats = feats.astype(np.float32)
        if self.scaler is not None:
            feats = self.scaler.transform(feats)
        feats = torch.from_numpy(feats)
        return feats

    def produce_loader(self, rgen, dwti_to_label_train,
            tubes_per_batch, frames_per_tube):
        batches = Data_access_big._quick_shuffle_batches(
            self.dwti_to_inds_big, rgen, dwti_to_label_train,
            tubes_per_batch, frames_per_tube)
        loader = Data_access_big._quick_dataloader(self.BIG, batches, self.scaler)
        return loader

    @staticmethod
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

    @staticmethod
    def _quick_dataloader(BIG, batches, scaler):
        # // torch dataset
        td_h5 = Data_access_big.TD_thin_over_BIG(
                BIG, batches, scaler)
        loader = torch.utils.data.DataLoader(td_h5,
            batch_size=None, num_workers=0,
            collate_fn=None)
        return loader

    @staticmethod
    def get_dwti_big_mapping(
            connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti],
            box_inds2
            ) -> Dict[I_dwein, np.ndarray]:
        dwti_h5_inds: Dict[I_dwein, np.ndarray] = {}
        for con, bi2 in zip(connections_f.values(), box_inds2):
            bi_range = np.arange(bi2[0], bi2[1])
            for dwti, bi in zip(con['dwti_sources'], bi_range):
                dwti_h5_inds.setdefault(dwti, []).append(bi)
        dwti_h5_inds = {k: np.array(sorted(v))
                for k, v in dwti_h5_inds.items()}
        return dwti_h5_inds

    @staticmethod
    def load_big_features(tubes_featfold):
        """Load whole npy file"""
        tubes_featfold = Path(tubes_featfold)
        # Load connections, arrange back into dwt_index based structure
        connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti] = \
                small.load_pkl(tubes_featfold/'connections_f.pkl')
        # (N_frames, 2) ndarray of BIG indices
        box_inds2 = small.load_pkl(tubes_featfold/'box_inds2.pkl')
        # Mapping dwti -> 1D ndarray of BIG indices
        dwti_to_inds_big = Data_access_big.get_dwti_big_mapping(
                connections_f, box_inds2)
        # Features
        with small.QTimer('big numpy load'):
            BIG = np.load(str(tubes_featfold/"feats.npy"))
        return BIG, dwti_to_inds_big


class TDataset_over_bilinked_boxes(torch.utils.data.Dataset):
    def __init__(self, frame_groups, BIG, scaler, KFX):
        self.frame_groups = frame_groups
        self.keys_vf = list(frame_groups.keys())
        self.BIG = BIG
        self.scaler = scaler
        self.KFX = KFX

    def __getitem__(self, index):
        key_vf = self.keys_vf[index]
        boxes = self.frame_groups[key_vf]

        # Features
        bi_presence = np.array(['bi' in b for b in boxes])
        if bi_presence.all():
            bis = np.array([b['bi'] for b in boxes])
            feats = self.BIG[bis]
            feats = feats.astype(np.float32)
        else:
            feats = np.zeros((len(boxes),
                self.BIG.shape[-1]), dtype=np.float32)
            for i, box in enumerate(boxes):
                if 'bi' in box:
                    feats[i] = self.BIG[box['bi']]
                elif 'kfi' in box:
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


class Manager_loader_big_v2(object):
    def __init__(self, tubes_featfold, scaler,
            stride, top_n_matches,
            dataset, tubes_dwein_train, tubes_dgt_train, tkfeats_train):
        self.dataset = dataset
        self.scaler = scaler
        self.tkfeats_train = tkfeats_train

        tubes_featfold = Path(tubes_featfold)
        connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti] = \
                small.load_pkl(tubes_featfold/'connections_f.pkl')
        box_inds2 = small.load_pkl(tubes_featfold/'box_inds2.pkl')
        # DWTI -> Frame -> bi (big index)
        with small.QTimer('Creating dwti -> big index structure'):
            dwti_f_bi = {}
            for con, bi2 in zip(connections_f.values(), box_inds2):
                bi_range = np.arange(bi2[0], bi2[1])
                for dwti, bi in zip(con['dwti_sources'], bi_range):
                    dwti_f_bi.setdefault(dwti, {})[con['frame_ind']] = bi
        # Features
        with small.QTimer('big numpy load'):
            BIG = np.load(str(tubes_featfold/"feats.npy"))
        self.dwti_f_bi = dwti_f_bi
        self.BIG = BIG
        # from thes.dataset.daly import ()
        tubes_dgt = tubes_dgt_train
        tubes_dwein = tubes_dwein_train

        # // Associate tubes
        matched_dwts: Dict[I_dgt, Dict[I_dwein, float]] = \
                get_dwein_overlaps_per_dgt(tubes_dgt, tubes_dwein)
        fg_meta, bg_meta = select_fg_bg_tubes(matched_dwts, top_n_matches)
        log.info('Selected {} FG and {} BG tubes from a total of {}'.format(
            len(fg_meta), len(bg_meta), len(tubes_dwein)))
        # Merge fg/bg
        tube_metas = {}
        tube_metas.update(fg_meta)
        tube_metas.update(bg_meta)
        # Break into frames, sort by distance
        self.dist_boxes_train = group_dwein_frames_wrt_kf_distance(
            dataset, stride, tubes_dwein_train, tube_metas)

    def get(self, model, dwti):
        inds_big = np.array(list(self.dwti_f_bi[dwti].values()))
        feats = self.BIG[inds_big]
        feats = feats.astype(np.float32)
        if self.scaler is not None:
            feats = self.scaler.transform(feats)
        feats = torch.from_numpy(feats)
        return feats

    def get_train_loader(
            self, batch_size, rgen, max_distance, add_keyframes=True):
        labeled_boxes = []
        for i, boxes in self.dist_boxes_train.items():
            if i > max_distance:
                break
            for box in boxes:
                (vid, bunch_id, tube_id) = box['dwti']
                bi = self.dwti_f_bi[box['dwti']][box['frame_ind']]
                if box['kind'] == 'fg':
                    (vid, action_name, ins_id) = box['dgti']
                    label = self.dataset.action_names.index(action_name)
                else:
                    label = len(self.dataset.action_names)
                lbox = {
                    'vid': vid,
                    'frame_ind': box['frame_ind'],
                    'bi': bi,
                    'box': box['box'],
                    'label': label}
                labeled_boxes.append(lbox)
        if add_keyframes:
            for kfi, kf in enumerate(self.tkfeats_train['kf']):
                lbox = {
                    'vid': kf['vid'],
                    'frame_ind': kf['frame0'],
                    'kfi': kfi,
                    'box': kf['bbox'],
                    'label': kf['action_id']}
                labeled_boxes.append(lbox)

        # Group into frame_groups
        frame_groups: Dict[Tuple[Vid_daly, int], Dict] = {}
        for lbox in labeled_boxes:
            vid = lbox['vid']
            frame_ind = lbox['frame_ind']
            frame_groups.setdefault((vid, frame_ind), []).append(lbox)

        # Loader
        NUM_WORKERS = 0
        td = TDataset_over_bilinked_boxes(
                frame_groups, self.BIG, self.scaler,
                self.tkfeats_train['X'].numpy())
        sampler = NumpyRandomSampler(td, rgen)
        loader = torch.utils.data.DataLoader(td,
            batch_size=batch_size, num_workers=NUM_WORKERS,
            sampler=sampler, shuffle=False,
            pin_memory=True,
            collate_fn=sequence_batch_collate_v2)
        return loader


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

    def __init__(self, tkfeats_train, initial_seed, TRAIN_BATCH_SIZE):
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
        self.period_ibatch_loss_log = '0::10'

    def train_step(self, model, optimizer, loss_fn, i_epoch):
        for i_batch, (data_input) in enumerate(self.train_krgb_loader):
            feats, labels, keyframes = data_input
            result = model(feats)
            pred_train = result['x_final']
            loss = loss_fn(pred_train, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if snippets.check_step_sslice(
                    i_batch, self.period_ibatch_loss_log):
                Nb = len(self.train_krgb_loader)
                log.info(f'{i_epoch=}, {i_batch=}/{Nb}: loss {loss.item()}')
        return loss


def define_mlp_model(cf, D_in, D_out):
    dropout_rate = cf['net.ll_dropout']
    if cf['net.kind'] == 'layer0':
        model = Net_mlp_zerolayer(
            D_in, D_out, dropout_rate)
    elif cf['net.kind'] == 'layer1':
        model = Net_mlp_onelayer(
            D_in, D_out, cf['net.layer1.H'], dropout_rate)
    return model


def _tubefeats_display_evalresults(result, sset_eval):
    acc_flattube_synt = result['acc_flattube_synt']
    df_ap_full = (result['df_ap_full']*100).round(2)
    apline = '/'.join(df_ap_full.loc['all'].values.astype(str))
    log.info('Tube evaluation results ({}): '.format(sset_eval))
    log.debug('AP (full tube tubes):\n{}'.format(df_ap_full))
    log.info(' '.join(
        ('Flattube synthetic acc: {:.2f};'.format(acc_flattube_synt*100),
        'Full tube AP357: {}'.format(apline))))


def _mlp_set_defaults(cfg):
    cfg.set_deftype("""
    split_assignment: ['train/val', ['train/val', 'trainval/test']]
    seed: [42, int]
    inputs:
        tubes_dwein: [~, str]
        big:
            fold: [~, ~]
    data_scaler: ['keyframes', ['keyframes', 'no']]
    net:
        kind: ['layer1', ['layer0', 'layer1']]
        layer1:
            H: [32, int]
        ll_dropout: [0.5, float]
    """)


# Inference


def _predict_softmaxes_for_dwein_tubes_in_da_big(
        model,
        da_big: Data_access_big,
        dwtis: Iterable[I_dwein]
        ) -> Dict[I_dwein, np.ndarray]:
    """
    Predict softmaxes for dwein tubes in Data_access_big
    """
    tube_sofmaxes = {}
    model.eval()
    with torch.no_grad():
        for dwti in dwtis:
            preds_softmax = model(da_big.get(model, dwti))['x_final']
            tube_sofmaxes[dwti] = preds_softmax.numpy()
    return tube_sofmaxes


# Evaluation


def _quick_accuracy_over_kfeat(
        tkfeats: E_tkfeats, model, cutoff_last_dim):
    """
    Quick evaluation of E_tkfeat accuracy via torch methods
    """
    def _qacc(pred, Y):
        return pred.argmax(1).eq(Y).sum().item()/len(Y)
    with torch.no_grad():
        pred_eval = model(tkfeats['X'])['x_final']
    if cutoff_last_dim:
        pred_eval = pred_eval[:, :-1]
    acc = _qacc(pred_eval, tkfeats['Y'])
    return acc


def _compute_flattube_syntlabel_acc(
        tube_softmaxes: Dict[I_dwein, np.ndarray],
        dwti_to_label: Dict[I_dwein, int]) -> float:
    """
    Compute synthetic per-frame accuracy over dwein tubes
    """
    # Assert presence of background cls
    x = next(iter(tube_softmaxes.values()))
    assert x.shape[-1] == 11

    flat_sm_ = []
    flat_label_ = []
    for dwti, label in dwti_to_label.items():
        softmaxes = tube_softmaxes[dwti]
        flat_sm_.append(softmaxes)
        flat_label_.append(np.repeat(label, len(softmaxes)))
    flat_sm = np.vstack(flat_sm_)
    flat_label = np.hstack(flat_label_)
    return accuracy_score(flat_label, flat_sm.argmax(axis=1))


def _quick_assign_scores_to_dwein_tubes(
        tubes_dwein: Dict[I_dwein, T_dwein],
        tube_softmaxes: Dict[I_dwein, np.ndarray],
        dataset: Dataset_daly_ocv
        ) -> AV_dict[T_dwein_scored]:
    """
    Softmaxes should correspond to dataset.action_names
    """
    # Assert absence of background cls
    x = next(iter(tube_softmaxes.values()))
    assert x.shape[-1] == 10

    av_stubes: AV_dict[T_dwein_scored] = {}
    for dwt_index, tube in tubes_dwein.items():
        softmaxes = tube_softmaxes[dwt_index]
        scores = softmaxes.mean(axis=0)
        (vid, bunch_id, tube_id) = dwt_index
        for action_name, score in zip(dataset.action_names, scores):
            stube = cast(T_dwein_scored, tube.copy())
            stube['score'] = score
            stube = cast(T_dwein_scored, stube)
            (av_stubes
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes


def _tubefeats_trainset_perf(
        model,
        da_big: Data_access_big,
        dwti_to_label_train: Dict[I_dwein, int]):
    with small.QTimer() as t:
        tube_sofmaxes_train: Dict[I_dwein, np.ndarray] = \
            _predict_softmaxes_for_dwein_tubes_in_da_big(
                model, da_big, dwti_to_label_train.keys())
        acc_full_train = _compute_flattube_syntlabel_acc(
                tube_sofmaxes_train, dwti_to_label_train)
    tsec = t.time
    return acc_full_train, tsec


def _tubefeats_evalset_perf(
        model,
        da_big: Data_access_big,
        dwti_to_label_eval: Dict[I_dwein, int],
        dataset: Dataset_daly_ocv,
        tubes_dwein_eval: Dict[I_dwein, T_dwein],
        tubes_dgt_eval: Dict[I_dgt, T_dgt]
            ) -> Dict:
    tube_sofmaxes_eval: Dict[I_dwein, np.ndarray] = \
        _predict_softmaxes_for_dwein_tubes_in_da_big(
            model, da_big, tubes_dwein_eval.keys())
    # Flat accuracy: only dwto_to_label_eval tubes and includes
    # background cls, should be over 11 classes
    acc_flattube_synt = _compute_flattube_syntlabel_acc(
            tube_sofmaxes_eval, dwti_to_label_eval)
    # MAP: all tubes in tubes_dwein_eval, excludes background (last cls)
    tube_sofmaxes_eval_nobg = {k: v[:, :-1]
            for k, v in tube_sofmaxes_eval.items()}
    av_stubes_eval: AV_dict[T_dwein_scored] = \
        _quick_assign_scores_to_dwein_tubes(
            tubes_dwein_eval, tube_sofmaxes_eval_nobg, dataset)
    df_ap_full, _ = quick_tube_eval(av_stubes_eval, tubes_dgt_eval)
    result = {'acc_flattube_synt': acc_flattube_synt,
            'df_ap_full': df_ap_full}
    return result


# kffeats_train_mlp


def _kffeats_train_mlp_single_run(
        cf, initial_seed, da_big,
        tkfeats_train: E_tkfeats,
        tkfeats_eval: E_tkfeats,
        tubes_dwein_eval, tubes_dgt_eval,
        dataset, sset_eval):

    torch.manual_seed(initial_seed)
    period_log = cf['train.period.log']
    period_eval_evalset = cf['train.period.eval']

    n_epochs = cf['train.n_epochs']
    D_in = tkfeats_train['X'].shape[-1]
    model = define_mlp_model(cf, D_in, 10)
    model.init_weights(0.01, initial_seed)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])

    trsampler: Train_sampler
    if cf['train.mode'] == 'full':
        trsampler = Full_train_sampler(tkfeats_train)
    elif cf['train.mode'] == 'partial':
        train_batch_size = cf['train.partial.train_batch_size']
        trsampler = Partial_Train_sampler(
                tkfeats_train, initial_seed, train_batch_size)
    else:
        raise NotImplementedError()

    def evaluate(i_epoch):
        model.eval()
        result = _kffeats_eval(
                cf, model, da_big, tkfeats_train, tkfeats_eval,
                tubes_dwein_eval, tubes_dgt_eval, dataset)
        model.train()
        log.info(f'Perf at {i_epoch=}')
        _kffeats_display_evalresults(result, sset_eval)
        return result

    for i_epoch in range(n_epochs):
        loss = trsampler.train_step(model, optimizer, loss_fn, i_epoch)
        if snippets.check_step_sslice(i_epoch, period_log):
            model.eval()
            kacc_train = _quick_accuracy_over_kfeat(tkfeats_train, model, False)
            kacc_eval = _quick_accuracy_over_kfeat(tkfeats_eval, model, False)
            model.train()
            log.info(f'{i_epoch}: {loss.item():.4f} '
                    f'K.Acc.Train: {kacc_train*100:.2f}; '
                    f'K.Acc.Eval: {kacc_eval*100:.2f}')
        if snippets.check_step_sslice(i_epoch, period_eval_evalset):
            result = evaluate(i_epoch)
    result = evaluate(i_epoch)
    return result


def _kffeats_eval(
        cf, model,
        da_big: Data_access_big,
        tkfeats_train: E_tkfeats,
        tkfeats_eval: E_tkfeats,
        tubes_dwein_eval: Dict[I_dwein, T_dwein],
        tubes_dgt_eval: Dict[I_dgt, T_dgt],
        dataset: Dataset_daly_ocv) -> Dict:
    """
    Evaluation of keyframe feats includes a lot of stuff:
     + kacc_train, kacc_eval - runtime E_tkfeats perf
     + kf_acc, kf_roc_auc - perf for classification of keyframes
     + df_ap_cheat - perf for cheating evaluation
     + acc_flattube_synt - synthetic per-frame accuracy of dwein tubes
     + df_ap_full - perf for full evaluation
    """
    assert not model.training, 'Wrong results in train mode'
    # Already softmax when in eval mode
    with torch.no_grad():
        Y_test_softmax = model(tkfeats_eval['X'])['x_final'].numpy()
    kacc_train = _quick_accuracy_over_kfeat(tkfeats_train, model, False)
    kacc_eval = _quick_accuracy_over_kfeat(tkfeats_eval, model, False)
    kf_acc, kf_roc_auc = keyframe_cls_scores(
            Y_test_softmax, tkfeats_eval['Y'])
    av_stubes_cheat: AV_dict[T_dwein_scored] = cheating_tube_scoring(
        Y_test_softmax, tkfeats_eval['kf'], tubes_dwein_eval, dataset)
    df_ap_cheat, df_recall_cheat = \
        quick_tube_eval(av_stubes_cheat, tubes_dgt_eval)
    result = {
        'kacc_train': kacc_train,
        'kacc_eval': kacc_eval,
        'kf_acc': kf_acc,
        'kf_roc_auc': kf_roc_auc,
        'df_recall_cheat': df_recall_cheat,
        'df_ap_cheat': df_ap_cheat}

    # // Evaluation of full wein-tubes with the trained model
    if cf['eval.full_tubes']:
        # here we are mirroring '_tubefeats_evalset_perf'
        _, dwti_to_label_eval = qload_synthetic_tube_labels(
            tubes_dgt_eval, tubes_dwein_eval, dataset)
        tube_sofmaxes_eval: Dict[I_dwein, np.ndarray] = \
            _predict_softmaxes_for_dwein_tubes_in_da_big(
                model, da_big, tubes_dwein_eval.keys())
        # When computing synt accuracy add dummy background prediction
        tube_sofmaxes_eval_0bg = {k: np.pad(v, ((0, 0), (0, 1)))
                for k, v in tube_sofmaxes_eval.items()}
        acc_flattube_synt = _compute_flattube_syntlabel_acc(
                tube_sofmaxes_eval_0bg, dwti_to_label_eval)
        # MAP, all tubes
        av_stubes_eval: AV_dict[T_dwein_scored] = \
            _quick_assign_scores_to_dwein_tubes(
                tubes_dwein_eval, tube_sofmaxes_eval, dataset)
        df_ap_full, _ = quick_tube_eval(av_stubes_eval, tubes_dgt_eval)
        result['acc_flattube_synt'] = acc_flattube_synt
        result['df_ap_full'] = df_ap_full
    return result


def _kffeats_display_evalresults(result, sset_eval):
    df_recall_cheat = (result['df_recall_cheat']*100).round(2)
    df_ap_cheat = (result['df_ap_cheat']*100).round(2)
    cheating_apline = '/'.join(df_ap_cheat.loc['all'].values.astype(str))
    log.info('Keyframe classification results ({}): '.format(sset_eval))
    log.debug('Recall (cheating tubes) \n{}'.format(df_recall_cheat))
    log.debug('AP (cheating tubes):\n{}'.format(df_ap_cheat))
    log.info(' '.join((
        'kacc_train: {:.2f};'.format(result['kacc_train']*100),
        'kacc_eval: {:.2f};'.format(result['kacc_eval']*100))))
    log.info(' '.join((
        'K.Acc: {:.2f};'.format(result['kf_acc']*100),
        'RAUC {:.2f}'.format(result['kf_roc_auc']*100))))
    log.info('Cheat AP357: {}'.format(cheating_apline))

    df_ap_full = result.get('df_ap_full')
    if df_ap_full is not None:
        acc_flattube_synt = result['acc_flattube_synt']
        df_ap_full = (df_ap_full*100).round(2)
        apline = '/'.join(df_ap_full.loc['all'].values.astype(str))
        log.debug('AP357 (full tube tubes):\n{}'.format(df_ap_full))
        log.info('Flattube synthetic acc: {:.2f}'.format(acc_flattube_synt*100))
        log.info('Full tube AP357: {}'.format(apline))


# tubefeats_train_mlp


def _tubefeats_train_mlp_single_run_pretraining(
        cf, model, loss_fn,
        tkfeats_train: E_tkfeats,
        tkfeats_eval: E_tkfeats):
    pretrain_lr = cf['kf_pretrain.train.lr']
    pretrain_weight_decay = cf['kf_pretrain.train.weight_decay']
    pretrain_optimizer = torch.optim.AdamW(model.parameters(),
            lr=pretrain_lr, weight_decay=pretrain_weight_decay)
    pretrain_n_epochs = cf['kf_pretrain.train.n_epochs']
    pretrain_period_log = cf['kf_pretrain.period.log']
    model.train()
    for epoch in range(pretrain_n_epochs):
        out_train = model(tkfeats_train['X'])['x_final']
        loss = loss_fn(out_train, tkfeats_train['Y'])
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()
        if snippets.check_step_sslice(epoch, pretrain_period_log):
            model.eval()
            kacc_train = _quick_accuracy_over_kfeat(
                    tkfeats_train, model, True)
            kacc_eval = _quick_accuracy_over_kfeat(
                    tkfeats_eval, model, True)
            model.train()
            log.info(f'{epoch}: {loss.item():.4f} '
                    f'K.Acc.Train: {kacc_train*100:.2f}; '
                    f'K.Acc.Eval: {kacc_eval*100:.2f}')


def _tubefeats_train_mlp_single_run_training(
        cf, model, loss_fn, rgen,
        da_big: Data_access_big,
        tkfeats_train: E_tkfeats,
        tkfeats_eval: E_tkfeats,
        dwti_to_label_train: Dict[I_dwein, int],
        dwti_to_label_eval: Dict[I_dwein, int],
        tubes_dwein_eval: Dict[I_dwein, T_dwein],
        tubes_dgt_eval: Dict[I_dgt, T_dgt],
        sset_eval,
        dataset: Dataset_daly_ocv
        ):
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])
    period_log = cf['train.period.log']
    period_eval_trainset = cf['train.period.eval.trainset']
    period_eval_evalset = cf['train.period.eval.evalset']
    n_epochs = cf['train.n_epochs']
    tubes_per_batch = cf['train.tubes_per_batch']
    frames_per_tube = cf['train.frames_per_tube']
    model.train()
    for epoch in range(n_epochs):
        loader = da_big.produce_loader(rgen, dwti_to_label_train,
            tubes_per_batch, frames_per_tube)
        for i_batch, (feats, labels) in enumerate(loader):
            pred_train = model(feats)['x_final']
            loss = loss_fn(pred_train, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if snippets.check_step_sslice(epoch, period_log):
            model.eval()
            kacc_train = _quick_accuracy_over_kfeat(
                    tkfeats_train, model, True)*100
            kacc_eval = _quick_accuracy_over_kfeat(
                    tkfeats_eval, model, True)*100
            model.train()
            log.info(f'{epoch}: {loss.item()} '
                    f'{kacc_train=:.2f} {kacc_eval=:.2f}')
        if snippets.check_step_sslice(epoch, period_eval_trainset):
            model.eval()
            acc, tsec = _tubefeats_trainset_perf(
                    model, da_big, dwti_to_label_train)
            model.train()
            log.info('Train full keyframe acc: '
                'Train {:.2f}, took {:.2f} sec'.format(acc*100, tsec))
        if snippets.check_step_sslice(epoch, period_eval_evalset):
            model.eval()
            evalset_result = _tubefeats_evalset_perf(
                    model, da_big, dwti_to_label_eval,
                    dataset, tubes_dwein_eval, tubes_dgt_eval)
            model.train()
            log.info(f'Evalset perf at {epoch=}')
            _tubefeats_display_evalresults(evalset_result, sset_eval)


def _tubefeats_train_mlp_single_run(
        cf, initial_seed,
        da_big: Data_access_big,
        tkfeats_train: E_tkfeats,
        tkfeats_eval: E_tkfeats,
        dwti_to_label_train: Dict[I_dwein, int],
        dwti_to_label_eval: Dict[I_dwein, int],
        tubes_dwein_eval: Dict[I_dwein, T_dwein],
        tubes_dgt_eval: Dict[I_dgt, T_dgt],
        sset_eval,
        dataset: Dataset_daly_ocv
        ):

    torch.manual_seed(initial_seed)
    rgen = np.random.default_rng(initial_seed)

    D_in = da_big.BIG.shape[-1]
    model = define_mlp_model(cf, D_in, 11)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # pretraining
    if cf['kf_pretrain.enabled']:
        _tubefeats_train_mlp_single_run_pretraining(
            cf, model, loss_fn, tkfeats_train, tkfeats_eval)
    _tubefeats_train_mlp_single_run_training(
        cf, model, loss_fn, rgen, da_big,
        tkfeats_train, tkfeats_eval,
        dwti_to_label_train, dwti_to_label_eval,
        tubes_dwein_eval, tubes_dgt_eval,
        sset_eval, dataset)

    # proper map evaluation
    model.eval()
    result = _tubefeats_evalset_perf(
        model, da_big, dwti_to_label_eval, dataset,
        tubes_dwein_eval, tubes_dgt_eval)
    _tubefeats_display_evalresults(result, sset_eval)
    return result


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
        log.info(f'Saved model. Epoch {i_epoch}')
        log.debug('Saved to {}. Took {:.2f}s'.format(
            save_filepath, qtr.time))

    def restore_model_magic(
            self, checkpoint_path, starting_model=None, training_start_epoch=0):
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
            log.info(('Starting new training, '
                'empty model, at epoch {}').format(start_epoch))
        return start_epoch

# Experiments


def kffeats_train_mlp(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_kfeats.set_defcfg(cfg)
    _mlp_set_defaults(cfg)
    cfg.set_deftype("""
    train:
        mode: ['full', ['full', 'partial']]
        partial:
            train_batch_size: [32, int]
    """)
    cfg.set_defaults("""
    train:
        lr: 1.0e-5
        weight_decay: 5.0e-2
        n_epochs: 2001
        period:
            log: '0::500'
            eval: '0::500'
    eval:
        full_tubes: False
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
    if cf['eval.full_tubes']:
        BIG, dwti_to_inds_big = Data_access_big.load_big_features(
                cf['inputs.big.fold'])
        da_big = Data_access_big(BIG, dwti_to_inds_big, scaler)
    else:
        da_big = None

    # / Torch section
    tkfeats_train = tkfeats_d[sset_train]
    tkfeats_eval = tkfeats_d[sset_eval]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    log.info(f'Train/eval splits: {sset_train} {sset_eval=}')

    def experiment(i):
        log.info(f'Experiment {i}')
        result = _kffeats_train_mlp_single_run(
            cf, initial_seed+i, da_big,
            tkfeats_train, tkfeats_eval,
            tubes_dwein_eval, tubes_dgt_eval,
            dataset, sset_eval)
        return result

    if n_trials == 1:
        experiment(0)
        return

    isaver = snippets.Isaver_simple(
            small.mkdir(out/'isaver_ntrials'),
            range(n_trials), experiment)
    trial_results = isaver.run()

    df_keys = ['df_recall_cheat', 'df_ap_cheat']
    scalar_keys = ['kacc_train', 'kacc_eval', 'kf_acc', 'kf_roc_auc']
    if 'df_ap_full' in trial_results[0]:
        df_keys.append('df_ap_full')
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
    _kffeats_display_evalresults(avg_result, sset_eval)


def tubefeats_train_mlp(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_kfeats.set_defcfg(cfg)
    _mlp_set_defaults(cfg)
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
    kf_pretrain:
        enabled: False
        train:
            lr: 1.0e-5
            weight_decay: 5.0e-2
            n_epochs: 2001
        period:
            log: '0::500'

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
    BIG, dwti_to_inds_big = Data_access_big.load_big_features(
            cf['inputs.big.fold'])
    da_big = Data_access_big(BIG, dwti_to_inds_big, scaler)

    cls_labels, dwti_to_label_train = qload_synthetic_tube_labels(
            tubes_dgt_d[sset_train], tubes_dwein_d[sset_train], dataset)
    _, dwti_to_label_eval = qload_synthetic_tube_labels(
            tubes_dgt_d[sset_eval], tubes_dwein_d[sset_eval], dataset)

    # / Torch section
    tkfeats_train = tkfeats_d[sset_train]
    tkfeats_eval = tkfeats_d[sset_eval]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    def experiment(i):
        log.info(f'Experiment {i}')
        result = _tubefeats_train_mlp_single_run(
            cf, initial_seed+i, da_big,
            tkfeats_train, tkfeats_eval,
            dwti_to_label_train, dwti_to_label_eval,
            tubes_dwein_eval, tubes_dgt_eval,
            sset_eval, dataset)
        return result

    if n_trials == 1:
        experiment(0)
        return

    isaver = snippets.Isaver_simple(
            small.mkdir(out/'isaver_ntrials'),
            range(n_trials), experiment)
    trial_results = isaver.run()
    avg_result = {}
    avg_result['acc_flattube_synt'] = \
            np.mean([tr['acc_flattube_synt'] for tr in trial_results])
    to_avg = [tr['df_ap_full'] for tr in trial_results]
    avg_result['df_ap_full'] = \
        pd.concat(
            to_avg, keys=range(len(to_avg)), axis=1).mean(axis=1, level=1)
    log.info('Results for average over {} trials'.format(n_trials))
    _tubefeats_display_evalresults(avg_result, sset_eval)


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
        big:
            fold: ~
        ckpt: ~
    seed: 42
    split_assignment: !def ['train/val',
        ['train/val', 'trainval/test']]
    data_scaler: !def ['no',
        ['keyframes', 'no']]
    net:
        kind: !def ['layer1',
            ['layer0', 'layer1']]
        layer1:
            H: 32
        ll_dropout: 0.5
    train:
        lr: 1.0e-4
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
            save: '0::1'
            loss_log: '0::1'
            full_eval: '0::1'
    """)
    cf = cfg.parse()

    # Seeds
    initial_seed = cf['seed']
    torch.manual_seed(initial_seed)
    rgen = np.random.default_rng(initial_seed)

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
    man_bigv2 = Manager_loader_big_v2(cf['inputs.big.fold'], scaler,
            cf['train.tubes.stride'],
            cf['train.tubes.top_n_matches'],
            dataset, tubes_dwein_train, tubes_dgt_train, tkfeats_train)

    # Model
    D_in = man_bigv2.BIG.shape[-1]
    model = define_mlp_model(cf, D_in, 11)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])

    ckpt = Checkpointer(model, optimizer)

    # Restore previous run
    rundir = small.mkdir(out/'rundir')
    checkpoint_path = (Manager_checkpoint_name.find_last_checkpoint(rundir))
    if '--new' in add_args:
        checkpoint_path = None
    start_epoch = (ckpt.restore_model_magic(checkpoint_path,
        cf['inputs.ckpt'], cf['train.start_epoch']))

    # Training
    n_epochs = cf['train.n_epochs']
    for i_epoch in range(start_epoch, n_epochs):
        loader = man_bigv2.get_train_loader(
            cf['train.batch_size'], rgen,
            cf['train.tubes.frame_dist'],
            cf['train.tubes.add_keyframes'])
        # loader = man_bigv2.get_train_loader(batch_size, rgen, 64)
        model.train()
        l_avg = snippets.misc.Averager()
        avg_bs = snippets.misc.Averager()
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

            if snippets.check_step_sslice(i_batch,
                    cf['period.i_batch.loss_log']):
                Nb = len(loader)
                loss_str = (f'loss(all/last):{l_avg.avg:.4f}/{l_avg.last:.4f}')
                log.info(f'{i_epoch=}, {i_batch=}/{Nb}; {loss_str}')
        log.info('Epoch stats: avg_batchsize {}, loader_size {} '.format(
            avg_bs.avg, len(loader)))
        if snippets.check_step_sslice(i_epoch,
                cf['period.i_epoch.save']):
            ckpt.save_epoch(rundir, i_epoch)
        if snippets.check_step_sslice(i_epoch,
                cf['period.i_epoch.loss_log']):
            model.eval()
            kacc_train = _quick_accuracy_over_kfeat(
                    tkfeats_train, model, True)*100
            kacc_eval = _quick_accuracy_over_kfeat(
                    tkfeats_eval, model, True)*100
            model.train()
            loss_str = (f'loss(all/last):{l_avg.avg:.4f}/{l_avg.last:.4f}')
            log.info(f'{i_epoch=}: {loss_str} '
                    f'{kacc_train=:.2f} {kacc_eval=:.2f}')
        if snippets.check_step_sslice(i_epoch,
                cf['period.i_epoch.full_eval']):
            model.eval()
            evalset_result = _tubefeats_evalset_perf(
                    model, man_bigv2, dwti_to_label_eval,
                    dataset, tubes_dwein_eval, tubes_dgt_eval)
            model.train()
            log.info(f'Evalset perf at {i_epoch=}')
            _tubefeats_display_evalresults(evalset_result, sset_eval)
