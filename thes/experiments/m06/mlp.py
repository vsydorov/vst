import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import (Dict, Any, Optional, List, cast, Tuple)
from sklearn.preprocessing import (StandardScaler)
from sklearn.metrics import (
    accuracy_score, roc_auc_score)
from scipy.special import softmax

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from vsydorov_tools import small

from thes.data.dataset.daly import (
    Ncfg_daly, load_gt_and_wein_tubes)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly,
    get_daly_split_vids, split_off_validation_set, Action_name_daly)
from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    get_daly_gt_tubes, remove_hard_dgt_tubes, push_into_avdict,
    AV_dict, loadconvert_tubes_dwein, Objaction_dets,
    dtindex_filter_split, av_stubes_above_score,
    Box_connections_dwti)
from thes.data.tubes.routines import (
    score_ftubes_via_objaction_overlap_aggregation,
    qload_synthetic_tube_labels)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)
from thes.tools import snippets
from thes.pytorch import (sequence_batch_collate_v2, default_collate)

log = logging.getLogger(__name__)


class Ncfg_kfeats:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        inputs:
            keyframes:
                fold: [~, str]
                featname: [~, ~]
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
    def split_off_D(outputs, keyframes, vids):
        global_kf_vids = [kf['vid'] for kf in keyframes]
        d = dict()
        d['X'] = Ncfg_kfeats.split_off(outputs, global_kf_vids, vids)
        d['kf'] = Ncfg_kfeats.split_off(keyframes, global_kf_vids, vids)
        d['Y'] = np.array([kf['action_id'] for kf in d['kf']])
        return d

    @staticmethod
    def load(cf, vgroup):
        # Produce keyframe datasets realquick
        featname = cf['inputs.keyframes.featname']
        keyframes_featfold = Path(cf['inputs.keyframes.fold'])
        keyframes = small.load_pkl(keyframes_featfold/'keyframes.pkl')
        outputs = small.load_pkl(
                keyframes_featfold/'dict_outputs.pkl')[featname]
        kfeats_d = {}
        for sset, vids in vgroup.items():
            kfeats_d[sset] = Ncfg_kfeats.split_off_D(
                    outputs, keyframes, vids)
        return kfeats_d

    @staticmethod
    def to_torch(kfeats_d):
        tkfeats_d = {}
        for sset, kfeats in kfeats_d.items():
            tkfeats = kfeats.copy()
            tkfeats['X'] = torch.from_numpy(tkfeats['X'])
            tkfeats['Y'] = torch.from_numpy(tkfeats['Y'])
            tkfeats_d[sset] = tkfeats
        return tkfeats_d


def create_kinda_objaction_struct(dataset, test_kfs, Y_conf_scores_sm):
    # // Creating kinda objaction structure
    # Group vid -> frame
    grouped_kfscores_vf: Dict[Vid_daly, Dict[int, Any]] = {}
    for kf, scores in zip(test_kfs, Y_conf_scores_sm):
        vid = kf['vid']
        frame0 = kf['frame0']
        pred_box = kf['bbox']
        (grouped_kfscores_vf
                .setdefault(vid, {})
                .setdefault(frame0, [])
                .append([pred_box, scores]))
    # fake objactions
    objactions_vf: Dict[Vid_daly, Dict[int, Objaction_dets]] = {}
    for vid, grouped_kfscores_f in grouped_kfscores_vf.items():
        for frame_ind, gkfscores in grouped_kfscores_f.items():
            all_scores, all_boxes, all_classes = [], [], []
            for (box, scores) in gkfscores:
                all_boxes.append(np.tile(box, (len(scores), 1)))
                all_classes.append(np.array(dataset.action_names))
                all_scores.append(scores)
            all_scores_ = np.hstack(all_scores)
            all_classes_ = np.hstack(all_classes)
            all_boxes_ = np.vstack(all_boxes)
            detections = {
                    'pred_boxes': all_boxes_,
                    'scores': all_scores_,
                    'pred_classes': all_classes_}
            objactions_vf.setdefault(vid, {})[frame_ind] = detections
    return objactions_vf


class Net_mlp_onelayer(nn.Module):
    def __init__(self, D_in, D_out, H, dropout_rate=0.5):
        super().__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, D_out)
        self.dropout = nn.Dropout(dropout_rate)
        # self.bn = nn.BatchNorm1d(H, momentum=0.1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        # x = self.bn(x)
        x = self.linear2(x)
        return x


def fitscale_kfeats(kfeats_d):
    # Optional standard scaling on trianval
    scaler = StandardScaler()
    scaler.fit(kfeats_d['trainval']['X'])
    for sset, kfeats in kfeats_d.items():
        kfeats['X'] = scaler.transform(kfeats['X'])
    return scaler


def get_dwti_big_mapping(
        connections_f, box_inds2
        ) -> Dict[I_dwein, np.ndarray]:
    dwti_h5_inds: Dict[I_dwein, np.ndarray] = {}
    for con, bi2 in zip(connections_f.values(), box_inds2):
        bi_range = np.arange(bi2[0], bi2[1])
        for dwti, bi in zip(con['dwti_sources'], bi_range):
            dwti_h5_inds.setdefault(dwti, []).append(bi)
    dwti_h5_inds = {k: np.array(sorted(v))
            for k, v in dwti_h5_inds.items()}
    return dwti_h5_inds


def load_big_features(tubes_featfold):
    """Load whole npy file"""
    tubes_featfold = Path(tubes_featfold)
    # Load connections, arrange back into dwt_index based structure
    connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti] = \
            small.load_pkl(tubes_featfold/'connections_f.pkl')
    # (N_frames, 2) ndarray of BIG indices
    box_inds2 = small.load_pkl(tubes_featfold/'box_inds2.pkl')
    # Mapping dwti -> 1D ndarray of BIG indices
    dwti_to_inds_big = get_dwti_big_mapping(connections_f, box_inds2)
    # Features
    with small.QTimer('big numpy load'):
        BIG = np.load(str(tubes_featfold/"feats.npy"))
    return BIG, dwti_to_inds_big


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


def _quick_dataloader(BIG, batches, scaler):
    # // torch dataset
    td_h5 = TD_thin_over_BIG(BIG, batches, scaler)
    loader = torch.utils.data.DataLoader(td_h5,
        batch_size=None, num_workers=0,
        collate_fn=None)
    return loader


def _quick_kf_eval(kfeat_t, model, cutoff_last_dim):
    def _qacc(pred, Y):
        return pred.argmax(1).eq(Y).sum().item()/len(Y)
    with torch.no_grad():
        pred_eval = model(kfeat_t['X'])
    if cutoff_last_dim:
        pred_eval = pred_eval[:, :-1]
    acc = _qacc(pred_eval, kfeat_t['Y'])
    return acc


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
        batches = _quick_shuffle_batches(
            self.dwti_to_inds_big, rgen, dwti_to_label_train,
            tubes_per_batch, frames_per_tube)
        loader = _quick_dataloader(self.BIG, batches, self.scaler)
        return loader


def _eval_tube_softmaxes(model, da_big, dwtis):
    tube_sofmaxes = {}
    with torch.no_grad():
        for dwti in dwtis:
            preds = model(da_big.get(model, dwti))
            preds = softmax(preds, axis=-1)
            tube_sofmaxes[dwti] = preds.numpy()
    return tube_sofmaxes


def _compute_flattube_syntlabel_acc(tube_softmaxes, dwti_to_label):
    # Assert absence of background cls
    x = next(iter(tube_softmaxes.values()))
    assert x.shape[-1] == 11

    flat_sm = []
    flat_label = []
    for dwti, label in dwti_to_label.items():
        softmaxes = tube_softmaxes[dwti]
        flat_sm.append(softmaxes)
        flat_label.append(np.repeat(label, len(softmaxes)))
    flat_sm = np.vstack(flat_sm)
    flat_label = np.hstack(flat_label)
    return accuracy_score(flat_label, flat_sm.argmax(axis=1))


def _quick_fulltube_assign(tubes_dwein, tube_softmaxes, dataset):
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
            stube = tube.copy()
            stube['score'] = score
            stube = cast(T_dwein_scored, stube)
            (av_stubes
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))
    return av_stubes

def _quick_tubeval(av_stubes, tubes_dgt):
    av_stubes_ = av_stubes_above_score(av_stubes, 0.0)
    av_stubes_ = compute_nms_for_av_stubes(av_stubes_, 0.3)
    iou_thresholds = [.3, .5, .7]
    av_gt_tubes_test: AV_dict[T_dgt] = push_into_avdict(tubes_dgt)
    df_ap_s_nodiff = compute_ap_for_avtubes_as_df(
            av_gt_tubes_test, av_stubes_, iou_thresholds, False, False)
    return df_ap_s_nodiff


def _kffeats_eval(
        cf, model, da_big, tkfeats_train, tkfeats_eval,
        tubes_dwein_eval, tubes_dgt_eval, dataset):
    # // Proxy results: Evaluation of kf classifier (acc, rauc) and cheating
    # GT-overlap
    kacc_train = _quick_kf_eval(tkfeats_train, model, False)
    kacc_eval = _quick_kf_eval(tkfeats_eval, model, False)
    with torch.no_grad():
        Y_test = model(tkfeats_eval['X']).numpy()
    Y_test_pred = np.argmax(Y_test, axis=1)
    Y_test_softmax = softmax(Y_test, axis=1).copy()
    kf_acc = accuracy_score(tkfeats_eval['Y'], Y_test_pred)

    kf_roc_auc = roc_auc_score(tkfeats_eval['Y'],
            Y_test_softmax, multi_class='ovr')
    # // Tube evaluation (via fake GT intersection)
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(
            tubes_dgt_eval)
    objactions_vf = create_kinda_objaction_struct(
            dataset, tkfeats_eval['kf'], Y_test_softmax)
    # Assigning scores based on intersections
    av_stubes: AV_dict[T_dwein_scored] = \
        score_ftubes_via_objaction_overlap_aggregation(
            dataset, objactions_vf, tubes_dwein_eval, 'iou',
            0.1, 0.0, enable_tqdm=False)
    av_stubes_ = av_stubes_above_score(
            av_stubes, 0.0)
    av_stubes_ = compute_nms_for_av_stubes(
            av_stubes_, 0.3)
    iou_thresholds = [.3, .5, .7]
    df_recall_cheat = compute_recall_for_avtubes_as_dfs(
        av_gt_tubes, av_stubes_, iou_thresholds, False)[0]
    df_ap_cheat = compute_ap_for_avtubes_as_df(
        av_gt_tubes, av_stubes_, iou_thresholds, False, False)
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
        tube_sofmaxes_eval = _eval_tube_softmaxes(
            model, da_big, tubes_dwein_eval.keys())
        # Flattube, synthetic accuracy
        tube_sofmaxes_eval_0bg = {k: np.pad(v, ((0, 0), (0, 1)))
                for k, v in tube_sofmaxes_eval.items()}
        acc_flattube_synt = _compute_flattube_syntlabel_acc(tube_sofmaxes_eval_0bg, dwti_to_label_eval)
        # MAP, all tubes
        av_stubes_eval = _quick_fulltube_assign(tubes_dwein_eval, tube_sofmaxes_eval, dataset)
        df_ap_full = _quick_tubeval(av_stubes_eval, tubes_dgt_eval)
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


def _kffeats_experiment(
        cf, initial_seed, da_big, tkfeats_train,
        tkfeats_eval, tubes_dwein_eval, tubes_dgt_eval,
        dataset):

    torch.manual_seed(initial_seed)
    train_period_log = cf['train.period.log']
    n_epochs = cf['train.n_epochs']
    D_in = tkfeats_train['X'].shape[-1]
    model = Net_mlp_onelayer(D_in, 10, cf['net.H'])
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])

    for epoch in range(n_epochs):
        out_train = model(tkfeats_train['X'])
        loss = loss_fn(out_train, tkfeats_train['Y'])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if snippets.check_step_sslice(epoch, train_period_log):
            model.eval()
            kacc_train = _quick_kf_eval(tkfeats_train, model, False)
            kacc_eval = _quick_kf_eval(tkfeats_eval, model, False)
            model.train()
            log.info(f'{epoch}: {loss.item():.4f} '
                    f'K.Acc.Train: {kacc_train*100:.2f}; '
                    f'K.Acc.Eval: {kacc_eval*100:.2f}')
    # / Final evaluation
    model.eval()
    result = _kffeats_eval(
            cf, model, da_big, tkfeats_train, tkfeats_eval,
            tubes_dwein_eval, tubes_dgt_eval, dataset)
    return result


def _tubefeats_pretraining(cf, model, loss_fn,
        tkfeats_train, tkfeats_eval):
    pretrain_lr = cf['kf_pretrain.train.lr']
    pretrain_weight_decay = cf['kf_pretrain.train.weight_decay']
    pretrain_optimizer = torch.optim.AdamW(model.parameters(),
            lr=pretrain_lr, weight_decay=pretrain_weight_decay)
    pretrain_n_epochs = cf['kf_pretrain.train.n_epochs']
    pretrain_period_log = cf['kf_pretrain.period.log']
    model.train()
    for epoch in range(pretrain_n_epochs):
        out_train = model(tkfeats_train['X'])
        loss = loss_fn(out_train, tkfeats_train['Y'])
        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()
        if snippets.check_step_sslice(epoch, pretrain_period_log):
            model.eval()
            kacc_train = _quick_kf_eval(tkfeats_train, model, True)
            kacc_eval = _quick_kf_eval(tkfeats_eval, model, True)
            model.train()
            log.info(f'{epoch}: {loss.item():.4f} '
                    f'K.Acc.Train: {kacc_train*100:.2f}; '
                    f'K.Acc.Eval: {kacc_eval*100:.2f}')

def _tubefeats_trainset_perf(model, da_big, dwti_to_label_train):
    with small.QTimer() as t:
        tube_sofmaxes_train = _eval_tube_softmaxes(model, da_big, dwti_to_label_train.keys())
        acc_full_train = _compute_flattube_syntlabel_acc(tube_sofmaxes_train, dwti_to_label_train)
    tsec = t.time
    return acc_full_train, tsec


def _tubefeats_evalset_perf(model, da_big, dwti_to_label_eval,
        dataset, tubes_dwein_eval, tubes_dgt_eval):
    tube_sofmaxes_eval = _eval_tube_softmaxes(
            model, da_big, tubes_dwein_eval.keys())
    # Flat accuracy: only dwto_to_label_eval tubes and includes
    # background cls, should be over 11 classes
    acc_flattube_synt = _compute_flattube_syntlabel_acc(
            tube_sofmaxes_eval, dwti_to_label_eval)
    # MAP: all tubes in tubes_dwein_eval, excludes background (last cls)
    tube_sofmaxes_eval_nobg = {k: v[:, :-1]
            for k, v in tube_sofmaxes_eval.items()}
    av_stubes_eval = _quick_fulltube_assign(
            tubes_dwein_eval, tube_sofmaxes_eval_nobg, dataset)
    df_ap_full = _quick_tubeval(av_stubes_eval, tubes_dgt_eval)
    result = {'acc_flattube_synt': acc_flattube_synt,
            'df_ap_full': df_ap_full}
    return result


def _tubefeats_training(cf, model, loss_fn, rgen, da_big,
        dwti_to_label_train, dwti_to_label_eval,
        tkfeats_train, tkfeats_eval,
        tubes_dwein_eval, tubes_dgt_eval, sset_eval, dataset):
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
            pred_train = model(feats)
            loss = loss_fn(pred_train, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if snippets.check_step_sslice(epoch, period_log):
            model.eval()
            kacc_train = _quick_kf_eval(tkfeats_train, model, True)*100
            kacc_eval = _quick_kf_eval(tkfeats_eval, model, True)*100
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


def _tubefeats_experiment(
        cf, initial_seed, da_big,
        tkfeats_train, tkfeats_eval,
        dwti_to_label_train, dwti_to_label_eval,
        tubes_dwein_eval, tubes_dgt_eval,
        sset_eval, dataset):
    torch.manual_seed(initial_seed)
    rgen = np.random.default_rng(initial_seed)

    model = Net_mlp_onelayer(2304, 11, cf['net.H'])
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # pretraining
    if cf['kf_pretrain.enabled']:
        _tubefeats_pretraining(cf, model, loss_fn,
        tkfeats_train, tkfeats_eval)
    _tubefeats_training(cf, model, loss_fn, rgen, da_big,
        dwti_to_label_train, dwti_to_label_eval,
        tkfeats_train, tkfeats_eval,
        tubes_dwein_eval, tubes_dgt_eval,
        sset_eval, dataset)

    # proper map evaluation
    model.eval()
    result = _tubefeats_evalset_perf(
        model, da_big, dwti_to_label_eval, dataset,
        tubes_dwein_eval, tubes_dgt_eval)
    return result

def _tubefeats_display_evalresults(result, sset_eval):
    acc_flattube_synt = result['acc_flattube_synt']
    df_ap_full = (result['df_ap_full']*100).round(2)
    apline = '/'.join(df_ap_full.loc['all'].values.astype(str))
    log.info('Tube evaluation results ({}): '.format(sset_eval))
    log.debug('AP (full tube tubes):\n{}'.format(df_ap_full))
    log.info(' '.join(
        ('Flattube synthetic acc: {:.2f};'.format(acc_flattube_synt*100),
        'Full tube AP357: {}'.format(apline))))


# Experiments


def kffeats_train_mlp(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_kfeats.set_defcfg(cfg)
    cfg.set_deftype("""
    seed: [42, int]
    inputs:
        tubes_dwein: [~, str]
        big:
            fold: [~, ~]
    data_scaler: ['keyframes', ['keyframes', 'no']]
    split_assignment: ['train/val', ['train/val', 'trainval/test']]
    """)
    cfg.set_defaults("""
    net:
        H: 32
    train:
        lr: 1.0e-5
        weight_decay: 5.0e-2
        n_epochs: 2001
        period:
            log: '0::500'
    eval:
        full_tubes: True
    n_trials: 5
    """)
    cf = cfg.parse()
    # params
    initial_seed = cf['seed']
    n_trials = cf['n_trials']
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # Inputs
    dataset = Ncfg_daly.get_dataset(cf)
    vgroup = Ncfg_daly.get_vids(cf, dataset)
    kfeats_d = Ncfg_kfeats.load(cf, vgroup)
    if cf['data_scaler'] == 'keyframes':
        scaler = fitscale_kfeats(kfeats_d)
    elif cf['data_scaler'] == 'no':
        scaler = None
    else:
        raise RuntimeError()
    if cf['eval.full_tubes']:
        BIG, dwti_to_inds_big = load_big_features(cf['inputs.big.fold'])
        da_big = Data_access_big(BIG, dwti_to_inds_big, scaler)
    else:
        da_big = None
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)

    # / Torch section
    tkfeats_d = Ncfg_kfeats.to_torch(kfeats_d)
    tkfeats_train = tkfeats_d[sset_train]
    tkfeats_eval = tkfeats_d[sset_eval]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    log.info(f'Train/eval splits: {sset_train} {sset_eval=}')

    def experiment(i):
        log.info(f'Experiment {i}')
        result = _kffeats_experiment(
            cf, initial_seed+i, da_big,
            tkfeats_train, tkfeats_eval,
            tubes_dwein_eval, tubes_dgt_eval,
            dataset)
        _kffeats_display_evalresults(result, sset_eval)
        return result

    isaver = snippets.Isaver_simple(
            small.mkdir(out/'isaver_ntrials'), range(n_trials), experiment)
    trial_results = isaver.run()

    if len(trial_results) == 1:
        return  # no need to avg
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
    cfg.set_deftype("""
    seed: [42, int]
    inputs:
        tubes_dwein: [~, str]
        big:
            fold: [~, str]
    data_scaler: ['keyframes', ['keyframes', 'no']]
    split_assignment: ['train/val', ['train/val', 'trainval/test']]
    """)
    cfg.set_defaults("""
    net:
        H: 32
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

    n_trials: 5
    """)
    cf = cfg.parse()
    # params
    initial_seed = cf['seed']
    n_trials = cf['n_trials']
    sset_train, sset_eval = cf['split_assignment'].split('/')
    # Inputs
    dataset = Ncfg_daly.get_dataset(cf)
    vgroup = Ncfg_daly.get_vids(cf, dataset)
    kfeats_d = Ncfg_kfeats.load(cf, vgroup)
    if cf['data_scaler'] == 'keyframes':
        scaler = fitscale_kfeats(kfeats_d)
    elif cf['data_scaler'] == 'no':
        scaler = None
    else:
        raise RuntimeError()
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, vgroup)
    BIG, dwti_to_inds_big = load_big_features(cf['inputs.big.fold'])
    da_big = Data_access_big(BIG, dwti_to_inds_big, scaler)

    cls_labels, dwti_to_label_train = qload_synthetic_tube_labels(
            tubes_dgt_d[sset_train], tubes_dwein_d[sset_train], dataset)
    _, dwti_to_label_eval = qload_synthetic_tube_labels(
            tubes_dgt_d[sset_eval], tubes_dwein_d[sset_eval], dataset)

    # / Torch section
    tkfeats_d = Ncfg_kfeats.to_torch(kfeats_d)

    tkfeats_train = tkfeats_d[sset_train]
    tkfeats_eval = tkfeats_d[sset_eval]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    def experiment(i):
        log.info(f'Experiment {i}')
        result = _tubefeats_experiment(
            cf, initial_seed+i, da_big,
            tkfeats_train, tkfeats_eval,
            dwti_to_label_train, dwti_to_label_eval,
            tubes_dwein_eval, tubes_dgt_eval,
            sset_eval, dataset)
        _tubefeats_display_evalresults(result, sset_eval)
        return result

    isaver = snippets.Isaver_simple(
            small.mkdir(out/'isaver_ntrials'), range(n_trials), experiment)
    trial_results = isaver.run()
    if len(trial_results) == 1:
        return  # no need to avg
    avg_result = {}
    avg_result['acc_flattube_synt'] = np.mean([tr['acc_flattube_synt'] for tr in trial_results])
    to_avg = [tr['df_ap_full'] for tr in trial_results]
    avg_result['df_ap_full']= pd.concat(to_avg, keys=range(len(to_avg)), axis=1).mean(axis=1, level=1)
    log.info('Results for average over {} trials'.format(n_trials))
    _tubefeats_display_evalresults(avg_result, sset_eval)
