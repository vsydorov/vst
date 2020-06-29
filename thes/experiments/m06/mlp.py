import h5py
from tqdm import tqdm
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

from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly,
    get_daly_split_vids, split_off_validation_set, Action_name_daly)
from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    get_daly_gt_tubes, remove_hard_dgt_tubes, push_into_avdict,
    AV_dict, loadconvert_tubes_dwein, Objaction_dets,
    dtindex_filter_split, av_stubes_above_score,
    Box_connections_dwti)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)
from thes.tools import snippets
from thes.data.tubes.routines import (
    score_ftubes_via_objaction_overlap_aggregation)
from thes.data.tubes.routines import (
    spatial_tube_iou_v3,
    temporal_ious_where_positive)
from thes.pytorch import (sequence_batch_collate_v2, default_collate)

log = logging.getLogger(__name__)


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


def split_off(X, linked_vids, good_vids):
    if isinstance(X, np.ndarray):
        isin = np.in1d(linked_vids, good_vids)
        result = X[isin]
    elif isinstance(X, list):
        result = [x for x, v in zip(X, linked_vids) if v in good_vids]
    else:
        raise RuntimeError()
    return result


def split_off_D(outputs, keyframes, vids):
    global_kf_vids = [kf['vid'] for kf in keyframes]
    d = dict()
    d['X'] = split_off(outputs, global_kf_vids, vids)
    d['kf'] = split_off(keyframes, global_kf_vids, vids)
    d['Y'] = np.array([kf['action_id'] for kf in d['kf']])
    return d


class Ncfg_daly:
    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        dataset:
            name: ['daly', ['daly']]
            cache_folder: [~, str]
            mirror: ['uname', ~]
            val_split:
                fraction: [0.1, float]
                nsamplings: [20, int]
                seed: [42, int]
        """)

    @staticmethod
    def get_dataset(cf):
        dataset = Dataset_daly_ocv(cf['dataset.mirror'])
        dataset.populate_from_folder(cf['dataset.cache_folder'])
        return dataset

    @staticmethod
    def get_vids(cf, dataset):
        v_fraction = cf['dataset.val_split.fraction']
        v_nsamplings = cf['dataset.val_split.nsamplings']
        v_seed = cf['dataset.val_split.seed']

        class Vgroup:
            val, train = split_off_validation_set(
                    dataset, v_fraction, v_nsamplings, v_seed)
            trainval = get_daly_split_vids(dataset, 'train')
            test = get_daly_split_vids(dataset, 'test')
        return Vgroup


class Ncfg_keyframe_mlp:
    class Net_mlp_featcls(nn.Module):
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

    @staticmethod
    def set_defcfg(cfg):
        cfg.set_deftype("""
        seed: [42, int]
        tubes_dwein: [~, str]
        inputs:
            featfold: [~, str]
            featname: [~, ~]
        """)
        cfg.set_defaults("""
        net:
            H: 32
        train:
            lr: 1.0e-5
            weight_decay: 5.0e-2
            niters: 2000
        n_trials: 5
        log_period: '::'
        """)

    @staticmethod
    def split_features(outputs, keyframes, Vgroup):
        global_kf_vids = [kf['vid'] for kf in keyframes]

        class D_trainval():
            X = split_off(outputs, global_kf_vids, Vgroup.trainval)
            kf = split_off(keyframes, global_kf_vids, Vgroup.trainval)
            kf_vids = [kf['vid'] for kf in kf]
            Y = np.array([kf['action_id'] for kf in kf])

        scaler = StandardScaler()
        D_trainval.X = scaler.fit_transform(D_trainval.X)

        class D_train():
            X = split_off(D_trainval.X, D_trainval.kf_vids, Vgroup.train)
            kf = split_off(D_trainval.kf, D_trainval.kf_vids, Vgroup.train)
            Y = np.array([kf['action_id'] for kf in kf])

        class D_val():
            X = split_off(D_trainval.X, D_trainval.kf_vids, Vgroup.val)
            kf = split_off(D_trainval.kf, D_trainval.kf_vids, Vgroup.val)
            Y = np.array([kf['action_id'] for kf in kf])

        class D_test():
            X = split_off(outputs, global_kf_vids, Vgroup.test)
            kf = split_off(keyframes, global_kf_vids, Vgroup.test)
            Y = np.array([kf['action_id'] for kf in kf])
            X = scaler.transform(X)

        return D_trainval, D_train, D_val, D_test, scaler

    @classmethod
    def create_model(self, dataset, outputs, H):
        D_in = outputs.shape[-1]
        D_out = len(dataset.action_names)
        H = 32
        model = self.Net_mlp_featcls(D_in, D_out, H)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        return model, loss_fn

    @staticmethod
    def qacc(pred, Y):
        return pred.argmax(1).eq(Y).sum().item()/len(Y) * 100

    @classmethod
    def optimize(self, cf, model, loss_fn, D_to_train, log_period='::'):
        optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])
        niters = cf['train.niters']
        X_to_train = torch.from_numpy(D_to_train.X)
        Y_to_train = torch.from_numpy(D_to_train.Y)
        model.train()
        for t in range(niters):
            pred_train = model(X_to_train)
            loss = loss_fn(pred_train, Y_to_train)
            if snippets.check_step_sslice(t, log_period):
                # Train perf
                model.eval()
                with torch.no_grad():
                    pred_trainval = model(X_to_train)
                    acc_trainval = self.qacc(pred_trainval, Y_to_train)
                model.train()
                log.info(f'{t}: {loss.item()} {acc_trainval=:.2f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @staticmethod
    def evaluate_acc(model, D_to_test):
        X_to_test = torch.from_numpy(D_to_test.X)

        model.eval()
        with torch.no_grad():
            Y_test = model(X_to_test)
        Y_test_np = Y_test.numpy()
        Y_test_preds = np.argmax(Y_test_np, axis=1)
        acc = accuracy_score(D_to_test.Y, Y_test_preds)

        Y_test_np_softmax = softmax(Y_test_np, axis=1)
        Y_test_np_softmax = Y_test_np_softmax.copy()
        roc_auc = roc_auc_score(D_to_test.Y, Y_test_np_softmax,
                multi_class='ovr')
        return Y_test_np_softmax, acc, roc_auc

    @staticmethod
    def evaluate_tubes(cf, Vgroup, D_test, dataset, Y_test_np_softmax):
        # // Tube AP (very, very rought performance)
        # Dwein tubes
        tubes_dwein: Dict[I_dwein, T_dwein] = \
                loadconvert_tubes_dwein(cf['tubes_dwein'])
        tubes_dwein_test = dtindex_filter_split(tubes_dwein, Vgroup.test)
        # GT tubes
        dgt_tubes: Dict[I_dgt, T_dgt] = \
                get_daly_gt_tubes(dataset)
        dgt_tubes_test = dtindex_filter_split(dgt_tubes, Vgroup.test)
        av_gt_tubes_test: AV_dict[T_dgt] = push_into_avdict(dgt_tubes_test)

        objactions_vf = create_kinda_objaction_struct(
                dataset, D_test.kf, Y_test_np_softmax)
        # Assigning scores based on intersections
        av_stubes: AV_dict[T_dwein_scored] = \
            score_ftubes_via_objaction_overlap_aggregation(
                dataset, objactions_vf, tubes_dwein_test, 'iou',
                0.1, 0.0, enable_tqdm=False)
        av_stubes_ = av_stubes_above_score(
                av_stubes, 0.0)
        av_stubes_ = compute_nms_for_av_stubes(
                av_stubes_, 0.3)
        iou_thresholds = [.3, .5, .7]
        df_recall_s_nodiff = compute_recall_for_avtubes_as_dfs(
                av_gt_tubes_test, av_stubes_, iou_thresholds, False)[0]
        df_ap_s_nodiff = compute_ap_for_avtubes_as_df(
                av_gt_tubes_test, av_stubes_, iou_thresholds, False, False)
        return df_recall_s_nodiff, df_ap_s_nodiff


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


def record_overlaps(tubes_dgt, tubes_dwein):
    overlap_hits = {}
    for dgt_index, gt_tube in tqdm(tubes_dgt.items(),
            total=len(tubes_dgt)):
        vid, action_name, ins_id = dgt_index
        dwt_vid = {k: v for k, v in tubes_dwein.items()
                if k[0] == vid}
        dwt_vid_keys = list(dwt_vid.keys())
        dwt_vid_values = list(dwt_vid.values())
        dwt_frange = np.array([
            (x['start_frame'], x['end_frame']) for x in dwt_vid_values])
        # Temporal
        t_ious, pids = temporal_ious_where_positive(
            gt_tube['start_frame'], gt_tube['end_frame'], dwt_frange)
        # Spatial (where temporal >0)
        dwt_intersect = [dwt_vid_values[pid] for pid in pids]
        sp_mious = [spatial_tube_iou_v3(p, gt_tube)
                for p in dwt_intersect]
        for p, t_iou, sp_miou in zip(pids, t_ious, sp_mious):
            st_iou = t_iou * sp_miou
            if st_iou > 0:
                dwt_vid = dwt_vid_keys[p]
                overlap_hits.setdefault(dwt_vid, []).append(
                        [action_name, (t_iou, sp_miou, st_iou)])
    best_ious = {}
    for k, v in overlap_hits.items():
        vsorted_last = sorted(v, key=lambda x: x[1][0])[-1]
        action_name = vsorted_last[0]
        st_miou = vsorted_last[1][2]
        best_ious[k] = (action_name, st_miou)
    return best_ious


def create_synthetic_tube_labels(tubes_dwein, best_ious):
    # Assign to classes
    POS_THRESH = 0.5
    HN_THRESH = 0.3
    labels: Dict[I_dgt, str] = {}
    for dwt_index in tubes_dwein.keys():
        label = 'background'
        if dwt_index in best_ious:
            best_cls, best_iou = best_ious[dwt_index]
            if best_iou > POS_THRESH:
                label = best_cls
            elif 0 < best_iou < HN_THRESH:
                label = 'background_hard'
            else:
                label = 'none'
        labels[dwt_index] = label
    return labels


def _stuff():
    # / Load only useful features to RAM, to be sent to
    # // Prepare indices
    dwtis = list(ilabels_train.keys())
    inds_h5 = []
    for dwti in dwtis:
        inds_h5.append(dwti_to_inds_h5[dwti])
    counts = np.cumsum([len(x) for x in inds_h5])
    flat_inds_h5 = np.hstack(inds_h5)
    as_ind = np.argsort(flat_inds_h5)
    rev_as_ind = np.argsort(as_ind)
    flat_sorted_inds_h5 = flat_inds_h5[as_ind]
    # /// MAPPING: dwtis -> index over flat_sorted_feats
    # flat_sorted_inds_h5[irange[X]] == inds_h5[X]
    flsorted_irange = np.arange(len(flat_sorted_inds_h5))
    fl_irange = flsorted_irange[rev_as_ind]
    irange = np.split(fl_irange, counts[:-1])
    dwti_to_flsi_h5 = dict(zip(dwtis, irange))


def _qload_keyframe_datasets(cf, Vgroup):
    # Produce keyframe datasets realquick
    keyframes_featfold = Path(cf['inputs.keyframes_featfold'])
    keyframes = small.load_pkl(keyframes_featfold/'keyframes.pkl')
    outputs = small.load_pkl(
            keyframes_featfold/'dict_outputs.pkl')['roipooled']
    d = {}
    for sset in ['train', 'val', 'trainval', 'test']:
        vids = getattr(Vgroup, sset)
        d[sset] = split_off_D(outputs, keyframes, vids)
    return d


def _kf_feature_scale(kfeats):
    # Optional standard scaling on trianval
    scaler = StandardScaler()
    scaler.fit(kfeats['trainval']['X'])
    for sset, kfeat in kfeats.items():
        kfeat['X'] = scaler.transform(kfeat['X'])
    return scaler


def kf_features_to_torch(kfeats):
    kfeats_t = {}
    for sset, kfeat in kfeats.items():
        kfeat_t = kfeat.copy()
        kfeat_t['X'] = torch.from_numpy(kfeat_t['X'])
        kfeat_t['Y'] = torch.from_numpy(kfeat_t['Y'])
        kfeats_t[sset] = kfeat_t
    return kfeats_t


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


def _qload_tube_to_frame_mapping(tubes_featfold):
    """Load whole npy file"""
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


def _qload_tubes(cf, dataset, Vgroup):
    # / Load tubes
    tubes_dwein_all: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    tubes_dgt_all: Dict[I_dgt, T_dgt] = get_daly_gt_tubes(dataset)
    tubes_dgt_all = remove_hard_dgt_tubes(tubes_dgt_all)
    # // Per subset
    tubes_dwein_d = {}
    tubes_dgt_d = {}
    for sset in ['train', 'val']:
        vgroup = getattr(Vgroup, sset)
        tubes_dwein_d[sset] = \
                dtindex_filter_split(tubes_dwein_all, vgroup)
        tubes_dgt_d[sset] = \
                dtindex_filter_split(tubes_dgt_all, vgroup)
    return tubes_dwein_d, tubes_dgt_d


def _qload_synthetic_tube_labels(tubes_dgt, tubes_dwein, dataset):
    # / Divide trainval tubes into classes (intersection with GT tubes)
    best_ious = record_overlaps(tubes_dgt, tubes_dwein)
    labels_train: Dict[I_dgt, str] = create_synthetic_tube_labels(
            tubes_dwein, best_ious)
    # / Create classification dataset
    cls_labels = dataset.action_names + ['background']
    dwti_to_label = {}
    for dwti, label in labels_train.items():
        if label == 'none':
            continue
        elif label in ('background', 'background_hard'):
            ilabel = len(dataset.action_names)
        else:
            ilabel = dataset.action_names.index(label)
        dwti_to_label[dwti] = ilabel
    return cls_labels, dwti_to_label


def _create_quick_model(cls_labels):
    # Define model
    train_lr = 1.0e-05
    weight_decay = 5.0e-2
    D_in = 2304
    D_out = len(cls_labels)
    H = 32
    model = Net_mlp_onelayer(D_in, D_out, H)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=train_lr, weight_decay=weight_decay)
    return loss_fn, optimizer, model


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


def _quick_kf_eval(kfeat_t, model):
    def _qacc(pred, Y):
        return pred.argmax(1).eq(Y).sum().item()/len(Y) * 100
    with torch.no_grad():
        pred_trainval = model(kfeat_t['X'])
    acc = _qacc(pred_trainval[:, :-1], kfeat_t['Y'])
    return acc


def _quick_fulltube_assign(
        BIG, dwti_to_inds_big, tubes_dwein, model, scaler, dataset):
    test_preds_avg = []
    with torch.no_grad():
        for i, dwti in enumerate(tubes_dwein.keys()):
            inds_big = dwti_to_inds_big[dwti]
            feats = BIG[inds_big]
            feats = feats.astype(np.float32)
            if scaler is not None:
                feats = scaler.transform(feats)
            feats = torch.from_numpy(feats)
            preds = model(feats)
            preds = softmax(preds, axis=-1)
            preds = preds[:, :-1].mean(0).numpy()
            test_preds_avg.append(preds)

    av_stubes: AV_dict[T_dwein_scored] = {}
    for (dwt_index, tube), scores in zip(
            tubes_dwein.items(), test_preds_avg):
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
    return (df_ap_s_nodiff*100).round(2)


# Experiments


def torchmlp_feat_classify_validate(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_keyframe_mlp.set_defcfg(cfg)
    cf = cfg.parse()

    # Inputs
    initial_seed = cf['seed']
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_daly.get_vids(cf, dataset)
    computed_featfold = Path(cf['inputs.featfold'])
    featname = cf['inputs.featname']
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'dict_outputs.pkl')[featname]
    D_trainval, D_train, D_val, D_test, _ = \
            Ncfg_keyframe_mlp.split_features(outputs, keyframes, Vgroup)

    torch.manual_seed(initial_seed)
    model, loss_fn = Ncfg_keyframe_mlp.create_model(dataset, outputs, cf['net.H'])

    # model, loss_fn = Ncfg_keyframe_mlp.create_model(cf['net.H'])


def torchmlp_feat_classify_test(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_keyframe_mlp.set_defcfg(cfg)
    cf = cfg.parse()

    # Inputs
    initial_seed = cf['seed']
    n_trials = cf['n_trials']
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_daly.get_vids(cf, dataset)
    computed_featfold = Path(cf['inputs.featfold'])
    featname = cf['inputs.featname']
    log_period = cf['log_period']
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'dict_outputs.pkl')[featname]
    D_trainval, D_train, D_val, D_test, _ = \
            Ncfg_keyframe_mlp.split_features(outputs, keyframes, Vgroup)

    def experiment(i):
        torch.manual_seed(initial_seed+i)
        model, loss_fn = Ncfg_keyframe_mlp.create_model(dataset, outputs, cf['net.H'])
        Ncfg_keyframe_mlp.optimize(cf, model, loss_fn, D_trainval, log_period)
        Y_test_np_softmax, acc, roc_auc = Ncfg_keyframe_mlp.evaluate_acc(model, D_test)
        df_recall_s_nodiff, df_ap_s_nodiff = Ncfg_keyframe_mlp.evaluate_tubes(
                cf, Vgroup, D_test, dataset, Y_test_np_softmax)
        return [acc, roc_auc, df_recall_s_nodiff, df_ap_s_nodiff]

    isaver = snippets.Isaver_simple(small.mkdir(out/'isaver_ntrials'), range(n_trials), experiment)
    trial_results = isaver.run()
    [acc, roc_auc, recall, ap] = zip(*trial_results)
    acc, roc_auc = map(lambda x: np.array(x)*100, (acc, roc_auc))
    recall_, ap_ = map(lambda x: pd.concat(
        x, keys=range(len(x)), axis=1).mean(axis=1, level=1), (recall, ap))
    with small.np_printoptions(precision=2):
        log.info('Accuracy; mean: {:.2f}, all: {} '.format(np.mean(acc), acc))
        log.info('Roc_auc; mean: {:.2f} all: {} '.format(np.mean(roc_auc), roc_auc))
    log.info(f'Mean Recall\n{recall_}')
    log.info(f'mean AP:\n{ap_}')


def torchmlp_hack_around_rcnn_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_keyframe_mlp.set_defcfg(cfg)
    cf = cfg.parse()

    # Inputs
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_keyframe_mlp.get_vids(cf, dataset)
    computed_featfold = Path(cf['inputs.featfold'])
    featname = cf['inputs.featname']
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'dict_outputs.pkl')[featname]

    # Check test perf via raw predictions
    global_kf_vids = [kf['vid'] for kf in keyframes]
    X = split_off(outputs, global_kf_vids, Vgroup.test)
    kf = split_off(keyframes, global_kf_vids, Vgroup.test)
    Y = np.array([kf['action_id'] for kf in kf])

    Y_test_preds = X[:, 1:].argmax(1)
    acc = accuracy_score(Y, Y_test_preds)
    Y_test_np_softmax = softmax(X[:, 1:], axis=1)
    Y_test_np_softmax = Y_test_np_softmax.copy()
    roc_auc = roc_auc_score(Y, Y_test_np_softmax, multi_class='ovr')
    log.info(f'{acc*100=:.2f}, {roc_auc*100=:.2f}')

    class D_test:
        pass

    D_test.kf = kf
    df_recall_s_nodiff, df_ap_s_nodiff = Ncfg_keyframe_mlp.evaluate_tubes(
            cf, Vgroup, D_test, dataset, X[:, 1:])
    log.info(f'AP=\n{df_ap_s_nodiff}')


def tubefeats_train_mlp_in_ram_npy(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    cfg.set_deftype("""
    seed: [42, int]
    tubes_dwein: [~, str]
    inputs:
        featfold: [~, str]
        keyframes_featfold: [~, ~]
    """)
    cfg.set_defaults("""
    net:
        H: 32
    train:
        lr: 1.0e-5
        weight_decay: 5.0e-2
        niters: 2000
    n_trials: 5
    log_period: '::'
    """)
    cf = cfg.parse()
    initial_seed = cf['seed']

    # Inputs
    dataset = Ncfg_daly.get_dataset(cf)
    Vgroup = Ncfg_daly.get_vids(cf, dataset)
    kfeats = _qload_keyframe_datasets(cf, Vgroup)
    scaler = _kf_feature_scale(kfeats)

    tubes_dwein_d, tubes_dgt_d = _qload_tubes(cf, dataset, Vgroup)
    tubes_featfold = Path(cf['inputs.featfold'])
    BIG, dwti_to_inds_big = _qload_tube_to_frame_mapping(tubes_featfold)

    cls_labels, dwti_to_label = _qload_synthetic_tube_labels(
            tubes_dgt_d['train'], tubes_dwein_d['train'], dataset)

    # / Torch section
    kfeats_t = kf_features_to_torch(kfeats)
    torch.manual_seed(initial_seed)
    rgen = np.random.default_rng(42)
    loss_fn, optimizer, model = _create_quick_model(cls_labels)

    log_period = '::1'
    map_eval_period = '0::10'
    model.train()
    N_epochs = 120
    TUBES_PER_BATCH = 500
    FRAMES_PER_TUBE = 2
    for epoch in range(N_epochs):
        batches = _quick_shuffle_batches(
            dwti_to_inds_big, rgen, dwti_to_label,
            TUBES_PER_BATCH, FRAMES_PER_TUBE)
        loader = _quick_dataloader(BIG, batches, scaler)
        for i_batch, (feats, labels) in enumerate(loader):
            pred_train = model(feats)
            loss = loss_fn(pred_train, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if snippets.check_step_sslice(epoch, log_period):
            model.eval()
            acc_train = _quick_kf_eval(kfeats_t['train'], model)
            acc_val = _quick_kf_eval(kfeats_t['val'], model)
            model.train()
            log.info(f'{epoch}: {loss.item()} '
                    f'{acc_train=:.2f} {acc_val=:.2f}')
        if snippets.check_step_sslice(epoch, map_eval_period):
            model.eval()
            with small.QTimer() as t:
                av_stubes_val = _quick_fulltube_assign(
                    BIG, dwti_to_inds_big, tubes_dwein_d['val'], model, scaler, dataset)
                df_ap = _quick_tubeval(av_stubes_val, tubes_dgt_d['val'])
            apline = '/'.join(df_ap.loc['all'].values.astype(str))
            log.info(f'val AP: {apline}, took {t.time:.2f}s')
            model.train()

    # proper map evaluation
    model.eval()
    av_stubes_val = _quick_fulltube_assign(
        BIG, dwti_to_inds_big, tubes_dwein_d['val'], model, scaler, dataset)
    df_ap = _quick_tubeval(av_stubes_val, tubes_dgt_d['val'])
    apline = '/'.join(df_ap.loc['all'].values.astype(str))
    log.info(f'Final val perf: {apline}')
