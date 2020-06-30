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
    def load(cf, Vgroup):
        # Produce keyframe datasets realquick
        featname = cf['inputs.keyframes.featname']
        keyframes_featfold = Path(cf['inputs.keyframes.fold'])
        keyframes = small.load_pkl(keyframes_featfold/'keyframes.pkl')
        outputs = small.load_pkl(
                keyframes_featfold/'dict_outputs.pkl')[featname]
        kfeats = {}
        for sset in ['train', 'val', 'trainval', 'test']:
            vids = getattr(Vgroup, sset)
            kfeats[sset] = Ncfg_kfeats.split_off_D(
                    outputs, keyframes, vids)
        return kfeats

    @staticmethod
    def to_torch(kfeats):
        kfeats_t = {}
        for sset, kfeat in kfeats.items():
            kfeat_t = kfeat.copy()
            kfeat_t['X'] = torch.from_numpy(kfeat_t['X'])
            kfeat_t['Y'] = torch.from_numpy(kfeat_t['Y'])
            kfeats_t[sset] = kfeat_t
        return kfeats_t


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


# def evaluate_tubes(cf, Vgroup, D_test, dataset, Y_test_np_softmax):
#
#     return df_recall_s_nodiff, df_ap_s_nodiff


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


def fitscale_kfeats(kfeats):
    # Optional standard scaling on trianval
    scaler = StandardScaler()
    scaler.fit(kfeats['trainval']['X'])
    for sset, kfeat in kfeats.items():
        kfeat['X'] = scaler.transform(kfeat['X'])
    return scaler


def load_gt_and_wein_tubes(tubes_dwein_fold, dataset, Vgroup):
    # / Load tubes
    tubes_dwein_all: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(tubes_dwein_fold)
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


def _eval_full_tubes(BIG, dwti_to_inds_big, dwtis, model, scaler):
    tube_sofmaxes = {}
    with torch.no_grad():
        for dwti in dwtis:
            inds_big = dwti_to_inds_big[dwti]
            feats = BIG[inds_big]
            feats = feats.astype(np.float32)
            if scaler is not None:
                feats = scaler.transform(feats)
            feats = torch.from_numpy(feats)
            preds = model(feats)
            preds = softmax(preds, axis=-1)
            tube_sofmaxes[dwti] = preds.numpy()
    return tube_sofmaxes


def _compute_tube_full_acc(tube_softmaxes, dwti_to_label):
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
    av_stubes: AV_dict[T_dwein_scored] = {}
    for dwt_index, tube in tubes_dwein.items():
        softmaxes = tube_softmaxes[dwt_index]
        # scores = softmaxes[:, :-1].mean(axis=0)
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
    train_period_log = cf['train.period.log']
    n_epochs = cf['train.n_epochs']
    # Inputs
    dataset = Ncfg_daly.get_dataset(cf)
    Vgroup = Ncfg_daly.get_vids(cf, dataset)
    kfeats = Ncfg_kfeats.load(cf, Vgroup)
    if cf['data_scaler'] == 'keyframes':
        scaler = fitscale_kfeats(kfeats)
    elif cf['data_scaler'] == 'no':
        scaler = None
    else:
        raise RuntimeError()
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, Vgroup)

    # / Torch section
    kfeats_t = Ncfg_kfeats.to_torch(kfeats)

    def experiment():
        torch.manual_seed(initial_seed)
        D_in = kfeats['train']['X'].shape[-1]
        model = Net_mlp_onelayer(D_in, 10, cf['net.H'])
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.AdamW(model.parameters(),
                lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])

        for epoch in range(n_epochs):
            out_train = model(kfeats_t['train']['X'])
            loss = loss_fn(out_train, kfeats_t['train']['Y'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if snippets.check_step_sslice(epoch, train_period_log):
                model.eval()
                kacc_train = _quick_kf_eval(kfeats_t['train'], model)
                kacc_val = _quick_kf_eval(kfeats_t['val'], model)
                model.train()
                log.info(f'{epoch}: {loss.item()} '
                        f'{kacc_train=:.2f} {kacc_val=:.2f}')
        return model

    model = small.stash2(out/'exp_model.pkl')(experiment)

    # / Final evaluation
    model.eval()

    # // Tube evaluation (real tubes)
    if cf['eval.full_tubes']:
        BIG, dwti_to_inds_big = load_big_features(cf['inputs.big.fold'])
        tube_sofmaxes_val = _eval_full_tubes(
            BIG, dwti_to_inds_big, tubes_dwein_d['val'].keys(), model, scaler)
        av_stubes_val = _quick_fulltube_assign(tubes_dwein_d['val'], tube_sofmaxes_val, dataset)
        df_ap = _quick_tubeval(av_stubes_val, tubes_dgt_d['val'])
        apline = '/'.join(df_ap.loc['all'].values.astype(str))
        log.info(f'Final val perf, full tube evaluation: {apline}')

    # ACC and ROC_AUC
    with torch.no_grad():
        Y_test = model(kfeats_t['val']['X']).numpy()
    Y_test_pred = np.argmax(Y_test, axis=1)
    Y_test_softmax = softmax(Y_test, axis=1).copy()
    acc = accuracy_score(kfeats_t['val']['Y'], Y_test_pred)
    roc_auc = roc_auc_score(kfeats_t['val']['Y'],
            Y_test_softmax, multi_class='ovr')
    log.info(f'{acc=}, {roc_auc=}')
    # // Tube evaluation (via fake GT intersection)
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(
            tubes_dgt_d['val'])
    objactions_vf = create_kinda_objaction_struct(
            dataset, kfeats['val']['kf'], Y_test_softmax)
    # Assigning scores based on intersections
    av_stubes: AV_dict[T_dwein_scored] = \
        score_ftubes_via_objaction_overlap_aggregation(
            dataset, objactions_vf, tubes_dwein_d['val'], 'iou',
            0.1, 0.0, enable_tqdm=False)
    av_stubes_ = av_stubes_above_score(
            av_stubes, 0.0)
    av_stubes_ = compute_nms_for_av_stubes(
            av_stubes_, 0.3)
    iou_thresholds = [.3, .5, .7]
    df_recall_s_nodiff = compute_recall_for_avtubes_as_dfs(
            av_gt_tubes, av_stubes_, iou_thresholds, False)[0]
    df_ap_s_nodiff = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes_, iou_thresholds, False, False)
    log.info(df_ap_s_nodiff)

    # def experiment(i):
    #     Ncfg_keyframe_mlp.optimize(cf, model, loss_fn, D_trainval, log_period)
    #     Y_test_np_softmax, acc, roc_auc = Ncfg_keyframe_mlp.evaluate_acc(model, D_test)
    #     df_recall_s_nodiff, df_ap_s_nodiff = Ncfg_keyframe_mlp.evaluate_tubes(
    #             cf, Vgroup, D_test, dataset, Y_test_np_softmax)
    #     return [acc, roc_auc, df_recall_s_nodiff, df_ap_s_nodiff]
    #
    # isaver = snippets.Isaver_simple(small.mkdir(out/'isaver_ntrials'), range(n_trials), experiment)
    # trial_results = isaver.run()
    # [acc, roc_auc, recall, ap] = zip(*trial_results)
    # acc, roc_auc = map(lambda x: np.array(x)*100, (acc, roc_auc))
    # recall_, ap_ = map(lambda x: pd.concat(
    #     x, keys=range(len(x)), axis=1).mean(axis=1, level=1), (recall, ap))
    # with small.np_printoptions(precision=2):
    #     log.info('Accuracy; mean: {:.2f}, all: {} '.format(np.mean(acc), acc))
    #     log.info('Roc_auc; mean: {:.2f} all: {} '.format(np.mean(roc_auc), roc_auc))
    # log.info(f'Mean Recall\n{recall_}')
    # log.info(f'mean AP:\n{ap_}')


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
            full_eval: '0::10'
            eval:
                full_train: '::'
                full_val: '0::20'
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
    # Inputs
    dataset = Ncfg_daly.get_dataset(cf)
    Vgroup = Ncfg_daly.get_vids(cf, dataset)
    kfeats = Ncfg_kfeats.load(cf, Vgroup)
    if cf['data_scaler'] == 'keyframes':
        scaler = fitscale_kfeats(kfeats)
    elif cf['data_scaler'] == 'no':
        scaler = None
    else:
        raise RuntimeError()
    tubes_dwein_d, tubes_dgt_d = load_gt_and_wein_tubes(
            cf['inputs.tubes_dwein'], dataset, Vgroup)
    BIG, dwti_to_inds_big = load_big_features(cf['inputs.big.fold'])

    cls_labels, dwti_to_label_train = _qload_synthetic_tube_labels(
            tubes_dgt_d['train'], tubes_dwein_d['train'], dataset)
    _, dwti_to_label_val = _qload_synthetic_tube_labels(
            tubes_dgt_d['val'], tubes_dwein_d['val'], dataset)

    # / Torch section
    kfeats_t = Ncfg_kfeats.to_torch(kfeats)
    torch.manual_seed(initial_seed)
    rgen = np.random.default_rng(initial_seed)
    model = Net_mlp_onelayer(2304, 11, cf['net.H'])
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(),
            lr=cf['train.lr'], weight_decay=cf['train.weight_decay'])

    # pretraining
    if cf['kf_pretrain.enabled']:
        pretrain_lr = cf['kf_pretrain.train.lr']
        pretrain_weight_decay = cf['kf_pretrain.train.weight_decay']
        pretrain_niters = cf['kf_pretrain.train.niters']
        pretrain_optimizer = torch.optim.AdamW(model.parameters(),
                lr=pretrain_lr, weight_decay=pretrain_weight_decay)
        pretrain_period_log = cf['kf_pretrain.period.log']
        model.train()
        for t in range(pretrain_niters):
            pred_train = model(kfeats_t['train']['X'])
            loss = loss_fn(pred_train, kfeats_t['train']['Y'])
            if snippets.check_step_sslice(t, pretrain_period_log):
                model.eval()
                kacc_train = _quick_kf_eval(kfeats_t['train'], model)
                kacc_val = _quick_kf_eval(kfeats_t['val'], model)
                model.train()
                log.info(f'{t}: {loss.item()} '
                        f'{kacc_train=:.2f} {kacc_val=:.2f}')
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()

    period_log = cf['train.period.log']
    period_eval_full_train = cf['train.period.eval.full_train']
    period_eval_full_val = cf['train.period.eval.full_val']
    N_epochs = cf['train.n_epochs']
    tubes_per_batch = cf['train.tubes_per_batch']
    frames_per_tube = cf['train.frames_per_tube']
    model.train()
    for epoch in range(N_epochs):
        batches = _quick_shuffle_batches(
            dwti_to_inds_big, rgen, dwti_to_label_train,
            tubes_per_batch, frames_per_tube)
        loader = _quick_dataloader(BIG, batches, scaler)
        for i_batch, (feats, labels) in enumerate(loader):
            pred_train = model(feats)
            loss = loss_fn(pred_train, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if snippets.check_step_sslice(epoch, period_log):
            model.eval()
            kacc_train = _quick_kf_eval(kfeats_t['train'], model)
            kacc_val = _quick_kf_eval(kfeats_t['val'], model)
            model.train()
            log.info(f'{epoch}: {loss.item()} '
                    f'{kacc_train=:.2f} {kacc_val=:.2f}')
        if snippets.check_step_sslice(epoch, period_eval_full_train):
            model.eval()
            with small.QTimer() as t:
                tube_sofmaxes_train = _eval_full_tubes(
                    BIG, dwti_to_inds_big, dwti_to_label_train.keys(), model, scaler)
                acc_full_train = _compute_tube_full_acc(tube_sofmaxes_train, dwti_to_label_train)
            tsec = t.time
            log.info('Train full keyframe acc: Train {:.2f}, took {:.2f} sec'.format(
                acc_full_train*100, tsec))
            model.train()
        if snippets.check_step_sslice(epoch, period_eval_full_val):
            # flat accuracy operates only on some tubes (according to our synthetic labels)
            tube_sofmaxes_val = _eval_full_tubes(
                BIG, dwti_to_inds_big, tubes_dwein_d['val'].keys(), model, scaler)
            acc_full_val = _compute_tube_full_acc(tube_sofmaxes_val, dwti_to_label_val)
            tube_sofmaxes_val_nobg = {k: v[:, :-1] for k, v in tube_sofmaxes_val.items()}
            # map operates over all tubes in val set (according to GT map labels)
            av_stubes_val = _quick_fulltube_assign(tubes_dwein_d['val'], tube_sofmaxes_val_nobg, dataset)
            df_ap = _quick_tubeval(av_stubes_val, tubes_dgt_d['val'])
            apline = '/'.join(df_ap.loc['all'].values.astype(str))
            log.info('Val full keyframe acc: {:.2f}, val tube ap {}'.format(
                acc_full_val*100, apline))

    # proper map evaluation
    model.eval()
    tube_sofmaxes_val = _eval_full_tubes(
        BIG, dwti_to_inds_big, tubes_dwein_d['val'].keys(), model, scaler)
    tube_sofmaxes_val_nobg = {k: v[:, :-1] for k, v in tube_sofmaxes_val.items()}
    av_stubes_val = _quick_fulltube_assign(tubes_dwein_d['val'], tube_sofmaxes_val_nobg, dataset)
    df_ap = _quick_tubeval(av_stubes_val, tubes_dgt_d['val'])
    apline = '/'.join(df_ap.loc['all'].values.astype(str))
    log.info(f'Final val perf: {apline}')
