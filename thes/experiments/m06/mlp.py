import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import (Dict, Any, Optional, List, cast)
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
    dtindex_filter_split, av_stubes_above_score)
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

class Ncfg_mlp:

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
        dataset:
            name: ['daly', ['daly']]
            cache_folder: [~, str]
            mirror: ['scratch2', ~]
            val_split:
                fraction: [0.1, float]
                nsamplings: [20, int]
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

        return D_trainval, D_train, D_val, D_test

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


# Experiments


def torchmlp_feat_classify_validate(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_mlp.set_defcfg(cfg)
    cf = cfg.parse()

    # Inputs
    initial_seed = cf['seed']
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_mlp.get_vids(cf, dataset)
    computed_featfold = Path(cf['inputs.featfold'])
    featname = cf['inputs.featname']
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'dict_outputs.pkl')[featname]
    D_trainval, D_train, D_val, D_test = \
            Ncfg_mlp.split_features(outputs, keyframes, Vgroup)

    torch.manual_seed(initial_seed)
    model, loss_fn = Ncfg_mlp.create_model(dataset, outputs, cf['net.H'])

    # model, loss_fn = Ncfg_mlp.create_model(cf['net.H'])


def torchmlp_feat_classify_test(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_mlp.set_defcfg(cfg)
    cf = cfg.parse()

    # Inputs
    initial_seed = cf['seed']
    n_trials = cf['n_trials']
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_mlp.get_vids(cf, dataset)
    computed_featfold = Path(cf['inputs.featfold'])
    featname = cf['inputs.featname']
    log_period = cf['log_period']
    keyframes = small.load_pkl(computed_featfold/'keyframes.pkl')
    outputs = small.load_pkl(computed_featfold/'dict_outputs.pkl')[featname]
    D_trainval, D_train, D_val, D_test = \
            Ncfg_mlp.split_features(outputs, keyframes, Vgroup)

    def experiment(i):
        torch.manual_seed(initial_seed+i)
        model, loss_fn = Ncfg_mlp.create_model(dataset, outputs, cf['net.H'])
        Ncfg_mlp.optimize(cf, model, loss_fn, D_trainval, log_period)
        Y_test_np_softmax, acc, roc_auc = Ncfg_mlp.evaluate_acc(model, D_test)
        df_recall_s_nodiff, df_ap_s_nodiff = Ncfg_mlp.evaluate_tubes(
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
    log.info(f'Mean Recall\n{recall}')
    log.info(f'mean AP:\n{ap}')

def torchmlp_hack_around_rcnn_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_mlp.set_defcfg(cfg)
    cf = cfg.parse()

    # Inputs
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_mlp.get_vids(cf, dataset)
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
    df_recall_s_nodiff, df_ap_s_nodiff = Ncfg_mlp.evaluate_tubes(
            cf, Vgroup, D_test, dataset, X[:, 1:])
    log.info(f'AP=\n{df_ap_s_nodiff}')


class Net_mlp_featcls2(nn.Module):
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


def hack_w_tubefeats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
        mirror: ['uname', ~]
        val_split:
            fraction: [0.1, float]
            nsamplings: [20, int]
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

    # Inputs
    initial_seed = cf['seed']
    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    Vgroup = Ncfg_mlp.get_vids(cf, dataset)

    tubes_featfold = Path(cf['inputs.featfold'])

    # // Load tubes
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    tubes_dgt: Dict[I_dgt, T_dgt] = get_daly_gt_tubes(dataset)
    tubes_dgt_nh = remove_hard_dgt_tubes(tubes_dgt)

    # // Divide trainval tubes into classes (intersection with GT tubes)
    tubes_dwein_trainval = dtindex_filter_split(tubes_dwein, Vgroup.trainval)
    tubes_dgt_trainval = dtindex_filter_split(tubes_dgt_nh, Vgroup.trainval)
    overlap_hits = {}
    for dgt_index, gt_tube in tqdm(tubes_dgt_trainval.items(),
            total=len(tubes_dgt_trainval)):
        vid, action_name, ins_id = dgt_index
        dwt_vid = {k: v for k, v in tubes_dwein_trainval.items()
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
        sp_mious = [spatial_tube_iou_v3(p, gt_tube) for p in dwt_intersect]
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

    # Assign to classes
    tubes_dwein_train = dtindex_filter_split(tubes_dwein, Vgroup.trainval)
    POS_THRESH = 0.5
    HN_THRESH = 0.3
    labels = {}
    for dwt_index in tubes_dwein_train.keys():
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

    # Produce keyframe datasets realquick
    keyframes_featfold = Path(cf['inputs.keyframes_featfold'])
    keyframes = small.load_pkl(keyframes_featfold/'keyframes.pkl')
    outputs = small.load_pkl(keyframes_featfold/'dict_outputs.pkl')['roipooled']
    D_trainval, D_train, D_val, D_test = \
            Ncfg_mlp.split_features(outputs, keyframes, Vgroup)

    # Load connections, arrange back into dwt_index based structure
    connections_f = small.load_pkl(tubes_featfold/'connections_f.pkl')
    box_inds2 = small.load_pkl(tubes_featfold/'box_inds2.pkl')
    dwti_binds = {}
    for con, bi2 in tqdm(zip(connections_f.values(), box_inds2),
            total=len(connections_f)):
        bi_range = np.arange(bi2[0], bi2[1])
        for dwti, bi in zip(con['dwti_sources'], bi_range):
            dwti_binds.setdefault(dwti, []).append(bi)
    dwti_binds = {k: np.array(sorted(v)) for k, v in dwti_binds.items()}

    # Separate out the test tube features
    tubes_dwein_test = dtindex_filter_split(tubes_dwein, Vgroup.trainval)
    test_binds = []
    for t in tubes_dwein_test:
        test_binds.append(dwti_binds[t])
    counts = np.cumsum(np.array([len(x) for x in test_binds]))
    flat_test_binds = np.hstack(test_binds)

    # Gather these features (as float 16)
    hf = h5py.File(tubes_featfold/"feats.h5", 'r', libver="latest")
    dset = hf["roipooled_feats"]
    as_ind = np.argsort(flat_test_binds)
    rev_as_ind = np.argsort(as_ind)
    flat_test_binds_asorted = flat_test_binds[as_ind]
    flat_test_feats_asorted = dset[flat_test_binds_asorted, :]

    def experiment(exp_ind):
        # Quickly train a fast one
        torch.manual_seed(initial_seed+exp_ind)
        D_in = 2304
        D_out = len(dataset.action_names)
        H = 32
        model = Net_mlp_featcls2(D_in, D_out, H)
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        Ncfg_mlp.optimize(cf, model, loss_fn, D_trainval, '1999::500')
        Y_test_np_softmax, acc, roc_auc = Ncfg_mlp.evaluate_acc(model, D_test)
        df_recall_s_nodiff, df_ap_s_nodiff = Ncfg_mlp.evaluate_tubes(
            cf, Vgroup, D_test, dataset, Y_test_np_softmax)

        # // Quick evaluation over flat_feats
        # iterating over proper indices right away
        batched_inds = snippets.leqn_split(rev_as_ind, 25_000)
        model.eval()
        flat_test_preds_sm = []
        for b_inds in tqdm(batched_inds):
            b_feats = flat_test_feats_asorted[b_inds]
            X_to_test = torch.from_numpy(b_feats.astype(np.float32))
            with torch.no_grad():
                Y_test = model(X_to_test)
            flat_test_preds_sm.append(softmax(Y_test, axis=1))
        flat_test_preds_sm_np = np.vstack([
            x.numpy() for x in flat_test_preds_sm])

        # Merge predictions according to counts
        test_preds_avg = []
        for i, b in enumerate(np.r_[0, counts[:-1]]):
            e = counts[i]
            preds = flat_test_preds_sm_np[b:e]
            test_preds_avg.append(preds.sum(0))
        test_preds_avg = np.vstack(test_preds_avg)

        # assign scores to the test tubes
        av_stubes: AV_dict[T_dwein_scored] = {}
        for (dwt_index, tube), scores in zip(
                tubes_dwein_test.items(), test_preds_avg):
            (vid, bunch_id, tube_id) = dwt_index
            for action_name, score in zip(dataset.action_names, scores):
                stube = tube.copy()
                stube['score'] = score
                stube = cast(T_dwein_scored, stube)
                (av_stubes
                        .setdefault(action_name, {})
                        .setdefault(vid, []).append(stube))
        return av_stubes

    n_trials = 5
    isaver = snippets.Isaver_simple(
            small.mkdir(out/'isaver_ntrials'), range(n_trials), experiment)
    trial_results = isaver.run()
    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    pass

    # av_stubes_ = av_stubes_above_score(
    #         av_stubes, 0.0)
    # av_stubes_ = compute_nms_for_av_stubes(
    #         av_stubes_, 0.3)
    # iou_thresholds = [.3, .5, .7]
    # tubes_dgt_test = dtindex_filter_split(tubes_dgt_nh, Vgroup.trainval)
    # av_gt_tubes_test: AV_dict[T_dgt] = push_into_avdict(tubes_dgt_test)
    # df_recall_s_nodiff = compute_recall_for_avtubes_as_dfs(
    #         av_gt_tubes_test, av_stubes_, iou_thresholds, False)[0]
    # df_ap_s_nodiff = compute_ap_for_avtubes_as_df(
    #         av_gt_tubes_test, av_stubes_, iou_thresholds, False, False)
    # return df_ap_s_nodiff
    # ap = zip(*trial_results)
    # mean_ap = pd.concat(

def tubefeats_train_mlp(workfolder, cfg_dict, add_args):
    train_lr = 1.0e-05
    weight_decay = 5.0e-2
    optimizer = torch.optim.AdamW(model.parameters(),
        lr=train_lr, weight_decay=weight_decay)
