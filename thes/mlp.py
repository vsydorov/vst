import logging
import copy
import numpy as np
from sklearn.preprocessing import (StandardScaler)
from sklearn.metrics import (accuracy_score,)
from pathlib import Path
from typing import (  # NOQA
        Dict, Any, Optional, List, cast, Tuple, TypedDict, Iterable, Literal)

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from vsydorov_tools import small

from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.evaluation.meta import (
    cheating_tube_scoring, quick_tube_eval,
    assign_scorefield, assign_scores_to_dwt_roipooled,
    assign_scores_to_dwt_fullframe)
from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    Frametube_scored,
    AV_dict, Box_connections_dwti,
    Tube_daly_wein_as_provided,
    av_stubes_above_score, push_into_avdict,
    get_daly_gt_tubes, dtindex_filter_split
)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.data.tubes.routines import (
    get_dwein_overlaps_per_dgt,
    select_fg_bg_tubes,
    qload_synthetic_tube_labels,
    compute_flattube_syntlabel_acc,
    quick_assign_scores_to_dwein_tubes
)
from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)

from thes.tools import snippets

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


class Manager_feats_tubes_dwein(object):
    pass

class Manager_feats_tubes_dwein_roipooled(Manager_feats_tubes_dwein):
    def __init__(self, tubes_dwein_feats_fold, scaler):
        self.scaler = scaler
        fold = Path(tubes_dwein_feats_fold)
        connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti] = \
                small.load_pkl(fold/'connections_f.pkl')
        box_inds2 = small.load_pkl(fold/'box_inds2.pkl')
        # DWTI -> Frame -> bi (big index)
        dwti_f_bi: Dict[I_dwein, Dict[int, int]]
        with small.QTimer('Creating dwti -> big index structure'):
            dwti_f_bi = {}
            for con, bi2 in zip(connections_f.values(), box_inds2):
                bi_range = np.arange(bi2[0], bi2[1])
                for dwti, bi in zip(con['dwti_sources'], bi_range):
                    dwti_f_bi.setdefault(dwti, {})[con['frame_ind']] = bi
        # Features
        with small.QTimer('big numpy load'):
            BIG = np.load(str(fold/"feats.npy"))
        self.connections_f = connections_f
        self.dwti_f_bi = dwti_f_bi
        self.BIG = BIG

    def get_all_tube_feats(self, dwti: I_dwein):
        """ Get all feats and optionall scale """
        inds_big = np.array(list(self.dwti_f_bi[dwti].values()))
        feats = self.BIG[inds_big]
        feats = feats.astype(np.float32)
        if self.scaler is not None:
            feats = self.scaler.transform(feats)
        feats = torch.from_numpy(feats)
        return feats


class Manager_feats_tubes_dwein_full(Manager_feats_tubes_dwein):
    def __init__(self, tubes_featfold, scaler):
        self.scaler = scaler
        tubes_featfold = Path(tubes_featfold)
        connections_f: Dict[
            Tuple[Vid_daly, int], Box_connections_dwti] = \
                small.load_pkl(tubes_featfold/'connections_f.pkl')
        fullframe = small.load_pkl(tubes_featfold/'fullframe.pkl')
        self.connections_f = connections_f
        self.fullframe_feats = fullframe


def create_preextracted_feats_manager(cf, scaler, detect_mode):
    kind = cf['inputs.tubes_dwein_feats.kind']
    fold = cf['inputs.tubes_dwein_feats.fold']
    if kind == 'roipooled':
        man_feats_dwt = \
            Manager_feats_tubes_dwein_roipooled(fold, scaler)
    elif kind == 'fullframe':
        man_feats_dwt = \
            Manager_feats_tubes_dwein_full(fold, scaler)
    else:
        raise RuntimeError()
    assert kind == detect_mode
    return man_feats_dwt


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


def define_mlp_model(cf, D_in, D_out):
    dropout_rate = cf['net.ll_dropout']
    if cf['net.kind'] == 'layer0':
        model = Net_mlp_zerolayer(
            D_in, D_out, dropout_rate)
    elif cf['net.kind'] == 'layer1':
        model = Net_mlp_onelayer(
            D_in, D_out, cf['net.layer1.H'], dropout_rate)
    return model


# Evaluation


def quick_accuracy_over_kfeat(
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


def mlp_perf_kf_evaluate(
        model,
        tkfeats_train: E_tkfeats,
        tkfeats_eval: E_tkfeats,
        tubes_dwein_eval: Dict[I_dwein, T_dwein],
        tubes_dgt_eval: Dict[I_dgt, T_dgt],
        dataset: Dataset_daly_ocv,
        output_dims: int,
            ) -> Dict:
    """
    This function should MLP KF evaluations
     + kacc_train, kacc_eval - runtime E_tkfeats perf
     + df_recall_cheat, df_ap_cheat - perf for cheating evaluation
    """
    assert not model.training, 'Wrong results in train mode'
    assert output_dims in (10, 11)
    kf_cut_last = output_dims == 11
    result = {}
    # // Quick acc
    result['kacc_train'] = quick_accuracy_over_kfeat(
            tkfeats_train, model, kf_cut_last)
    result['kacc_eval'] = quick_accuracy_over_kfeat(
            tkfeats_eval, model, kf_cut_last)
    # // Cheating MAP
    with torch.no_grad():
        Y_test_softmax = model(tkfeats_eval['X'])['x_final'].numpy()
    if kf_cut_last:
        Y_test_softmax = Y_test_softmax[:, :-1]
    assert Y_test_softmax.shape[-1] == 10
    # Quick confirm that test accuracy is the same
    preds = np.argmax(Y_test_softmax, axis=1)
    kf_acc = accuracy_score(tkfeats_eval['Y'], preds)
    assert np.isclose(result['kacc_eval'], kf_acc)
    # Perform cheating evaluation
    av_stubes_cheat: AV_dict[T_dwein_scored] = cheating_tube_scoring(
        Y_test_softmax, tkfeats_eval['kf'], tubes_dwein_eval, dataset)
    df_ap_cheat, df_recall_cheat = \
        quick_tube_eval(av_stubes_cheat, tubes_dgt_eval)
    result.update({
        'df_recall_cheat': df_recall_cheat,
        'df_ap_cheat': df_ap_cheat})
    return result

def mlp_perf_fulltube_evaluate(
        model,
        man_feats_dwt,
        tubes_dwein_eval: Dict[I_dwein, T_dwein],
        tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided],
        tubes_dgt_eval: Dict[I_dgt, T_dgt],
        dwti_to_label_eval: Dict[I_dwein, int],
        dataset: Dataset_daly_ocv,
        output_dims: int,
        # stats for fulltube eval
        f_detect_mode: Literal['roipooled', 'fullframe'],
        f_nms: float,
        f_field_nms: str,
        f_field_det: str,
        ) -> Dict:
    """
    This function should perform MLP fulltube evaluations
     + acc_flattube_synt - synthetic per-frame accuracy of dwein tubes
     + df_recall_full, df_ap_full - perf for full evaluation

    """
    iou_thresholds = [.3, .5, .7]
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(tubes_dgt_eval)

    acc_flattube_synt: float
    if f_detect_mode == 'roipooled':
        # // Full AUC (Evaluation of full wein-tubes with a trained model)
        assert isinstance(man_feats_dwt, Manager_feats_tubes_dwein_roipooled)
        tube_softmaxes_eval: Dict[I_dwein, np.ndarray] = {}
        model.eval()
        with torch.no_grad():
            for dwti in tubes_dwein_eval.keys():
                dwti_feats = man_feats_dwt.get_all_tube_feats(dwti)
                preds_softmax = model(dwti_feats)['x_final']
                tube_softmaxes_eval[dwti] = preds_softmax.numpy()
        # We need 2 kinds of tubes: with background and without
        if output_dims == 10:
            tube_softmaxes_eval_nobg = tube_softmaxes_eval
            tube_softmaxes_eval_bg = {k: np.pad(v, ((0, 0), (0, 1)))
                    for k, v in tube_softmaxes_eval.items()}
        else:
            tube_softmaxes_eval_nobg = {k: v[:, :-1]
                    for k, v in tube_softmaxes_eval.items()}
            tube_softmaxes_eval_bg = tube_softmaxes_eval
        # Flat accuracy (over synthetic labels, needs background)
        acc_flattube_synt = compute_flattube_syntlabel_acc(
                tube_softmaxes_eval_bg, dwti_to_label_eval)
        # Assign scores to tubes
        av_stubes_with_scores = assign_scores_to_dwt_roipooled(
                tubes_dwein_eval, tubes_dwein_prov,
                tube_softmaxes_eval_nobg, dataset)
    elif f_detect_mode =='fullframe':
        assert output_dims == 10
        # Aggregate frame scores
        assert isinstance(man_feats_dwt, Manager_feats_tubes_dwein_full)
        connections_f = man_feats_dwt.connections_f
        fullframe_feats = man_feats_dwt.fullframe_feats
        # model run
        model.eval()
        with torch.no_grad():
            t_fullframe_feats = torch.from_numpy(fullframe_feats)
            x_final = model(t_fullframe_feats)['x_final'].numpy()
        # Aggregate frame scores
        frame_scores: Dict[Tuple[Vid_daly, int], np.ndarray] = {}
        for cons, outputs_ in zip(connections_f.values(), x_final):
            vid = cons['vid']
            frame_ind = cons['frame_ind']
            frame_scores[(vid, frame_ind)] = outputs_
        assert len(frame_scores) == len(connections_f)
        # Flat accuracy not possible
        acc_flattube_synt = np.NAN
        # Assign scores to tubes
        av_stubes_with_scores = assign_scores_to_dwt_fullframe(
                tubes_dwein_eval, tubes_dwein_prov,
                frame_scores, dataset)
    else:
        raise RuntimeError()

    # Full evaluation
    av_stubes: Any = copy.deepcopy(av_stubes_with_scores)
    av_stubes = assign_scorefield(av_stubes, f_field_nms)
    av_stubes = av_stubes_above_score(av_stubes, 0.0)
    av_stubes = compute_nms_for_av_stubes(av_stubes, f_nms)
    av_stubes = assign_scorefield(av_stubes, f_field_det)

    df_recall_full = compute_recall_for_avtubes_as_dfs(
        av_gt_tubes, av_stubes, iou_thresholds, False)[0]
    df_ap_full = compute_ap_for_avtubes_as_df(
        av_gt_tubes, av_stubes, iou_thresholds, False, False)

    result = {
        'acc_flattube_synt': acc_flattube_synt,
        'df_recall_full': df_recall_full,
        'df_ap_full': df_ap_full}
    return result


def mlp_perf_display(result, sset_eval):
    df_recall_cheat = (result['df_recall_cheat']*100).round(2)
    df_ap_cheat = (result['df_ap_cheat']*100).round(2)
    cheating_apline = '/'.join(df_ap_cheat.loc['all'].values.astype(str))
    log.info('Keyframe classification results ({}): '.format(sset_eval))
    log.debug('Recall (cheating tubes) \n{}'.format(df_recall_cheat))
    log.debug('AP (cheating tubes):\n{}'.format(df_ap_cheat))
    kacc_train = result['kacc_train']*100
    kacc_eval = result['kacc_eval']*100
    log.info(' '.join((
        'kacc_train: {:.2f};'.format(kacc_train),
        'kacc_eval: {:.2f};'.format(kacc_eval))))
    log.info('Cheat AP357: {}'.format(cheating_apline))

    fields = [kacc_train, kacc_eval] + list(df_ap_cheat.loc['all'])

    df_ap_full = result.get('df_ap_full')
    if df_ap_full is not None:
        acc_flattube_synt = result['acc_flattube_synt']*100
        df_recall_full = (result['df_recall_full']*100).round(2)
        df_ap_full = (df_ap_full*100).round(2)
        apline = '/'.join(df_ap_full.loc['all'].values.astype(str))
        log.debug('Recall (full tubes) \n{}'.format(df_recall_full))
        log.debug('AP (full tubes):\n{}'.format(df_ap_full))
        log.info('Flattube synthetic acc: {:.2f}'.format(acc_flattube_synt))
        log.info('Full tube AP357: {}'.format(apline))
        fields += [acc_flattube_synt, *list(df_ap_full.loc['all'])]
    else:
        fields += [None] * 4

    fields = [f'{x:.2f}' if isinstance(x, float) else '' for x in fields]
    header = ['kacc_train', 'kacc_test', 'c3', 'c5', 'c7',
            's_acc', 'f3', 'f5', 'f7']
    log.info('\n{}'.format(snippets.table.string_table(
        [fields], header=header, pad=2)))
