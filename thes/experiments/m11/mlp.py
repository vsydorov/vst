import copy
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from sklearn.metrics import (accuracy_score,)
from typing import (  # NOQA
        Dict, Any, Optional, List, cast, Tuple, TypedDict, Iterable, Literal)

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from vsydorov_tools import small

from thes.tools import snippets


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
from thes.data.tubes.routines import (
    get_dwein_overlaps_per_dgt,
    select_fg_bg_tubes,
    qload_synthetic_tube_labels,
    compute_flattube_syntlabel_acc,
    quick_assign_scores_to_dwein_tubes
)
from thes.data.dataset.daly import (
    Ncfg_daly, load_gt_and_wein_tubes,
    group_dwein_frames_wrt_kf_distance,
    sample_daly_frames_from_instances,
    Frame_labeled, Box_labeled,
    prepare_label_fullframes_for_training,
    prepare_label_roiboxes_for_training
)
from thes.mlp import (
    Ncfg_kfeats, E_tkfeats, create_preextracted_feats_manager,
    define_mlp_model, mlp_perf_kf_evaluate,
    mlp_perf_fulltube_evaluate, mlp_perf_display,
    quick_accuracy_over_kfeat,
    Manager_feats_tubes_dwein_full, Manager_feats_tubes_dwein_roipooled)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)
from thes.evaluation.recall import (
    compute_recall_for_avtubes_as_dfs,)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)

log = logging.getLogger(__name__)


def quick_tube_eval_spatiotemporal(
        av_stubes: AV_dict[T_dwein_scored],
        tubes_dgt: Dict[I_dgt, T_dgt],
        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Quick evaluation of daly tubes with default parameters
    """
    av_stubes_ = av_stubes_above_score(av_stubes, 0.0)
    av_stubes_ = compute_nms_for_av_stubes(av_stubes_, 0.3)
    iou_thresholds = [.3, .5, .7]
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(tubes_dgt)
    df_recall = compute_recall_for_avtubes_as_dfs(
        av_gt_tubes, av_stubes_, iou_thresholds, True)[0]
    df_ap = compute_ap_for_avtubes_as_df(
        av_gt_tubes, av_stubes_, iou_thresholds, True, False)
    return df_ap, df_recall


def mlp_perf_kf_evaluate_this(
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
    result['kacc_train'] = 0.0
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
        quick_tube_eval_spatiotemporal(av_stubes_cheat, tubes_dgt_eval)
    result.update({
        'df_recall_cheat': df_recall_cheat,
        'df_ap_cheat': df_ap_cheat})
    return result


def mlp_perf_fulltube_evaluate_this(
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


def evaluate_mlp_model(workfolder, cfg_dict, add_args):
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
    result = mlp_perf_kf_evaluate_this(
            model, None, tkfeats_eval,
            tubes_dwein_eval, tubes_dgt_eval,
            dataset, output_dims)
    # result_fulltube = mlp_perf_fulltube_evaluate_this(
    #         model, man_feats_dwt,
    #         tubes_dwein_eval, tubes_dwein_prov,
    #         tubes_dgt_eval, dwti_to_label_eval,
    #         dataset, output_dims,
    #         cf['eval.full_tubes.detect_mode'],
    #         cf['eval.full_tubes.nms'],
    #         cf['eval.full_tubes.field_nms'],
    #         cf['eval.full_tubes.field_det'])
    # result.update(result_fulltube)
    model.train()
    log.info(f'Evalset perf at {i_epoch=}')
    mlp_perf_display(result, sset_eval)
