import logging
import copy
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from typing import (  # NOQA
        Dict, Any, Optional, List, cast, Tuple, TypedDict, Iterable)

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from vsydorov_tools import small, cv as vt_cv
from vsydorov_tools.small import mkdir as mkd

from thes.data.tubes.types import (
    I_dwein, T_dwein, T_dwein_scored, I_dgt, T_dgt,
    AV_dict, Box_connections_dwti,
    push_into_avdict, av_stubes_above_score,
    Tube_daly_wein_as_provided
)
from thes.data.tubes.routines import (
    numpy_inner_overlap_NN, _bareas,
    quick_assign_scores_to_dwein_tubes
)
from thes.evaluation.ap.convert import (
    compute_ap_for_avtubes_as_df)
from thes.data.tubes.nms import (
    compute_nms_for_av_stubes,)

from thes.data.dataset.external import (
    Dataset_daly_ocv, Dataset_charades_ocv)
from thes.data.dataset.daly import (
    Ncfg_daly, load_gt_and_wein_tubes,
    group_dwein_frames_wrt_kf_distance)
from thes.tools import snippets
from thes.experiments.m06.mlp import (
    Ncfg_kfeats, define_mlp_model,
    _quick_accuracy_over_kfeat,
    _predict_softmaxes_for_dwein_tubes_in_da_big)


log = logging.getLogger(__name__)


def get_n_colormaps(n):
    starting_colors = sns.color_palette('hls', n)
    palettes = [sns.dark_palette(col, as_cmap=True) for col in starting_colors]
    return palettes


def get_n_colors(n):
    n_colormaps = get_n_colormaps(n)
    n_colors = []
    for colormap in n_colormaps:
        color = (np.array(colormap(0.95))[:3]*255).astype(int).tolist()
        n_colors.append(color)
    return n_colors


def _plot_close(
        f, legend_list=[], imgpath=None):
  if imgpath:
    f.savefig(str(imgpath),
              bbox_extra_artists=legend_list,
              bbox_inches='tight')
  else:
    plt.show()
  plt.close(f)


def charades_stats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset:
        cache_folder: [~, str]
        charades:
            mirror: ['gpuhost7', str]
            resolution: ['480', str]
    """)
    cf = cfg.parse()
    mirror = cf['dataset.charades.mirror']
    resolution = cf['dataset.charades.resolution']
    dataset = Dataset_charades_ocv(mirror, resolution)
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    pd_vs = pd.DataFrame(dataset.videos_ocv).T
    s_split = pd.Series(dataset.split)
    pd_vs['split'] = s_split
    pd_vs['action_labels'] = pd_vs.actions.apply(
        lambda x: [y['name'] for y in x])
    train_mask = pd_vs['split'] == 'train'
    val_mask = pd_vs['split'] == 'val'
    pd_vs_train = pd_vs[train_mask]
    pd_vs_val = pd_vs[val_mask]

    N_classes = len(dataset.action_names)
    log.info(f'N classes: {N_classes}')
    log.info('N videos (Total/Train/Val): {}/{}/{}'.format(
        len(pd_vs), len(pd_vs_train), len(pd_vs_val)))
    log.info('N frames (Total/Train/Val): {}/{}/{}'.format(
        pd_vs.nframes.sum(), pd_vs_train.nframes.sum(),
        pd_vs_val.nframes.sum()))

    l_ = []
    for i, a in enumerate(dataset.action_names):
        s = pd_vs.action_labels.apply(lambda x: a in x)
        s.name = i
        l_.append(s)
    action_presence = pd.concat(l_, axis=1)
    l_ = []
    for i, o in enumerate(dataset.object_names):
        s = pd_vs.objects.apply(lambda x: o in x)
        s.name = i
        l_.append(s)
    obj_presence = pd.concat(l_, axis=1)
    action_mapping_int = {
        dataset.action_names.index(a): dataset.object_names.index(o)
        for a, (o, v) in dataset.action_mapping.items()}
    column_mapped_obj = {}
    for i, column in action_presence.iteritems():
        column_mapped_obj.setdefault(action_mapping_int[i], []).append(column)
    obj_presence_mapped = pd.DataFrame({
            k: pd.concat(v, axis=1).any(axis=1)
            for k, v in column_mapped_obj.items()}).sort_index(axis=1)
    # 0/'None' object to be removed
    obj_presence0 = obj_presence.loc[:, 1:]
    obj_presence_mapped0 = obj_presence_mapped.loc[:, 1:]

    def frac_str(X, Y):
        return f'{X}/{Y}/{X/Y*100:.2f}%'

    def mean_frac(X):
        return frac_str(X.sum(), len(X))

    X_extra_present = ((obj_presence0 > obj_presence_mapped0).any(axis=1))
    log.info('Videos have extra objects present, beyond mapped: '
        '[Total: {}] [Train: {}] [Val: {}]'.format(
            mean_frac(X_extra_present),
            mean_frac(X_extra_present[train_mask]),
            mean_frac(X_extra_present[val_mask])))

    X_impobj_missing = (obj_presence0 < obj_presence_mapped0).any(axis=1)
    log.info('Videos have objects missing, though they are implied by mapping: '
        '[Total: {}] [Train: {}] [Val: {}]'.format(
            mean_frac(X_impobj_missing),
            mean_frac(X_impobj_missing[train_mask]),
            mean_frac(X_impobj_missing[val_mask])))

    def produce_map_annot(obj_presence_mapped0, obj_presence0):
        X = (obj_presence_mapped0 & obj_presence0).sum()
        XX = obj_presence_mapped0.sum()
        present_when_mapped = pd.concat((X/XX*100, X, XX), axis=1)
        present_when_mapped.index = dataset.object_names[1:]
        present_when_mapped.columns = ['%', 'matched', 'total_mapped']
        return present_when_mapped

    X = produce_map_annot(obj_presence_mapped0, obj_presence0)
    X_train = produce_map_annot(
            obj_presence_mapped0[train_mask], obj_presence0[train_mask])
    X_val = produce_map_annot(
            obj_presence_mapped0[val_mask], obj_presence0[val_mask])
    log.info('Videos per cls, where mapped object is annotated:\n{}'.format(
        pd.concat((X, X_train, X_val), keys=['total', 'train', 'val'], axis=1)))

    # Per-instance stats
    instances_ = {}
    for vid, video in dataset.videos_ocv.items():
        for ii, instance in enumerate(video['actions']):
            instance_ = copy.copy(instance)
            instance_['split'] = dataset.split[vid]
            instances_[(vid, ii)] = instance_
    pd_inst = pd.DataFrame(instances_).T
    pd_inst['nframes'] = pd_inst['end_frame'] - pd_inst['start_frame'] + 1
    pd_inst['action_ind'] = pd_inst.name.apply(
            lambda x: dataset.action_names.index(x))
    train_mask_i = pd_inst['split'] == 'train'
    val_mask_i = pd_inst['split'] == 'val'
    nframes_per_cls = pd_inst.groupby('action_ind').nframes.sum()
    log.info(f'N frames per cls, summed (total): {nframes_per_cls.sum()}')

    nframes_per_cls_train = (pd_inst[train_mask_i]
            .groupby('action_ind').nframes.sum())
    nframes_per_cls_val = (pd_inst[val_mask_i]
            .groupby('action_ind').nframes.sum())

    X = pd.concat((
        nframes_per_cls, nframes_per_cls_train, nframes_per_cls_val),
        axis=1, keys=['total', 'train', 'val'])
    log.info('Describe frames per cls:\n{}'.format(X.describe()))

    # Frame coverage
    fcoverage = {vid: np.zeros(row['nframes'], dtype=np.int)
            for vid, row in pd_vs.iterrows()}
    for (vid, i), row in pd_inst.iterrows():
        s = row['start_frame']
        e = row['end_frame']
        fcoverage[vid][s:e+1] += 1
    coverage_per_vid = {}
    for vid, cov in fcoverage.items():
        d = {
            'nframes': len(cov),
            'covered': (cov>0).sum(),
            'annos': cov.sum()}
        coverage_per_vid[vid] = d
    df_coverage = pd.DataFrame(coverage_per_vid).T
    df_coverage['frac'] = df_coverage['covered']/df_coverage['nframes']

    def anno_frac(X):
        return frac_str(X.covered.sum(), X.nframes.sum())

    def avg_anno_vf_frac(X):
        return frac_str(X.annos.sum(), X.nframes.sum())

    def avg_anno_af_frac(X):
        return frac_str(X.annos.sum(), X.covered.sum())

    log.info('Annotated fraction (per-vid): '
            '[Total: {:.2f}%] [Train: {:.2f}%] [Val: {:.2f}%]'.format(
                df_coverage.frac.mean()*100,
                df_coverage[train_mask].frac.mean()*100,
                df_coverage[val_mask].frac.mean()*100))
    log.info('Annotated fraction: '
            '[Total: {}] [Train: {}] [Val: {}]'.format(
                anno_frac(df_coverage),
                anno_frac(df_coverage[train_mask]),
                anno_frac(df_coverage[val_mask])))
    log.info('Avg. N annotation per video frame: '
            '[Total: {}] [Train: {}] [Val: {}]'.format(
                avg_anno_vf_frac(df_coverage),
                avg_anno_vf_frac(df_coverage[train_mask]),
                avg_anno_vf_frac(df_coverage[val_mask])))
    log.info('Avg. N annotation per annotated video frame: '
            '[Total: {}] [Train: {}] [Val: {}]'.format(
                avg_anno_af_frac(df_coverage),
                avg_anno_af_frac(df_coverage[train_mask]),
                avg_anno_af_frac(df_coverage[val_mask])))


class Daly_stats:

    @staticmethod
    def pd_stats(dataset):
        pd_vs = pd.DataFrame(dataset.videos_ocv).T
        s_split = pd.Series(dataset.split)
        pd_vs['split'] = s_split
        train_mask = pd_vs['split'] == 'train'
        test_mask = pd_vs['split'] == 'test'
        pd_vs_train = pd_vs[train_mask]
        pd_vs_test = pd_vs[test_mask]

        N_classes = len(dataset.action_names)
        log.info(f'N classes: {N_classes}')
        log.info('N videos (Total/Train/Val): {}/{}/{}'.format(
            len(pd_vs), len(pd_vs_train), len(pd_vs_test)))
        log.info('N frames (Total/Train/Val): {}/{}/{}'.format(
            pd_vs.nframes.sum(), pd_vs_train.nframes.sum(),
            pd_vs_test.nframes.sum()))
        log.info('Duration, hours (Total/Train/Val): {:.2f}/{:.2f}/{:.2f}'.format(
            pd_vs.est_length.sum()/3600,
            pd_vs_train.est_length.sum()/3600,
            pd_vs_test.est_length.sum()/3600))

        # Per-instance stats
        action_list_ = []
        for vid, video in dataset.videos_ocv.items():
            for action_name, ains in video['instances'].items():
                for ins_ind, instance in enumerate(ains):
                    actitem = {}
                    actitem.update({
                        'beginTime': instance['beginTime'],
                        'endTime': instance['endTime'],
                        'start_frame': instance['start_frame'],
                        'end_frame': instance['end_frame']
                        })
                    actitem.update({
                        'vid': vid,
                        'action_name': action_name,
                        'ins_ind': ins_ind,
                        'split': dataset.split[vid]
                        })
                    action_list_.append(actitem)
        pd_inst = pd.DataFrame(action_list_)
        pd_inst['duration'] = pd_inst['endTime'] - pd_inst['beginTime']
        pd_inst['nframes'] = pd_inst['end_frame'] - pd_inst['start_frame'] + 1
        train_mask_i = pd_inst['split'] == 'train'
        test_mask_i = pd_inst['split'] == 'test'

        log.info('Annotated action duration, hours: {:.2f}/{:.2f}/{:.2f}'.format(
            pd_inst.duration.sum()/3600,
            pd_inst[train_mask_i].duration.sum()/3600,
            pd_inst[test_mask_i].duration.sum()/3600,))
        IF_pcls_total = pd_inst.groupby('action_name').nframes.sum()
        IF_pcls_train = pd_inst[train_mask_i].groupby('action_name').nframes.sum()
        IF_pcls_test = pd_inst[test_mask_i].groupby('action_name').nframes.sum()
        IF_pcls = pd.concat([IF_pcls_total, IF_pcls_train, IF_pcls_test], axis=1,
                keys=['total', 'train', 'test'])
        IF_pcls.loc['TOTAL'] = IF_pcls.sum()
        log.info('Annotated action duration, frames:\n{}'.format(IF_pcls))

        # Per-keyframe stats
        kf_list_ = []
        for vid, video in dataset.videos_ocv.items():
            for action_name, ains in video['instances'].items():
                for ins_ind, instance in enumerate(ains):
                    for kf_ind, kf in enumerate(instance['keyframes']):
                        kfitem = {}
                        kfitem.update({
                            'time': kf['time'],
                            'frameNumber': kf['frameNumber'],
                            'vid': vid,
                            'action_name': action_name,
                            'ins_ind': ins_ind,
                            'kf_ind': kf_ind,
                            'inst_bbox': kf['boundingBox'][0],
                            'height': video['height'],
                            'width': video['width'],
                            'split': dataset.split[vid]
                            })
                        kf_list_.append(kfitem)
        pd_kf = pd.DataFrame(kf_list_)
        train_mask_k = pd_kf['split'] == 'train'
        test_mask_k = pd_kf['split'] == 'test'

        KF_pcls_total = pd_kf.groupby('action_name').vid.count()
        KF_pcls_train = pd_kf[train_mask_k].groupby('action_name').vid.count()
        KF_pcls_test = pd_kf[test_mask_k].groupby('action_name').vid.count()
        KF_pcls = pd.concat([KF_pcls_total, KF_pcls_train, KF_pcls_test],
                axis=1, keys=['total', 'train', 'test'])
        KF_pcls.loc['TOTAL'] = KF_pcls.sum()
        log.info('Annotated keyframes (per cls):\n{}'.format(
            KF_pcls))

        # Per-object stats
        obj_list_ = []
        for vid, video in dataset.videos_ocv.items():
            for action_name, ains in video['instances'].items():
                for ins_ind, instance in enumerate(ains):
                    for kf_ind, kf in enumerate(instance['keyframes']):
                        kf_objects = kf['objects']
                        for o_ind, kfo in enumerate(kf_objects):
                            [xmin, ymin, xmax, ymax,
                                objectID, isOccluded, isHallucinate] = kfo
                            isOccluded = bool(isOccluded)
                            isHallucinate = bool(isHallucinate)
                            obj_bbox = np.array([xmin, ymin, xmax, ymax])
                            objectID = int(objectID)
                            objitem = {}
                            objitem.update({
                                'objectID': objectID,
                                'object_name': dataset.object_names[objectID],
                                'obj_bbox': obj_bbox,
                                'isOccluded': isOccluded,
                                'isHallucinate': isHallucinate,
                                })
                            objitem.update({
                                'time': kf['time'],
                                'frame': kf['frame'],
                                'vid': vid,
                                'action_name': action_name,
                                'ins_ind': ins_ind,
                                'kf_ind': kf_ind,
                                'o_ind': o_ind,
                                'inst_bbox': kf['boundingBox'][0],
                                'height': video['height'],
                                'width': video['width'],
                                'split': dataset.split[vid]
                                })
                            obj_list_.append(objitem)
        pd_obj = pd.DataFrame(obj_list_)
        train_mask_o = pd_obj['split'] == 'train'
        test_mask_o = pd_obj['split'] == 'test'

        OBJ_pcls_total = pd_obj.groupby('object_name').vid.count()
        OBJ_pcls_train = pd_obj[train_mask_o].groupby('object_name').vid.count()
        OBJ_pcls_test = pd_obj[test_mask_o].groupby('object_name').vid.count()
        OBJ_pcls = pd.concat([OBJ_pcls_total, OBJ_pcls_train, OBJ_pcls_test],
                axis=1, keys=['total', 'train', 'test'])
        OBJ_pcls.loc['TOTAL'] = OBJ_pcls.sum()
        log.info('Annotated object locations (per cls):\n{}'.format(
            OBJ_pcls))

        # Intersection stats
        # Frame coverage
        fcoverage = {vid: np.zeros(row['nframes'], dtype=np.int)
                for vid, row in pd_vs.iterrows()}
        for ii, row in pd_inst.iterrows():
            s = row['start_frame']
            e = row['end_frame']
            vid = row['vid']
            fcoverage[vid][s:e+1] += 1
        coverage_per_vid = {}
        for vid, cov in fcoverage.items():
            d = {
                'nframes': len(cov),
                'covered': (cov>0).sum(),
                'annos': cov.sum()}
            coverage_per_vid[vid] = d
        df_coverage = pd.DataFrame(coverage_per_vid).T
        df_coverage['frac'] = df_coverage['covered']/df_coverage['nframes']

        def frac_str(X, Y):
            return f'{X}/{Y}/{X/Y*100:.2f}%'

        def anno_frac(X):
            return frac_str(X.covered.sum(), X.nframes.sum())

        def avg_anno_vf_frac(X):
            return frac_str(X.annos.sum(), X.nframes.sum())

        def avg_anno_af_frac(X):
            return frac_str(X.annos.sum(), X.covered.sum())

        log.info('Annotated fraction (per-vid): '
                '[Total: {:.2f}%] [Train: {:.2f}%] [Val: {:.2f}%]'.format(
                    df_coverage.frac.mean()*100,
                    df_coverage[train_mask].frac.mean()*100,
                    df_coverage[test_mask].frac.mean()*100))
        log.info('Annotated fraction: '
                '[Total: {}] [Train: {}] [Val: {}]'.format(
                    anno_frac(df_coverage),
                    anno_frac(df_coverage[train_mask]),
                    anno_frac(df_coverage[test_mask])))
        log.info('Avg. N annotation per video frame: '
                '[Total: {}] [Train: {}] [Val: {}]'.format(
                    avg_anno_vf_frac(df_coverage),
                    avg_anno_vf_frac(df_coverage[train_mask]),
                    avg_anno_vf_frac(df_coverage[test_mask])))
        log.info('Avg. N annotation per annotated video frame: '
                '[Total: {}] [Train: {}] [Val: {}]'.format(
                    avg_anno_af_frac(df_coverage),
                    avg_anno_af_frac(df_coverage[train_mask]),
                    avg_anno_af_frac(df_coverage[test_mask])))
        return pd_kf, pd_obj

    @staticmethod
    def inst_bbox_stats_hists(pd_kf, out):
        # Bounding box stats (for keyframe labels)
        np_ltrd = np.array(pd_kf.inst_bbox.tolist())
        np_WH = pd_kf[['width', 'height']].to_numpy()

        l, t, r, d = np_ltrd.T
        W, H = np_WH.T
        h = d - t
        w = r -l
        area = h*w
        pd_kf['area'] = area
        pd_kf['h_by_w'] = h/w
        mean_kf_area = pd_kf.groupby('action_name').area.mean()
        log.info('KF area, mean: {:.2f}%, per action:\n{}'.format(
            np.mean(area)*100, mean_kf_area*100))

        mean_h_by_w = pd_kf.groupby('action_name').h_by_w.mean()
        log.info('Heigh/Width ratio, {:.2f}, per action:\n{}'.format(
            np.mean(h/w), mean_h_by_w))

        # H by W
        fold = mkd(out/'h_by_w')

        def plot_hist(X, imgpath, bins, color=None):
            f, ax = plt.subplots(figsize=(20, 10), dpi=160)
            ax.hist(X, bins=bins, density=True, color=color)
            WO = ((X>max(bins)) | (X<min(bins))).mean()
            ax.set_xticks(bins)
            ax.set_title(f'Outside hist: {WO*100:.2f}%')
            _plot_close(f, imgpath=imgpath)

        imgpath = fold/'ALL_dhist.png'
        bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]
        plot_hist(pd_kf.h_by_w, imgpath, bins)

        imgpath = fold/'ALL_stacked.png'
        f, ax = plt.subplots(figsize=(20, 10), dpi=160)
        As, Xs = zip(*pd_kf.groupby('action_name').h_by_w)
        ax.hist(Xs, bins=bins, density=True, stacked=True, label=As)
        ax.set_xticks(bins)
        ax.legend()
        _plot_close(f, imgpath=imgpath)

        defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (action, group) in enumerate(pd_kf.groupby('action_name').h_by_w):
            imgpath = fold/f'{i}_{action}_dhist.png'
            color = defcolors[i]
            plot_hist(group, imgpath, bins, color)

        # Area
        fold = mkd(out/'area')
        imgpath = fold/'ALL_dhist.png'
        f, ax = plt.subplots(figsize=(20, 10), dpi=160)
        ax.hist(pd_kf.area, density=True)
        _plot_close(f, imgpath=imgpath)

        imgpath = fold/'ALL_stacked.png'
        f, ax = plt.subplots(figsize=(20, 10), dpi=160)
        As, Xs = zip(*pd_kf.groupby('action_name').area)
        ax.hist(Xs, density=True, stacked=True, label=As)
        ax.legend()
        _plot_close(f, imgpath=imgpath)

    @staticmethod
    def inst_bbox_stats_hmap(pd_kf, out, dataset):

        # Bbox heatmap
        # ocv image: rows, columns, channels
        # mplt image: rows, columns, channeels

        np_ltrd = np.array(pd_kf.inst_bbox.tolist())
        np_WH = pd_kf[['width', 'height']].to_numpy()
        W, H = np_WH.T

        # Assign the same heigh H = 480
        W_max = 1520
        H_max = 480
        scale_multiplier = H_max/H
        np_WH_scaled = np_WH * scale_multiplier[:, None]
        W_shift = (W_max - np_WH_scaled[:, 0])/2
        W_shift_4 = np.tile(np.c_[W_shift, np.zeros_like(W_shift)], (1, 2))
        boxes = np_ltrd * np.tile(np_WH_scaled, (1, 2))
        boxes_shifted = boxes + W_shift_4
        int_boxes = np.round(boxes_shifted).astype(int)

        fold = mkd(out/'box_heatmap_hfixed')

        aX = {a: np.zeros([W_max, H_max], dtype=np.int)
                for a in dataset.action_names}
        for i, box in enumerate(int_boxes):
            a = pd_kf.iloc[i]['action_name']
            l, t, r, d = box
            aX[a][l:r, t:d] += 1
        allX = np.dstack(list(aX.values())).sum(2)

        fold0 = mkd(fold/'mpl')

        def mpl_plot(aX, allX, fold0):
            for i, (a, X) in enumerate(aX.items()):
                imgpath = fold0/f'{i}_{a}.png'
                f, ax = plt.subplots(figsize=(20, 10), dpi=160)
                ax.imshow(X.T)
                _plot_close(f, imgpath=imgpath)

            imgpath = fold0/'all.png'
            f, ax = plt.subplots(figsize=(20, 10), dpi=160)
            ax.imshow(allX.T)
            _plot_close(f, imgpath=imgpath)

        fold0 = mkd(fold/'ocv')

        def ocv_hmap(X):
            X_ocv = ((X)/np.max(X)*255).astype(np.uint8)
            X_ocv = cv2.applyColorMap(X_ocv, cv2.COLORMAP_JET)
            return X_ocv

        for i, (a, X) in enumerate(aX.items()):
            imgpath = str(fold0/f'{i}_{a}_h.jpg')
            cv2.imwrite(str(imgpath), ocv_hmap(X.T))
        imgpath = str(fold0/'all.jpg')
        cv2.imwrite(str(imgpath), ocv_hmap(allX.T))

        fold = mkd(out/'box_heatmap_100')

        # Assign 100x100 grid
        boxes = np_ltrd * np.tile(100, (1, 4))
        int_boxes = np.round(boxes).astype(int)
        aX = {a: np.zeros([100, 100], dtype=np.int)
                for a in dataset.action_names}
        for i, box in enumerate(int_boxes):
            a = pd_kf.iloc[i]['action_name']
            l, t, r, d = box
            aX[a][l:r, t:d] += 1
        allX = np.dstack(list(aX.values())).sum(2)

        mpl_plot(aX, allX, mkd(fold/'mpl'))

    def obj_inst_bbox_stats(pd_obj, out, dataset):
        # one box is broken, exclude it
        broken_id = np.where(_bareas(
            np.array(pd_obj.obj_bbox.tolist())) == 0)[0][0]
        assert broken_id == 4220
        pd_obj_ = pd_obj.drop(4220)

        # How many objects lay outside instance boxes?
        O_box = np.array(pd_obj_.obj_bbox.tolist())
        I_box = np.array(pd_obj_.inst_bbox.tolist())

        obj_uncovered = 1 - numpy_inner_overlap_NN(O_box, I_box)
        pd_obj_['obj_uncovered'] = obj_uncovered

        fold = mkd(out/'obj_covered')

        imgpath = fold/'ALL_dhist.png'
        f, ax = plt.subplots(figsize=(20, 10), dpi=160)
        ax.hist(obj_uncovered, density=True)
        _plot_close(f, imgpath=imgpath)

        defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (action, group) in enumerate(
                pd_obj_.groupby('action_name').obj_uncovered):
            imgpath = fold/f'{i}_{action}_dhist.png'
            f, ax = plt.subplots(figsize=(20, 10), dpi=160)
            color = defcolors[i]
            ax.hist(group, density=True, color=color)
            _plot_close(f, imgpath=imgpath)

        imgpath = fold/'ALL_stacked.png'
        f, ax = plt.subplots(figsize=(20, 10), dpi=160)
        As, Xs = zip(*pd_obj_.groupby('action_name').obj_uncovered)
        ax.hist(Xs, density=True, stacked=True, label=As)
        ax.legend()
        _plot_close(f, imgpath=imgpath)

        # Relative locations (assume instance box is the new 0..1 coord basis)
        I_wh = I_box[:, 2:] - I_box[:, :2]
        O_rbox = O_box - np.tile(I_box[:, :2], (1, 2))
        O_rbox /= np.tile(I_wh, (1, 2))

        # Change basis to 100,100 box in the middle
        O_rbox_int = O_rbox * np.tile(100, (1, 4))
        O_rbox_int += np.tile(50, (1, 4))
        O_rbox_int = np.clip(O_rbox_int, 0, 200)
        O_rbox_int = np.round(O_rbox_int).astype(int)
        aX = {a: np.zeros([200, 200], dtype=np.int)
                for a in dataset.action_names}
        for i, box in enumerate(O_rbox_int):
            a = pd_obj_.iloc[i]['action_name']
            l, t, r, d = box
            aX[a][l:r, t:d] += 1
        allX = np.dstack(list(aX.values())).sum(2)

        fold0 = mkd(fold/'ocv')

        def ocv_hmap(X):
            X_ocv = ((X)/np.max(X)*255).astype(np.uint8)
            X_ocv = cv2.applyColorMap(X_ocv, cv2.COLORMAP_JET)
            cv2.rectangle(X_ocv, (50, 50), (150, 150), (30, 30, 255), 2)
            return X_ocv

        for i, (a, X) in enumerate(aX.items()):
            imgpath = str(fold0/f'{i}_{a}_h.jpg')
            cv2.imwrite(str(imgpath), ocv_hmap(X.T))
        imgpath = str(fold0/'all.jpg')
        cv2.imwrite(str(imgpath), ocv_hmap(allX.T))

def vidb_stats():
    # Check vid correspondencies
    eval_vid_bunches = sorted(list(set([(a, b) for (a, b, c) in tubes_dwein_eval])))
    best_idx_per_vidb = {}
    for i, vid_bunch in enumerate(eval_vid_bunches):
        for a, v_stubes in av_stubes_.items():
            stubes = v_stubes.get(vid_bunch[0])
            stubes_bunch = []
            for stube in stubes:
                if stube['index'][:2] == vid_bunch:
                    stubes_bunch.append(stube)
            o_stubes = sorted(stubes_bunch, key=lambda x: x['score'])[::-1]
            if len(o_stubes):
                best_idx = o_stubes[0]['index']
            else:
                best_idx = None
            best_idx_per_vidb.setdefault(vid_bunch, {})[a] = best_idx
    # Compute how many good
    agree_per_vidb = {}
    for vidb, best_idx in best_idx_per_vidb.items():
        good_cls = dataset.videos_ocv[vidb[0]]['suggestedClass']
        good_ind = best_idx[good_cls]
        agree_w_good = []
        for v in best_idx.values():
            agree_w_good.append(v == good_ind)
        agree_w_good = np.sum(agree_w_good)
        agree_per_vidb[vidb] = agree_w_good
    s = pd.Series(agree_per_vidb)


class Manager_big_simple(object):
    def __init__(self, tubes_featfold):
        tubes_featfold = Path(tubes_featfold)
        connections_f = small.load_pkl(tubes_featfold/'connections_f.pkl')
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

    def get(self, _, dwti):
        inds_big = np.array(list(self.dwti_f_bi[dwti].values()))
        feats = self.BIG[inds_big]
        feats = feats.astype(np.float32)
        feats = torch.from_numpy(feats)
        return feats

# Experiments

def daly_oldstats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.yconfig.YConfig_v2(cfg_dict, allowed_wo_defaults=[''])
    cfg.set_defaults_yaml("""
    dataset:
        cache_folder: ~
        mirror: 'uname'
    """)
    cf = cfg.parse()

    dataset = Dataset_daly_ocv(cf['dataset.mirror'])
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    pd_kf, pd_obj = Daly_stats.pd_stats(dataset)

    Daly_stats.inst_bbox_stats_hists(pd_kf, out)
    Daly_stats.inst_bbox_stats_hmap(pd_kf, out, dataset)
    Daly_stats.obj_inst_bbox_stats(pd_obj, out, dataset)

def daly_map_explore(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.yconfig.YConfig_v2(cfg_dict, allowed_wo_defaults=[''])
    cfg.set_defaults_yaml("""
    dataset:
        cache_folder: ~
        mirror: 'uname'
    split_assignment: 'train/val'
    inputs:
        tubes_dwein: ~
        keyframes:
            fold: ~
            featname: 'roipooled'
        big:
          fold: ~
        trained_mlp_roi: ~
    net:
        kind: 'layer1'
        layer1:
            H: 32
        ll_dropout: 0.5
    seed: 0
    data_scaler: 'no'
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
    tubes_dwein_prov: Dict[I_dwein, Tube_daly_wein_as_provided] = \
            small.load_pkl(cf['inputs.tubes_dwein'])

    # Sset
    tkfeats_train = tkfeats_d[sset_train]
    tkfeats_eval = tkfeats_d[sset_eval]
    tubes_dwein_train = tubes_dwein_d[sset_train]
    tubes_dwein_eval = tubes_dwein_d[sset_eval]
    tubes_dgt_train = tubes_dgt_d[sset_train]
    tubes_dgt_eval = tubes_dgt_d[sset_eval]

    # Big manager
    man_big = Manager_big_simple(cf['inputs.big.fold'])
    # Load create model
    D_in = man_big.BIG.shape[-1]
    model = define_mlp_model(cf, D_in, 11)
    states = torch.load(cf['inputs.trained_mlp_roi'])
    model.load_state_dict(states['model_sdict'])
    model.eval()

    # quick accc
    kf_cut_last = True
    result = {}
    result['kacc_train'] = _quick_accuracy_over_kfeat(
            tkfeats_train, model, kf_cut_last)
    result['kacc_eval'] = _quick_accuracy_over_kfeat(
            tkfeats_eval, model, kf_cut_last)

    # // Full AUC (Evaluation of full wein-tubes with a trained model)
    tube_softmaxes_eval: Dict[I_dwein, np.ndarray] = \
        _predict_softmaxes_for_dwein_tubes_in_da_big(
            model, man_big, tubes_dwein_eval.keys())
    tube_softmaxes_eval_nobg = {k: v[:, :-1]
            for k, v in tube_softmaxes_eval.items()}

    iou_thresholds = [.3, .5, .7]
    av_gt_tubes: AV_dict[T_dgt] = push_into_avdict(tubes_dgt_eval)

    # Universal detector experiments
    av_stubes_eval_augm: AV_dict[T_dwein_scored] = {}
    for dwt_index, tube in tubes_dwein_eval.items():
        softmaxes = tube_softmaxes_eval_nobg[dwt_index]
        scores = softmaxes.mean(axis=0)
        hscores = tubes_dwein_prov[dwt_index]['hscores']
        iscores = tubes_dwein_prov[dwt_index]['iscores']
        (vid, bunch_id, tube_id) = dwt_index
        for action_name, score in zip(dataset.action_names, scores):
            stube = cast(T_dwein_scored, tube.copy())
            stube['cls_score'] = score
            stube['univ_score'] = scores.sum()
            stube['hscore'] = hscores.mean()
            stube['iscore'] = np.nanmean(iscores)
            stube = cast(T_dwein_scored, stube)
            (av_stubes_eval_augm
                    .setdefault(action_name, {})
                    .setdefault(vid, []).append(stube))

    def assign_scorefield(av_stubes, score_field):
        for a, v_stubes in av_stubes.items():
            for v, stubes in v_stubes.items():
                for stube in stubes:
                    stube['score'] = stube[score_field]

    def specific_nms_exp(av_stubes, av_gt_tubes, score_field):
        assign_scorefield(av_stubes, score_field)
        av_stubes = av_stubes_above_score(av_stubes, 0.0)
        av_stubes = compute_nms_for_av_stubes(av_stubes, 0.3)
        assign_scorefield(av_stubes, 'cls_score')
        df_ap = compute_ap_for_avtubes_as_df(
            av_gt_tubes, av_stubes, iou_thresholds, False, False)
        return df_ap, av_stubes

    dfap_per_nmsfield = {}
    av_stubes_after_per_nmsfield = {}
    for score_field in ['cls_score', 'univ_score', 'hscore', 'iscore']:
        av_stubes = copy.deepcopy(av_stubes_eval_augm)
        df_ap, av_stubes_after = specific_nms_exp(av_stubes, av_gt_tubes, score_field)
        dfap_per_nmsfield[score_field] = df_ap
        av_stubes_after_per_nmsfield[score_field] = av_stubes_after
    x = pd.concat(dfap_per_nmsfield)[0.5].unstack(level=0)*100
    log.info('AP5 per NMSfield:\n{}'.format(snippets.df_to_table_v2(x)))

    def box_overlap_stats(score_field, av_stubes_after):
        vidfc_per_action = {}
        for a, v_stubes in av_stubes_after.items():
            vidf = {}  # frame groups
            for v, stubes in v_stubes.items():
                for stube in stubes:
                    for frame_ind in stube['frame_inds']:
                        vidf.setdefault((v, frame_ind), []).append(stube['index'])
            vidfc = pd.Series({k: len(v) for k, v in vidf.items()})
            vidfc_per_action[a] = vidfc
        s_eq1 = (pd.concat(vidfc_per_action)==1).unstack(level=0).mean()
        s_mean = pd.concat(vidfc_per_action).unstack(level=0).mean()
        df_boxoverlap = pd.concat((s_eq1, s_mean), axis=1, keys=['equal1', 'avg']).T
        log.info('Boxoverl for "{}"\n{}'.format(
            score_field, snippets.df_to_table_v2(df_boxoverlap)))

    # Record instances when same frame has multiple boxes (per class)
    for score_field, av_stubes_after in av_stubes_after_per_nmsfield.items():
        box_overlap_stats(score_field, av_stubes_after)
    box_overlap_stats(score_field, av_stubes_eval_augm)
