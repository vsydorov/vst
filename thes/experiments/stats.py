import logging
import copy
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


from vsydorov_tools import small, cv as vt_cv
from vsydorov_tools.small import mkdir as mkd

from thes.data.dataset.external import (
        Dataset_daly_ocv, Dataset_charades_ocv)
from thes.data.tubes.routines import (
        numpy_inner_overlap_NN, _bareas)
from thes.tools import snippets


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
        imgpath = str(fold0/f'all.jpg')
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
        imgpath = str(fold0/f'all.jpg')
        cv2.imwrite(str(imgpath), ocv_hmap(allX.T))


def daly_stats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset:
        cache_folder: [~, str]
    """)
    cf = cfg.parse()

    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])

    pd_kf, pd_obj = Daly_stats.pd_stats(dataset)

    Daly_stats.inst_bbox_stats_hists(pd_kf, out)
    Daly_stats.inst_bbox_stats_hmap(pd_kf, out, dataset)
    Daly_stats.obj_inst_bbox_stats(pd_obj, out, dataset)
