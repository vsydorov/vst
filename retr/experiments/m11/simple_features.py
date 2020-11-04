import concurrent
import pprint
import re
import time
import logging
from pathlib import Path
from typing import (  # NOQA
            Optional, Iterable, List, Dict,
            Any, Union, Callable, TypeVar)

import cv2
import pandas as pd
import numpy as np
import albumentations as A
import albumentations.pytorch
import sklearn.preprocessing
import sklearn.metrics
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate

import vst

from retr.misc import imread_func

log = logging.getLogger(__name__)


class Counter_repeated_action(object):
    """
    Will check whether repeated action should be performed
    """
    def __init__(self, sslice='::', seconds=None, iters=None):
        self.sslice = sslice
        self.seconds = seconds
        self.iters = iters
        self.tic(-1)

    def tic(self, i=None):
        self._time_last = time.perf_counter()
        if i is not None:
            self._i_last = i

    def check(self, i=None):
        ACTION = False
        if i is not None:
            ACTION |= vst.check_step(i, self.sslice)
            if self.iters is not None:
                ACTION |= (i - self._i_last) >= self.iters

        if self.seconds is not None:
            time_since_last = time.perf_counter() - self._time_last
            ACTION |= time_since_last >= self.seconds
        return ACTION


class Dataloader_isaver(
        vst.isave.Isaver_mixin_restore_save,
        vst.isave.Isaver_base):
    """
    Will process a list with a 'func',
    - prepare_func(start_i) is to be run before processing
    """
    def __init__(self, folder,
            total, func, prepare_func,
            save_period='::',
            save_interval=120,
            log_interval=None,):
        super().__init__(folder, total)
        self.func = func
        self.prepare_func = prepare_func
        self._save_period = save_period
        self._save_interval = save_interval
        self._log_interval = log_interval
        self.result = []

    def run(self):
        i_last = self._restore()
        countra = Counter_repeated_action(
                sslice=self._save_period,
                seconds=self._save_interval)

        result_cache = []

        def flush_purge():
            self.result.extend(result_cache)
            result_cache.clear()
            with vst.QTimer('saving pkl'):
                self._save(i_last)
            self._purge_intermediate_files()

        loader = self.prepare_func(i_last)
        pbar = tqdm(loader, total=len(loader))
        for i_batch, data_input in enumerate(pbar):
            result_dict, i_last = self.func(data_input)
            result_cache.append(result_dict)
            if countra.check(i_batch):
                flush_purge()
                log.debug(vst.tqdm_str(pbar))
                countra.tic(i_batch)
        flush_purge()
        return self.result


class ResNet_for_extraction(torchvision.models.resnet.ResNet):
    def __init__(self):
        block = torchvision.models.resnet.Bottleneck
        layers = [3, 4, 6, 3]
        super(ResNet_for_extraction, self).__init__(block, layers)
        # del self.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class TDataset_simple(torch.utils.data.Dataset):

    def __init__(
            self, source_list,):
        self.source_list = source_list

        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        anormalize = A.Normalize(mean=MEAN, std=STD)
        S = 256
        sub_transforms = [A.Resize(height=S, width=S)]
        sub_transforms.extend([
            anormalize,
            albumentations.pytorch.ToTensorV2(),
            ])
        self.atransform = A.Compose(sub_transforms)

    def collate(batch):
        im_torch, meta = zip(*batch)
        return [default_collate(im_torch), meta]

    def __len__(self):
        return len(self.source_list)

    def __getitem__(self, index):
        source = self.source_list[index]
        im = imread_func(source)
        im_torch = self.atransform(image=im)['image']
        # preprocessings
        meta = {'index': index, 'source': source}
        return im_torch, meta


class Isaver_threading(
        vst.isave.Isaver_mixin_restore_save, vst.isave.Isaver_base):
    """
    Will process a list with a func, in async manner
    """
    def __init__(self, folder, in_list, func,
            save_every=25, max_workers=5):
        super().__init__(folder, len(in_list))
        self.in_list = in_list
        self.result = {}
        self.func = func
        self._save_every = save_every
        self._max_workers = max_workers

    def run(self):
        self._restore()
        all_ii = set(range(len(self.in_list)))
        remaining_ii = all_ii - set(self.result.keys())

        io_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._max_workers)
        io_futures = []
        for i in remaining_ii:
            args = self.in_list[i]
            submitted = io_executor.submit(self.func, args)
            submitted.i = i
            io_futures.append(submitted)

        flush_dict = {}

        def flush_purge():
            self.result.update(flush_dict)
            flush_dict.clear()
            self._save(len(self.result))
            self._purge_intermediate_files()

        for io_future in tqdm(concurrent.futures.as_completed(io_futures),
                total=len(io_futures)):
            result = io_future.result()
            i = io_future.i
            flush_dict[i] = result
            if len(flush_dict) >= self._save_every:
                flush_purge()
        flush_purge()
        assert len(self.result) == len(self.in_list)
        result_list = [self.result[i] for i in all_ii]
        return result_list


# Experiments


def predict_convfeatures(workfolder, cfg_dict, add_arg):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict, 'threshdict.')
    cfg.set_defaults_yaml("""
    inputs:
        dataset: ~
    seed: 42
    """)
    cf = cfg.parse()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet_for_extraction()
    model_url = torchvision.models.resnet.model_urls['resnet50']
    state_dict = torchvision.models.resnet.load_state_dict_from_url(model_url)
    model.load_state_dict(state_dict)
    model.to(device)

    dataset = vst.load_pkl(cf['inputs.dataset'])

    # Get a list of only dev images
    pos_cluster_names = list(dataset.dev['positives_queries'].keys())
    pos_img_names = []
    for c in pos_cluster_names:
        pos_img_names.extend(dataset.clusters[c]['imkeys'])
    neg_img_names = dataset.dev['negative_images']

    img_names = pos_img_names + neg_img_names
    img_paths = [dataset.all_images[iname]['path'] for iname in img_names]

    def prepare_func(start_i):
        remaining_paths = img_paths[start_i+1:]
        dset = TDataset_simple(remaining_paths)
        batch_size = 32
        num_workers = 8
        dataloader = torch.utils.data.DataLoader(
                dset, batch_size=batch_size,
                shuffle=False, num_workers=num_workers,
                collate_fn=TDataset_simple.collate)
        return dataloader

    def func(data_input):
        data, meta = data_input
        data = data.to(device)
        output = model(data)
        out_np = output.detach().cpu().numpy().astype(np.float16)
        i_url = img_paths.index(meta[-1]['source'])
        return out_np, i_url

    # Predict features
    # pbar = enumerate(dataloader)
    # pbar = tqdm(pbar, total=len(dataloader))
    # outputs = []
    # for i_batch, data_input in pbar:
    #     outputs.append(out_np)

    model.eval()
    disaver_fold = vst.mkdir(out/'disaver')
    total = len(img_paths)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
            save_period='::', save_interval=30)
    outputs = disaver.run()

    # Stack
    conc = np.concatenate(outputs, axis=0)
    metadict = {
            'img_paths': img_paths, 'img_names': img_names}
    vst.save_pkl(out/'metadict.pkl', metadict)
    np.save(str(out/'concat_feats.npy'), conc)


def estimate_distances(workfolder, cfg_dict, add_arg):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict, 'threshdict.')
    cfg.set_defaults_yaml("""
    inputs:
        dataset: ~
        featurefold: ~
    seed: 42
    """)
    cf = cfg.parse()

    dataset = vst.load_pkl(cf['inputs.dataset'])
    ffold = Path(cf['inputs.featurefold'])
    meta = vst.load_pkl(ffold/'metadict.pkl')

    def qreduce():
        conc = np.load(str(ffold/'concat_feats.npy'))
        mean_pooled_conc = conc.mean(axis=(2, 3))
        max_pooled_conc = conc.max(axis=(2, 3))
        return mean_pooled_conc, max_pooled_conc

    (mean_pooled_conc, max_pooled_conc) = vst.stash2(
            out/'qreduced_feats.pkl')(qreduce)

    # (X[0]**2).sum() ~= 1
    X = sklearn.preprocessing.normalize(max_pooled_conc)
    # This is X indexed over img_names
    img_names = np.array(meta['img_names'])

    # Get xi and cluster_name for all queries
    # (xi is common index over X and img_names)
    reg = r'(.+)/(.+)\.'
    cluster_xi_mapping = {}
    for xi, img_name in enumerate(meta['img_names']):
        m = re.search(reg, img_name)
        cluster_name, im_name = m.groups()
        if cluster_name == 'negative_images':
            continue
        cluster = cluster_xi_mapping.setdefault(cluster_name, {})
        if im_name == 'query':
            cluster['query_xi'] = xi
        else:
            cluster.setdefault('data_xis', []).append(xi)

    ap_per_cluster = {}
    pbar = enumerate(cluster_xi_mapping.items())
    pbar = tqdm(pbar, total=len(cluster_xi_mapping))
    for i, (cluster_name, cluster) in pbar:
        qxi = cluster['query_xi']
        dxi = np.array(cluster['data_xis'])
        # Match to all database values
        x_query = X[qxi]
        similarity = X @ x_query.T

        # Positives are other guys from the cluster
        Y = np.zeros_like(similarity, dtype=np.int)
        Y[dxi] = 1
        # Remove query
        similarity_noquery = np.delete(similarity, qxi)
        Y_noquery = np.delete(Y, qxi)
        #
        # ap = sklearn.metrics.average_precision_score(Y_noquery, similarity_noquery)
        # ap_per_cluster[cluster_name] = ap

        # if i > 20:
        #     break

        # closest_inds = np.argsort(similarity)[::-1][:10]
        # closest_values = similarity[closest_inds]
        # closest_names = [meta['img_names'][i] for i in closest_inds]
        # query_name = meta['img_names'][ind]
        # s = pd.Series(closest_values, closest_names)
        # log.info('for file {} we found such top10 matches:\n{}'.format(
        #     query_name, s))
        #
    pprint.pprint(ap_per_cluster)


def estimate_mathieu_eval(workfolder, cfg_dict, add_arg):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict, 'threshdict.')
    cfg.set_defaults_yaml("""
    inputs:
        dataset: ~
        featurefold: ~
    seed: 42
    """)
    cf = cfg.parse()

    dataset = vst.load_pkl(cf['inputs.dataset'])
    ffold = Path(cf['inputs.featurefold'])
    meta = vst.load_pkl(ffold/'metadict.pkl')

    def qreduce():
        conc = np.load(str(ffold/'concat_feats.npy'))
        mean_pooled_conc = conc.mean(axis=(2, 3))
        max_pooled_conc = conc.max(axis=(2, 3))
        return mean_pooled_conc, max_pooled_conc

    (mean_pooled_conc, max_pooled_conc) = vst.stash2(
            out/'qreduced_feats.pkl')(qreduce)

    X = sklearn.preprocessing.normalize(max_pooled_conc)
    img_names_lst = meta['img_names']
    img_names = np.array(img_names_lst)

    score_keys = ["all", "alone_framing_different", "alone_color_change",
            "alone_layer_modification_superposition", "alone_other",
            "alone_similar", "framing_different", "color_change",
            "layer_modification_superposition", "other", "similar"]

    xis_negative = []
    reg = r'(.+)/(.+)\.'
    for xi, img_name in enumerate(img_names):
        m = re.search(reg, img_name)
        cluster_name, im_name = m.groups()
        if cluster_name == 'negative_images':
            xis_negative.append(xi)
    xis_negative = np.array(xis_negative)

    def compute_query_score(key, qi):
        query_score = {k: [] for k in score_keys}
        # Find query
        query_imgname = '{}/{}'.format(key, qi['query_image'])
        qxi = img_names_lst.index(query_imgname)
        x_query = X[qxi]
        similarity = X @ x_query
        for target_image in qi["target_images"]:
            target_imgname = '{}/{}'.format(key, target_image['image'])
            target_xi = img_names_lst.index(target_imgname)
            good_xis = np.r_[target_xi, xis_negative]
            scores_gxis = similarity[good_xis]

            argsorted_order10 = np.argsort(scores_gxis)[::-1][:10]
            argsorted_gxis = list(good_xis[argsorted_order10])

            # Inspired of MRR: Mean Reciprocal Rank
            score = (1/(argsorted_gxis.index(target_xi) + 1)) \
                if target_xi in argsorted_gxis else 0
            scores = [score] * target_image["multiplicator"]

            query_score["all"].extend(scores)
            transformations = target_image["transformations"]
            if transformations is not None:
                for transformation in transformations:
                    query_score[transformation].extend(scores)
                    if len(transformations) == 1:
                        query_score["alone_{}".format(
                            transformation)].extend(scores)
        return query_score

    isaver = Isaver_threading(
            vst.mkdir(out/'isave_qscore'),
            list(dataset.dev["positives_queries"].items()),
            lambda x: compute_query_score(x[0], x[1]),
            1000, 20)
    isaver_items = isaver.run()

    global_score = {k: [] for k in score_keys}
    for query_score in isaver_items:
        for k, v in query_score.items():
            if len(v) > 0:
                global_score[k].append(sum(v)/len(v))

    to_tabulate = []
    for transformation, list_score in global_score.items():
        if len(list_score) > 0:
            row = [transformation,
                  len(list_score),
                  "{:.4f}".format(sum(list_score)/len(list_score)),
                  "{:.2f}%".format(len([s for s in list_score if s != 0])/len(list_score) * 100)]
        else:
            row = [transformation, 0, 0, "0.00%"]
        to_tabulate.append(row)

    from tabulate import tabulate
    table = tabulate(to_tabulate,
             headers=['transformation', 'nb items', 'mean ranking score',
                 'percentage found'])

    log.info('Results:\n{}'.format(table))

    # for query_ind in query_indices[:10]:
    #     # Query process
    #     query_x = X[query_ind]
    #     similarity = X @ query_x.T
    #
    #     # Evaluation
    #     cluster_name = query_clusters[query_ind]
    #     cluster_imkeys = dataset.clusters[cluster_name]['imkeys']
    #     cluster_indices = [meta['img_names'].index(imkey)
    #             for imkey in cluster_imkeys]
    #
    #     query_name = meta['img_names'][ind]
    #     s = pd.Series(closest_values, closest_names)
    #     log.info('for file {} we found such top10 matches:\n{}'.format(
    #         query_name, s))
