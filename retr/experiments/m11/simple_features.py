import time
import logging
from pathlib import Path
from typing import (  # NOQA
            Optional, Iterable, List, Dict,
            Any, Union, Callable, TypeVar)

import cv2
import numpy as np
import albumentations as A
import albumentations.pytorch
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
        num_workers = 0
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
