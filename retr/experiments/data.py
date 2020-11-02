import pandas as pd
import json
import numpy as np
import copy
import concurrent
import imageio
import cv2
import logging
from pathlib import Path
from tqdm import tqdm

import natsort

import vst

log = logging.getLogger(__name__)


__ROOT = Path('/home/vlad/projects/dervo_deployed/2020_11_02_retrieve')


def get_dataset_path(data_id):
    return __ROOT/'links/datasets'/data_id


def imread_func(path):
    # Returns RGB
    spath = str(path)
    image = cv2.imread(spath)
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif spath.endswith('.gif'):
        image = imageio.mimread(spath)[0][..., :4]
    else:
        raise RuntimeError(f'Can not read {path}')
    return image


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


class Dataset_copyright8g(object):
    clusters = None
    all_images = None

    def __init__(self, root_path, clusters, all_images):
        self.root_path = root_path
        self.clusters = clusters
        self.all_images = all_images

        pass


# Experiments


def demo(workfolder, cfg_dict, add_arg):
    out, = vst.exp.get_subfolders(workfolder, ['out'])
    cfg = vst.exp.YConfig(cfg_dict)
    cfg.set_defaults_yaml("""
    seed: 42
    """)
    cf = cfg.parse()

    ns_key = natsort.natsort_keygen()
    root_path = get_dataset_path('copyright_rise')
    imfold = root_path/'copyright'
    clusters = {}
    all_images = {}
    for fold in sorted(imfold.iterdir(), key=lambda x: ns_key(x.name)):
        images = sorted(fold.iterdir(), key=lambda x: ns_key(x.name))
        imkeys = []
        for img in images:
            imkey = '/'.join(str(img).split('/')[-2:])
            all_images[imkey] = {'imkey': imkey, 'path': img}
            imkeys.append(imkey)
        clusters[fold.name] = {'name': fold.name, 'imkeys': imkeys}

    N_pos = [len(v) for k, v in clusters.items() if k!='negative_images']
    N_neg = len(clusters['negative_images'])

    # Get sizestats
    def get_sizestats(imkey, imdict):
        imdict = copy.copy(imdict)
        height, width = None, None
        success = False
        try:
            path = imdict['path']
            im = imread_func(path)
            height, width = im.shape[:2]
            success = True
        except Exception:
            log.info(f'{path} is borked')
        imdict['height'] = height
        imdict['width'] = width
        imdict['success'] = success
        return imdict

    isaver = Isaver_threading(
            vst.mkdir(out/'isave_sizestats'), list(all_images.items()),
            lambda x: get_sizestats(x[0], x[1]),
            1000, 8)
    isaver_items = isaver.run()
    assert np.all([ii['success'] for ii in isaver_items])

    all_images = {}
    for ii in isaver_items:
        del ii['success']
        imkey = ii['imkey']
        all_images[imkey] = ii

    # Assign cluster
    for cluster in clusters.values():
        for imkey in cluster['imkeys']:
            all_images[imkey]['cluster'] = cluster['name']

    dev_json = root_path/'dev_copyright_dataset.json'
    with dev_json.open('r') as f:
        dev = json.load(f)

    test_json = root_path/'test_copyright_dataset.json'
    with test_json.open('r') as f:
        test = json.load(f)

    pq_dev = dev['positives_queries'].keys()
    pq_test = test['positives_queries'].keys()
    log.info('PQ dev={} PQ test={}'.format(len(pq_dev), len(pq_test)))

    pq = dev['positives_queries']
    pq.update(test['positives_queries'])

    pq_lens = pd.Series({k: len(v['target_images'])
        for k, v in pq.items()}).sort_index()
    clusters_lens = pd.Series({k: len(v['imkeys']) for k, v in clusters.items()
            if k != 'negative_images'}).sort_index()
    assert (clusters_lens-1 == pq_lens).all()

    dataset = Dataset_copyright8g(root_path, clusters, all_images)
    dataset.dev = dev
    dataset.test = test

    vst.save_pkl(out/'dataset.pkl', dataset)
