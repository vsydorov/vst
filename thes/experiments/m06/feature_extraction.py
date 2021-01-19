import numpy as np
import logging
from pathlib import Path
from typing import (  # NOQA
    Dict, Any, List, Optional, Tuple, TypedDict, Set)
from tqdm import tqdm

import torch
import torch.utils.data

from vsydorov_tools import small

from thes.data.dataset.daly import (
    Ncfg_daly, sample_daly_frames_from_instances,
    create_keyframelist, to_keyframedict)
from thes.data.dataset.external import (
    Dataset_daly_ocv, Vid_daly)
from thes.data.tubes.types import (
    I_dwein, T_dwein,
    loadconvert_tubes_dwein, Box_connections_dwti)
from thes.data.tubes.routines import (
    group_tubes_on_frame_level, perform_connections_split)
from thes.tools import snippets
from thes.pytorch import (
    sequence_batch_collate_v2, Sampler_grid, Frameloader_video_slowfast,
    TDataset_over_keyframes, TDataset_over_connections,
    Dataloader_isaver, to_gpu_normalize_permute, )
from thes.feat_extract import (
    Ncfg_extractor)

log = logging.getLogger(__name__)

class Isaver_extract_rgb(snippets.isaver.Isaver_base):
    def __init__(
            self, folder,
            total, func, prepare_func,
            interval_iters=None,
            interval_seconds=120,  # every 2 minutes by default
                ):
        super(Isaver_extract_rgb, self).__init__(folder, total)
        self.func = func
        self.prepare_func = prepare_func
        self._interval_iters = interval_iters
        self._interval_seconds = interval_seconds
        self.result = []
        self.npy_array = None

    def _get_filenames(self, i) -> Dict[str, Path]:
        base_filenames = {
            'finished': self._fmt_finished.format(i, self._total)}
        base_filenames['pkl'] = Path(
                base_filenames['finished']).with_suffix('.pkl')
        base_filenames['npy'] = Path(
                base_filenames['finished']).with_suffix('.npy')
        filenames = {k: self._folder/v
                for k, v in base_filenames.items()}
        return filenames

    def _restore(self):
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        start_i, ifiles = max(intermediate_files.items(),
                default=(-1, None))
        if ifiles is not None:
            self.result = small.load_pkl(ifiles['pkl'])
            self.npy_array = np.load(ifiles['npy'])
            log.info('Restore from pkl: {} npy: {}'.format(
                ifiles['pkl'], ifiles['npy']))
        return start_i

    def _save(self, i):
        ifiles = self._get_filenames(i)
        small.save_pkl(ifiles['pkl'], self.result)
        np.save(ifiles['npy'], self.npy_array)
        ifiles['finished'].touch()

    def run(self):
        i_last = self._restore()
        countra = snippets.Counter_repeated_action(
                seconds=self._interval_seconds,
                iters=self._interval_iters)

        pkl_cache = []
        npy_cache = []

        def flush_purge():
            self.result.extend(pkl_cache)
            if self.npy_array is None:
                to_stack = npy_cache
            else:
                to_stack = (self.npy_array, *npy_cache)
            self.npy_array = np.vstack(to_stack)
            pkl_cache.clear()
            npy_cache.clear()
            with small.QTimer('saving'):
                self._save(i_last)
            self._purge_intermediate_files()

        loader = self.prepare_func(i_last)
        pbar = tqdm(loader, total=len(loader))
        for i_batch, data_input in enumerate(pbar):
            pkl_part, npy_part, i_last = self.func(data_input)
            pkl_cache.append(pkl_part)
            npy_cache.append(npy_part)
            if countra.check(i_batch):
                flush_purge()
                log.debug(snippets.tqdm_str(pbar))
                countra.tic(i_batch)
        flush_purge()
        return self.result, self.npy_array

# Experiments


def extract_keyframe_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_extractor.set_defcfg(cfg)
    cf = cfg.parse()

    # prepare extractor
    norm_mean_t, norm_std_t, sampler_grid, frameloader_vsf, fextractor = \
            Ncfg_extractor.prepare(cf)
    BATCH_SIZE = cf['extraction.batch_size']
    NUM_WORKERS = cf['extraction.num_workers']
    # prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    keyframes = create_keyframelist(dataset)
    keyframes_dict = to_keyframedict(keyframes)

    # / extract
    def prepare_func(start_i):
        remaining_keyframes_dict = dict(list(
            keyframes_dict.items())[start_i+1:])
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes_dict, sampler_grid, frameloader_vsf)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
                collate_fn=sequence_batch_collate_v2)
        return loader

    bboxes_batch_index = torch.arange(
        BATCH_SIZE).type(torch.DoubleTensor)[:, None]

    def func(data_input):
        metas, Xts, bboxes = data_input
        kkeys = [tuple(m['kkey']) for m in metas]
        Xts_f32c = [to_gpu_normalize_permute(
            x, norm_mean_t, norm_std_t) for x in Xts]

        bsize = bboxes.shape[0]
        bboxes0 = torch.cat(
                (bboxes_batch_index[:bsize], bboxes), axis=1)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)
        with torch.no_grad():
            result = fextractor.forward(Xts_f32c, bboxes0_c)
        result_dict = {k: v.cpu().numpy()
                for k, v in result.items()}
        last_i = list(keyframes_dict.keys()).index(kkeys[-1])
        return result_dict, last_i

    disaver_fold = small.mkdir(out/'disaver')
    total = len(keyframes)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
        save_interval_seconds=cf['extraction.save_interval'])
    outputs = disaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        stacked = np.vstack([o[k] for o in outputs])
        dict_outputs[k] = stacked
    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    small.save_pkl(out/'keyframes.pkl', keyframes)


def extract_philtube_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    Ncfg_extractor.set_defcfg(cfg)
    cfg.set_deftype("""
    tubes_dwein: [~, str]
    frame_coverage:
        subsample: [16, int]
    compute_split:
        enabled: [False, bool]
        chunk: [0, "VALUE >= 0"]
        total: [1, int]
    """)
    cf = cfg.parse()

    # prepare extractor
    norm_mean_t, norm_std_t, sampler_grid, frameloader_vsf, fextractor = \
            Ncfg_extractor.prepare(cf)
    BATCH_SIZE = cf['extraction.batch_size']
    NUM_WORKERS = cf['extraction.num_workers']
    # / prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    # // Frames to cover: keyframes and every 16th frame
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
        sample_daly_frames_from_instances(dataset, cf['frame_coverage.subsample'])
    connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    connections_f = group_tubes_on_frame_level(
            tubes_dwein, frames_to_cover)
    # Here we'll run our connection split
    if cf['compute_split.enabled']:
        cc, ct = (cf['compute_split.chunk'], cf['compute_split.total'])
        connections_f = perform_connections_split(connections_f, cc, ct)

    # / extract
    def prepare_func(start_i):
        # start_i defined wrt keys in connections_f
        remaining_dict = dict(list(
            connections_f.items())[start_i+1:])
        tdataset_kf = TDataset_over_connections(
            remaining_dict, dataset, sampler_grid, frameloader_vsf)
        loader = torch.utils.data.DataLoader(tdataset_kf,
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            collate_fn=sequence_batch_collate_v2)
        return loader

    def func(data_input):
        Xts, metas = data_input
        Xts_f32c = [to_gpu_normalize_permute(
            x, norm_mean_t, norm_std_t) for x in Xts]
        # bbox transformations
        bboxes_np = [m['bboxes_tldr'] for m in metas]
        counts = np.array([len(x) for x in bboxes_np])
        batch_indices = np.repeat(np.arange(len(counts)), counts)
        bboxes0 = np.c_[batch_indices, np.vstack(bboxes_np)]
        bboxes0 = torch.from_numpy(bboxes0)
        bboxes0_c = bboxes0.type(torch.cuda.FloatTensor)

        with torch.no_grad():
            result = fextractor.forward(Xts_f32c, bboxes0_c)
        result_dict = {}
        for k, v in result.items():
            out_np = v.cpu().numpy()
            out_split = np.split(out_np,
                np.cumsum(counts), axis=0)[:-1]
            result_dict[k] = out_split

        # Find last index over global structure
        # back to tuple, since dataloader casts to list
        ckey = tuple(metas[-1]['ckey'])
        ckeys = list(connections_f.keys())
        last_i = ckeys.index(ckey)
        return result_dict, last_i

    disaver_fold = small.mkdir(out/'disaver')
    total = len(connections_f)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
        save_interval_seconds=cf['extraction.save_interval'], log_interval=300)
    outputs = disaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        key_outputs = [oo for o in outputs for oo in o[k]]
        dict_outputs[k] = key_outputs

    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    small.save_pkl(out/'connections_f.pkl', connections_f)


def combine_split_philtube_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['inputs.cfolders'])
    Ncfg_daly.set_defcfg(cfg)
    cfg.set_deftype("""
    inputs:
        cfolders: [~, ~]
        dims: [~, int]
        key: ['roipooled', str]
    tubes_dwein: [~, str]
    frame_coverage:
        keyframes: [True, bool]
        subsample: [16, int]
    """)
    cf = cfg.parse()

    # / prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    # // Frames to cover: keyframes and every 16th frame
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
        sample_daly_frames_from_instances(dataset, cf['frame_coverage.subsample'])
    connections_f: Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    connections_f = group_tubes_on_frame_level(
            tubes_dwein, frames_to_cover)

    # Load inputs now
    input_cfolders = cf['inputs.cfolders']
    if not snippets.gather_check_all_present(input_cfolders, [
            'dict_outputs.pkl', 'connections_f.pkl']):
        return

    # Loading all piecemeal connections
    i_cons = {}
    for i, path in enumerate(input_cfolders):
        path = Path(path)
        local_connections_f = small.load_pkl(path/'connections_f.pkl')
        i_cons[i] = local_connections_f
    # Check consistency
    grouped_cons = {}
    for c in i_cons.values():
        grouped_cons.update(c)
    if grouped_cons.keys() != connections_f.keys():
        log.error('Loaded connections inconsistent with expected ones')

    partbox_numbering = []
    for lc in i_cons.values():
        nboxes = np.sum([len(c['boxes']) for c in lc.values()])
        partbox_numbering.append(nboxes)
    partbox_numbering = np.r_[0, np.cumsum(partbox_numbering)]

    # Create mapping of indices
    box_inds = [0]
    for c in connections_f.values():
        box_inds.append(len(c['boxes']))
    box_inds = np.cumsum(box_inds)
    box_inds2 = np.c_[box_inds[:-1], box_inds[1:]]

    small.save_pkl(out/'connections_f.pkl', connections_f)
    small.save_pkl(out/'box_inds2.pkl', box_inds2)

    np_filename = str(out/'feats.npy')
    dset = np.lib.format.open_memmap(np_filename, 'w+',
        dtype=np.float16, shape=(partbox_numbering[-1], cf['inputs.dims']))

    inputs_key = cf['inputs.key']
    if inputs_key != 'roipooled':
        log.warn(f"{inputs_key=} != 'roipooled'. You sure?")
    # Piecemeal conversion
    for i, path in enumerate(input_cfolders):
        log.info(f'Merging chunk {i=} at {path=}')
        path = Path(path)
        with small.QTimer('Unpickling'):
            local_dict_outputs = small.load_pkl(path/'dict_outputs.pkl')
        roipooled_feats = local_dict_outputs[inputs_key]
        with small.QTimer('Vstack'):
            cat_roipooled_feats = np.vstack(roipooled_feats)
        with small.QTimer('to float16'):
            feats16 = cat_roipooled_feats.astype(np.float16)
        b, e = partbox_numbering[i], partbox_numbering[i+1]
        assert e-b == feats16.shape[0]
        with small.QTimer(f'Saving to disk chunk {i=}'):
            dset[b:e] = feats16

    del dset


def combine_split_philtube_fullframe_features(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_defaults_handling(['inputs.cfolders'])
    Ncfg_daly.set_defcfg(cfg)
    cfg.set_deftype("""
    inputs:
        cfolders: [~, ~]
        dims: [~, int]
        key: ['fullframe', str]
    tubes_dwein: [~, str]
    frame_coverage:
        keyframes: [True, bool]
        subsample: [16, int]
    """)
    cf = cfg.parse()

    # / prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    tubes_dwein: Dict[I_dwein, T_dwein] = \
            loadconvert_tubes_dwein(cf['tubes_dwein'])
    # // Frames to cover: keyframes and every 16th frame
    frames_to_cover: Dict[Vid_daly, np.ndarray] = \
        sample_daly_frames_from_instances(dataset, cf['frame_coverage.subsample'])
    connections_f_: Dict[Tuple[Vid_daly, int], Box_connections_dwti]
    connections_f_ = group_tubes_on_frame_level(
            tubes_dwein, frames_to_cover)

    # Load inputs now
    input_cfolders = cf['inputs.cfolders']
    if not snippets.gather_check_all_present(input_cfolders, [
            'dict_outputs.pkl', 'connections_f.pkl']):
        return

    inputs_key = cf['inputs.key']
    if inputs_key != 'fullframe':
        log.warn(f"{inputs_key=} != 'fullframe'. You sure?")
    # Chunk merge
    connections_f = {}
    dict_outputs = {}
    for i, path in enumerate(input_cfolders):
        path = Path(path)
        local_outputs = small.load_pkl(path/'dict_outputs.pkl')
        for k, v in local_outputs.items():
            dict_outputs.setdefault(k, []).extend(v)
        connections_f.update(small.load_pkl(path/'connections_f.pkl'))
    # Check consistency
    if connections_f_.keys() != connections_f.keys():
        log.error('Loaded connections inconsistent with expected ones')

    # fix mistake
    fullframe = np.vstack(dict_outputs[inputs_key])
    small.save_pkl(out/'connections_f.pkl', connections_f)
    small.save_pkl(out/'fullframe.pkl', fullframe)


def extract_keyframe_rgb(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    Ncfg_daly.set_defcfg(cfg)
    cfg.set_deftype("""
    is_slowfast: [False, bool]
    slowfast_alpha: [4, int]
    nframes: [1, int]
    sample: [1, int]
    frame_size: [256, int]
    subset_vids: [~, ~]
    num_workers: [12, int]
    """)
    cf = cfg.parse()
    # prepare data
    dataset: Dataset_daly_ocv = Ncfg_daly.get_dataset(cf)
    vgroup: Dict[str, List[Vid_daly]] = Ncfg_daly.get_vids(cf, dataset)
    keyframes = create_keyframelist(dataset)
    if cf['subset_vids'] is not None:
        subset_vids = vgroup[cf['subset_vids']]
        keyframes = [kf for kf in keyframes if kf['vid'] in subset_vids]
    keyframes_dict = to_keyframedict(keyframes)
    # prepare others
    NUM_WORKERS = cf['num_workers']
    BATCH_SIZE = 32

    model_nframes = cf['nframes']
    model_sample = cf['sample']
    is_slowfast = cf['is_slowfast']
    slowfast_alpha = cf['slowfast_alpha']
    sampler_grid = Sampler_grid(model_nframes, model_sample)
    frameloader_vsf = Frameloader_video_slowfast(
            is_slowfast, slowfast_alpha, cf['frame_size'], 'ltrd')

    def prepare_func(start_i):
        remaining_keyframes_dict = dict(list(
            keyframes_dict.items())[start_i+1:])
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes_dict, sampler_grid, frameloader_vsf)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=False,
                collate_fn=sequence_batch_collate_v2)
        return loader

    def func(data_input):
        metas, frame_list, bboxes_tldr = data_input
        kkeys = [tuple(m['kkey']) for m in metas]
        bboxes_np = bboxes_tldr.cpu().numpy()
        if is_slowfast:
            # Get the non-subsampled version
            Xts_np = frame_list[1].cpu().numpy()
        else:
            Xts_np = frame_list[0].cpu().numpy()
        pkl_part = {
                'bboxes': bboxes_np,
                'kkeys': kkeys}
        npy_part = Xts_np

        last_i = list(keyframes_dict.keys()).index(kkeys[-1])
        return pkl_part, npy_part, last_i

    disaver_fold = small.mkdir(out/'disaver')
    total = len(keyframes)
    disaver = Isaver_extract_rgb(disaver_fold, total, func, prepare_func,
        interval_seconds=90)
    outputs, npy_array = disaver.run()
    keys = next(iter(outputs)).keys()
    dict_outputs = {}
    for k in keys:
        key_outputs = [oo for o in outputs for oo in o[k]]
        dict_outputs[k] = key_outputs

    small.save_pkl(out/'dict_outputs.pkl', dict_outputs)
    np.save(out/'rgb.npy', npy_array)
