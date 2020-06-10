import numpy as np
import logging
import time
import copy
from types import MethodType
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data

import slowfast.models
import slowfast.utils.misc as misc
import slowfast.utils.checkpoint as cu

from detectron2.layers import ROIAlign

from vsydorov_tools import small
from vsydorov_tools import cv as vt_cv
from vsydorov_tools import log as vt_log

from thes.data.dataset.external import (Dataset_daly_ocv)
from thes.tools import snippets
from thes.detectron.daly import (
    get_daly_split_vids)
from thes.slowfast.cfg import (base_sf_i3d_config)

log = logging.getLogger(__name__)


def np_to_gpu(X):
    X = torch.from_numpy(np.array(X))
    X = X.type(torch.cuda.FloatTensor)
    return X


norm_mean = np.array([0.45, 0.45, 0.45])
norm_mean_t = np_to_gpu(norm_mean)
norm_std = np.array([0.225, 0.225, 0.225])
norm_std_t = np_to_gpu(norm_std)
test_crop_size = 255
# DETECTION.ROI_XFORM_RESOLUTION
xform_resolution = 7
# SPATIAL_SCALE_FACTOR
spatial_scale_factor = 16
i3d_poolsize = [[2, 1, 1]]
# [[cfg.DATA.NUM_FRAMES // pool_size[0][0], 1, 1]]

def monkey_forward(self, x):
    x = self.s1(x)
    x = self.s2(x)
    for pathway in range(self.num_pathways):
        pool = getattr(self, "pathway{}_pool".format(pathway))
        x[pathway] = pool(x[pathway])
    x = self.s3(x)
    x = self.s4(x)
    x = self.s5(x)
    return x

class Extractor_roi(object):
    def __init__(self, model, model_nframes):
        self._model = copy.copy(model)
        self._model.forward = MethodType(monkey_forward, self._model)

        resolution = [xform_resolution] * 2
        # Definitions
        tpool_size = [model_nframes//i3d_poolsize[0][0], 1, 1]
        self.t_pool = nn.AvgPool3d(tpool_size, stride=1)
        self.roi_align = ROIAlign(resolution,
                spatial_scale=1.0/32,
                sampling_ratio=0,
                aligned=True)
        self.s_pool = nn.MaxPool2d(resolution, stride=1)

    def forward(self, X, bboxes):
        with torch.no_grad():
            # Forward through model
            x = self._model(X)
            # Forward
            assert len(x) == 1
            out = self.t_pool(x[0])
            assert out.shape[2] == 1
            out = torch.squeeze(out, 2)
            out = self.roi_align(out, bboxes)
            out = self.s_pool(out)
        return out


import cv2
import concurrent.futures


def yana_size_query(X, dsize):
    # https://github.com/hassony2/torch_videovision
    def _get_resize_sizes(im_h, im_w, size):
        if im_w < im_h:
            ow = size
            oh = int(size * im_h / im_w)
        else:
            oh = size
            ow = int(size * im_w / im_h)
        return oh, ow

    if isinstance(dsize, int):
        im_h, im_w, im_c = X[0].shape
        new_h, new_w = _get_resize_sizes(im_h, im_w, dsize)
        isize = (new_w, new_h)
    else:
        assert len(dsize) == 2
        isize = dsize[1], dsize[0]
    return isize

def threaded_ocv_resize_clip(
        X, dsize, max_workers=8,
        interpolation=cv2.INTER_LINEAR):
    isize = yana_size_query(X, dsize)
    thread_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)
    futures = []
    for img in X:
        futures.append(thread_executor.submit(
            cv2.resize, img, isize,
            interpolation=interpolation))
    concurrent.futures.wait(futures)
    thread_executor.shutdown()
    scaled = np.array([x.result() for x in futures])
    return scaled


def tfm_video_resize_threaded(X, dsize, max_workers=8):
    # 256 resize, normalize, group,
    h_before, w_before = X.shape[1:3]
    X = threaded_ocv_resize_clip(X, dsize, max_workers)
    h_resized, w_resized = X.shape[1:3]
    params = {'h_before': h_before, 'w_before': w_before,
              'h_resized': h_resized, 'w_resized': w_resized}
    return X, params


def tfm_video_center_crop(first64, th, tw):
    h_before, w_before = first64.shape[1:3]
    ccrop_i = int((h_before-th)/2)
    ccrop_j = int((w_before-tw)/2)
    first64 = first64[:,
            ccrop_i:ccrop_i+th,
            ccrop_j:ccrop_j+tw, :]
    params = {'h_before': h_before, 'w_before': w_before,
              'i': ccrop_i, 'j': ccrop_j,
              'th': th, 'tw': tw}
    return first64, params


def prepare_video(frames_u8):
    frames_rgb = np.flip(frames_u8, -1)
    # Resize
    X, resize_params = tfm_video_resize_threaded(
            frames_rgb, test_crop_size)
    # Centercrop
    X, ccrop_params = tfm_video_center_crop(
            X, test_crop_size, test_crop_size)
    # Convert to torch, add batch dimension
    Xt = torch.from_numpy(X)
    return Xt, resize_params, ccrop_params

def to_gpu_normalize_permute(Xt):
    X_f32c = Xt.type(torch.cuda.FloatTensor)
    X_f32c /= 255
    # Normalization after float conversion
    X_f32c = (X_f32c-norm_mean_t)/norm_std_t
    # Pad 0 dim and permute done last
    assert len(X_f32c.shape) == 5
    X_f32c = X_f32c.permute(0, 4, 1, 2, 3)
    return X_f32c

def prepare_box(bbox_ltrd, resize_params, ccrop_params):
    # X is NCHW
    # Resize bbox
    bbox_tldr = bbox_ltrd[[1, 0, 3, 2]]
    real_scale_h = resize_params['h_resized']/resize_params['h_before']
    real_scale_w = resize_params['w_resized']/resize_params['w_before']
    real_scale = np.tile(np.r_[real_scale_h, real_scale_w], 2)
    bbox_tldr = (bbox_tldr * real_scale)
    # Offset box
    i, j = ccrop_params['i'], ccrop_params['j']
    bbox_tldr -= [i, j, i, j]
    box_maxsize = np.tile(
            np.r_[ccrop_params['th'], ccrop_params['tw']], 2)
    bbox_tldr = np.clip(bbox_tldr, [0, 0, 0, 0], box_maxsize)
    return bbox_tldr

def _vis_boxes(out, bbox, frames_u8, X_f32c, bbox_tldr):
    fullsize_w_boxes = frames_u8.copy()
    for i, frame in enumerate(fullsize_w_boxes):
        snippets.cv_put_box_with_text(frame, bbox)
    snippets.qsave_video(out/'fullsize_w_boxes.mp4', fullsize_w_boxes)

    small_w_boxes = np.flip(X_f32c.cpu().numpy()[0].transpose(1, 2, 3, 0), -1)
    small_w_boxes = small_w_boxes.copy()
    small_w_boxes = small_w_boxes*norm_std + norm_mean
    small_w_boxes = (small_w_boxes * 255).astype(np.uint8)
    for i, frame in enumerate(small_w_boxes):
        snippets.cv_put_box_with_text(frame, bbox_tldr[[1, 0, 3, 2]])
    snippets.qsave_video(out/'small_w_boxes.mp4', small_w_boxes)


class TDataset_over_keyframes(torch.utils.data.Dataset):
    def __init__(self, keyframes, model_nframes, model_sample):
        self.keyframes = keyframes
        center_frame = (model_nframes-1)//2
        self.sample_grid0 = (np.arange(model_nframes)-center_frame)*model_sample

    def __getitem__(self, index):
        keyframe = self.keyframes[index]
        video_path = keyframe['video_path']
        i0 = keyframe['frame0']
        finds_to_sample = i0 + self.sample_grid0
        finds_to_sample = np.clip(
                finds_to_sample, 0, keyframe['nframes']-1)
        with vt_cv.video_capture_open(video_path) as vcap:
            frames_u8 = vt_cv.video_sample(vcap, finds_to_sample)
        frames_u8 = np.array(frames_u8)

        # frames_u8 is N,H,W,C
        Xt, resize_params, ccrop_params = prepare_video(frames_u8)
        bbox_tldr = prepare_box(keyframe['bbox'], resize_params, ccrop_params)
        bbox_tldr0 = np.r_[0, bbox_tldr]
        return index, Xt, bbox_tldr0

    def __len__(self) -> int:
        return len(self.keyframes)


def create_keyframelist(dataset):
    # Record keyframes
    keyframes = []
    for vid, ovideo in dataset.videos_ocv.items():
        nframes = ovideo['nframes']
        for action_name, instances in ovideo['instances'].items():
            for ins_ind, instance in enumerate(instances):
                fl = instance['flags']
                diff = fl['isReflection'] or fl['isAmbiguous']
                if diff:
                    continue
                for kf_ind, keyframe in enumerate(instance['keyframes']):
                    frame0 = keyframe['frame']
                    action_id = dataset.action_names.index(action_name)
                    kf_dict = {
                            'vid': vid,
                            'action_id': action_id,
                            'action_name': action_name,
                            'ins_ind': ins_ind,
                            'kf_ind': kf_ind,
                            'bbox': keyframe['bbox_abs'],
                            'video_path': ovideo['path'],
                            'frame0': int(frame0),
                            'nframes': nframes,
                            'height': ovideo['height'],
                            'width': ovideo['width'],
                            }
                    keyframes.append(kf_dict)
    return keyframes


class Dataloader_isaver(
        snippets.Isaver_mixin_restore_save, snippets.Isaver_base):
    """
    Will process a list with a 'func', 'prepare_func(start_i)' is to be run before processing
    """
    def __init__(self, folder,
            total, func, prepare_func,
            save_every=0,
            save_interval=120,  # every 2 minutes by default
            log_interval=None,):
        super().__init__(folder, total)
        self.func = func
        self.prepare_func = prepare_func
        self._save_every = save_every
        self._save_interval = save_interval
        self._log_interval = log_interval
        self.result = []

    def run(self):
        i_last = self._restore()
        self._time_last_save = time.perf_counter()
        self._time_last_log = time.perf_counter()
        self._i_last_saved = 0

        Ys_np = []

        def flush_purge():
            self.result.extend(Ys_np)
            Ys_np.clear()
            self._save(i_last)
            self._purge_intermediate_files()

        loader = self.prepare_func(i_last)
        pbar = tqdm(loader, total=len(loader))
        for i_batch, data_input in enumerate(pbar):
            II_np, Y_np = self.func(data_input)
            Ys_np.append(Y_np)
            i_last = II_np[-1]
            PURGE = False
            if self._save_every > 0:
                PURGE |= (i_last - self._i_last_saved) >= self._save_every
            if self._save_interval:
                since_last_save = time.perf_counter() - self._time_last_save
                PURGE |= since_last_save > self._save_interval
            if PURGE:
                flush_purge()
                self._time_last_save = time.perf_counter()
            if self._log_interval:
                since_last_log = time.perf_counter() - self._time_last_log
                if since_last_log > self._log_interval:
                    log.info(snippets._tqdm_str(pbar))
                    self._time_last_log = time.perf_counter()
        flush_purge()
        return self.result

def extract_slowfast_feats(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    seed: [42, int]
    dataset:
        name: ['daly', ['daly']]
        cache_folder: [~, str]
        subset: ['train', ~]
    """)
    cf = cfg.parse()

    # DALY Dataset
    dataset = Dataset_daly_ocv()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
    # split_label = cf['dataset.subset']
    # split_vids = get_daly_split_vids(dataset, split_label)

    sf_cfg = base_sf_i3d_config()
    sf_cfg.NUM_GPUS = 1
    sf_cfg.TEST.BATCH_SIZE = 16
    # Load model
    model = slowfast.models.build_model(sf_cfg)
    model.eval()
    # misc.log_model_info(model, sf_cfg, is_train=False)

    CHECKPOINT_FILE_PATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/102_slowfast/20_checkpoints/I3D_8x8_R50.pkl'
    with vt_log.logging_disabled(logging.WARNING):
        cu.load_checkpoint(
            CHECKPOINT_FILE_PATH, model, False, None,
            inflation=False, convert_from_caffe2=True,)

    keyframes = create_keyframelist(dataset)

    model_nframes = sf_cfg.DATA.NUM_FRAMES
    model_sample = sf_cfg.DATA.SAMPLING_RATE
    extractor_roi = Extractor_roi(model, model_nframes)

    BATCH_SIZE = 8
    NUM_WORKERS = 12

    def prepare_func(start_i):
        remaining_keyframes = keyframes[start_i+1:]
        tdataset_kf = TDataset_over_keyframes(
                remaining_keyframes, model_nframes, model_sample)
        loader = torch.utils.data.DataLoader(tdataset_kf,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True)
        return loader

    def func(data_input):
        II, Xts, bboxes = data_input
        Xs_f32c = to_gpu_normalize_permute(Xts)
        bboxes_c = bboxes.type(torch.cuda.FloatTensor)
        Y = extractor_roi.forward([Xs_f32c], bboxes_c)
        II_np = II.cpu().numpy()
        Y_np = Y.cpu().numpy()
        return II_np, Y_np

    disaver_fold = small.mkdir(out/'disaver')
    total = len(keyframes)
    disaver = Dataloader_isaver(disaver_fold, total, func, prepare_func,
            save_interval=60)
    outputs = disaver.run()
    sq_outputs = np.vstack(outputs).squeeze()
    small.save_pkl(out/'sq_outputs.pkl', sq_outputs)
    small.save_pkl(out/'keyframes.pkl', keyframes)
