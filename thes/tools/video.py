import time
import concurrent.futures
import logging
from contextlib import contextmanager
from typing import Tuple, List

import numpy as np
import cv2
import ffmpeg  # type: ignore

log = logging.getLogger(__name__)


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

def randint0(value):
    if value == 0:
        return 0
    else:
        return np.random.randint(value)


def yana_ocv_resize_clip(X, dsize):
    isize = yana_size_query(X, dsize)
    scaled = np.stack([
        cv2.resize(img, isize, interpolation=cv2.INTER_LINEAR) for img in X
    ])
    return scaled


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
    scaled = [x.result() for x in futures]
    return scaled


def _get_randombox(h_before, w_before, th, tw):
    i = randint0(h_before-th)
    j = randint0(w_before-tw)
    return np.r_[i, j, i+th, j+tw]


def _get_centerbox(h_before, w_before, th, tw):
    i = int((h_before-th)/2)
    j = int((w_before-tw)/2)
    return np.r_[i, j, i+th, j+tw]


def ffmpeg_video_info(path):
    probe = ffmpeg.probe(path)
    video_stream = next((
        stream for stream in probe['streams']
        if stream['codec_type'] == 'video'), None)
    assert video_stream is not None
    return video_stream


def ffmpeg_video_frames_read(video_path, fps):
    video_stream = ffmpeg_video_info(video_path)
    width = int(video_stream['width'])
    height = int(video_stream['height'])
    ffmpeg_prepare = ffmpeg.input(video_path)
    if fps is None:
        ffmpeg_prepare = ffmpeg_prepare.output(
                'pipe:', format='rawvideo', pix_fmt='rgb24')
    else:
        ffmpeg_prepare = ffmpeg_prepare.output(
                'pipe:', format='rawvideo', pix_fmt='rgb24', r=fps)
    fout, _ = ffmpeg_prepare.run(capture_stdout=True, capture_stderr=True)
    video_frames = (
        np
        .frombuffer(fout, np.uint8)
        .reshape([-1, height, width, 3])
    )
    if fps is None:
        fps = video_stream['avg_frame_rate'].split('/')
        fps = float(fps[0]) / float(fps[1])
    return video_frames, fps


""" Transforms """


def tfm_video_resize_threaded(X_list, dsize, max_workers=8):
    # 256 resize, normalize, group,
    h_before, w_before = X_list[0].shape[0:2]
    X_list = threaded_ocv_resize_clip(X_list, dsize, max_workers)
    h_resized, w_resized = X_list[0].shape[0:2]
    params = {'h_before': h_before, 'w_before': w_before,
              'h_resized': h_resized, 'w_resized': w_resized}
    return X_list, params


def tfm_video_random_crop(first64, th, tw):
    h_before, w_before = first64.shape[1:3]
    rcrop_i = randint0(h_before - th)
    rcrop_j = randint0(w_before - tw)
    first64 = first64[:,
            rcrop_i:rcrop_i+th,
            rcrop_j:rcrop_j+tw, :]
    params = {'h_before': h_before, 'w_before': w_before,
              'i': rcrop_i, 'j': rcrop_j,
              'th': th, 'tw': tw}
    return first64, params


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


def tfm_maybe_flip(first64):
    perform_video_flip = np.random.random() < 0.5
    if perform_video_flip:
        first64 = np.flip(first64, axis=2).copy()
    params = {'perform': perform_video_flip}
    return first64, params


""" Reverse Transforms """


def tfm_uncrop_box(box, params):
    i, j = params['i'], params['j']
    return box + [i, j, i, j]


def tfm_unresize_box(box, params):
    real_scale_h = params['h_resized']/params['h_before']
    real_scale_w = params['w_resized']/params['w_before']
    real_scale = np.tile(np.r_[real_scale_h, real_scale_w], 2)
    box = (box / real_scale).astype(int)
    return box


"""
Reading videos (copypaste from vsydorov_tools.cv)
"""


@contextmanager
def video_capture_open(video_path, tries=1):
    i = 0
    while i < tries:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                break
            else:
                raise IOError(f'OpenCV cannot open {video_path}')
        except Exception:
            time.sleep(1)
            i += 1

    if not cap.isOpened():
        raise IOError(f'OpenCV cannot open {video_path} after {i} tries')

    yield cap
    cap.release()


class VideoCaptureError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)


def video_sorted_enumerate(cap,
        sorted_framenumbers,
        throw_on_failure=True,
        debug_filename=None):
    """
    Fast opencv frame iteration
    - Only operates on sorted frames
    - Throws if failed to read the frame.
    - The failures happen quite often

    Args:
        cap: cv2.VideoCapture class
        sorted_framenumbers: Sorted sequence of framenumbers
        strict: Throw if failed to read frame
    Yields:
        (framenumber, frame_BGR)
    """

    def stop_at_0():
        if ret == 0:
            FAIL_MESSAGE = "Failed to read frame {} from '{}'".format(
                    f_current, debug_filename)
            if throw_on_failure:
                raise VideoCaptureError(FAIL_MESSAGE)
            else:
                log.warning(FAIL_MESSAGE)
                return

    assert (np.diff(sorted_framenumbers) >= 0).all(), \
            'framenumber must be nondecreasing'
    sorted_framenumbers = iter(sorted_framenumbers)
    try:  # Will stop iteration if empty
        f_current = next(sorted_framenumbers)
    except StopIteration:
        return
    f_next = f_current
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_current)
    while True:
        while f_current <= f_next:  # Iterate till f_current, f_next match
            f_current += 1
            ret = cap.grab()
            stop_at_0()
        ret, frame_BGR = cap.retrieve()
        yield (f_current-1, frame_BGR)
        try:  # Will stop iteration if empty
            f_next = next(sorted_framenumbers)
        except StopIteration:
            return
        assert f_current == int(cap.get(cv2.CAP_PROP_POS_FRAMES))


def video_sample(cap, framenumbers, debug_filename=None) -> List:
    sorted_framenumber = np.unique(framenumbers)
    frames_BGR = {}
    for i, frame_BGR in video_sorted_enumerate(cap, sorted_framenumber,
            debug_filename=debug_filename):
        frames_BGR[i] = frame_BGR
    sampled_BGR = []
    for i in framenumbers:
        sampled_BGR.append(frames_BGR[i])
    return sampled_BGR
