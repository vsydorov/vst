"""
Helpers for opencv functions
- Opening/closing files
- Rotations and stuff
"""
import subprocess
import time
import scipy
import numpy as np
import cv2  # type: ignore
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Union, Any, NamedTuple, List, Tuple, Callable, TypeVar, Iterator, Iterable, Sequence  # NOQA

log = logging.getLogger(__name__)

# // CV2 file management

# These values work for me
FOURCC_TO_CONTAINER = {
        'VP90': '.webm',
        'XVID': '.avi',
        'MJPG': '.avi',
        'H264': '.mp4',
        'MP4V': '.mp4'
}


def ffmpeg_convert_to_webm(convert_from, convert_to):
    """https://stackoverflow.com/questions/31624787/ffmpeg-very-slow-conversion'"""
    result = subprocess.run(
            "ffmpeg -i '{}' -deadline realtime -speed 8 -y '{}'".format(
                convert_from, convert_to),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise IOError(result.stdout)


def imwrite(path, im):
    path = str(path)
    succ = cv2.imwrite(path, im)
    if succ is False:
        raise IOError(f'OpenCV cannot save image {im.shape} to {path}')
    return path


# //// Video management
def suffix_helper(video_path: Path, fourcc) -> Path:
    """Change video suffix based on 4cc"""
    return video_path.with_suffix(FOURCC_TO_CONTAINER[fourcc])


@contextmanager
def video_writer_open(
        video_path: Path,
        size_WH: Tuple[int, int],
        framerate: float,
        fourcc):

    cv_fourcc = cv2.VideoWriter_fourcc(*fourcc)
    vout = cv2.VideoWriter(str(video_path), cv_fourcc, framerate, size_WH)
    try:
        if not vout.isOpened():
            raise IOError(f'OpenCV cannot open {video_path} for writing')
        yield vout
    finally:
        vout.release()


@contextmanager
def video_capture_open(video_path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise IOError(f'OpenCV cannot open {video_path}')
        yield cap
    finally:
        cap.release()


@contextmanager
def video_capture_open_v2(video_path, tries=1):
    i = 0
    while i < tries:
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                break
            else:
                raise IOError(f'OpenCV cannot open {video_path}')
        except Exception as e:
            time.sleep(1)
            i += 1

    if not cap.isOpened():
        raise IOError(f'OpenCV cannot open {video_path} after {i} tries')

    yield cap
    cap.release()


def video_getHW(cap) -> Tuple[int, int]:
    return (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))


def video_open_get_frame(video_path, framenumber):
    with video_capture_open(video_path) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenumber)
        ret, frame_BGR = cap.retrieve()
        if ret == 0:
            raise OSError(f"Can't read frame {framenumber} from {video_path}")
    return frame_BGR


def video_enumerate(cap, sorted_framenumbers):
    """
    Args:
        cap: cv2.VideoCapture class
        sorted_framenumbers: Sorted sequence of framenumbers
    Yields:
        (framenumber, frame_BGR)
    """

    def stop_at_0():
        if ret == 0:
            log.warning('Failed to read frame')
            raise StopIteration()

    sorted_framenumbers = iter(sorted_framenumbers)
    f_current = next(sorted_framenumbers)  # Will stop iteration if empty
    f_next = f_current
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_current)
    while True:
        while f_current <= f_next:  # Iterate till f_current, f_next match
            f_current += 1
            ret = cap.grab()
            stop_at_0()
        ret, frame_BGR = cap.retrieve()
        yield (f_current-1, frame_BGR)
        f_next = next(sorted_framenumbers)
        assert f_current == int(cap.get(cv2.CAP_PROP_POS_FRAMES))


def framenumbers_from_times(cap, times: List[float]) -> List[int]:
    """
        times: timerange in milliseconds
    """
    framenumbers = []
    for _time in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, _time)
        framenumbers.append(int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    return framenumbers

# // CV2 drawing and stuff


def discrete_gaussian(mean_x, mean_y, sigma,
        floor_mean=False,
        ixrange=(-np.inf, np.inf),
        iyrange=(-np.inf, np.inf),
        sigma_number=3):
    """
    Gives discrete gaussian values in
    Args:
        mean_x/mean_y: --
        sigma:  --
        floor_mean: floor the mean position?
        ixrange: Constrain X indices
        iyrange: Constrain Y indices
        sigma_number: How many sigmas around mean to return
    Returns:
        X, Y: meshgrid of indices,
        values: 2D array of gaussian weights
    """
    def gaus_1d(mean, min_, max_):
        center = np.floor(mean) if floor_mean else mean
        inds = np.arange(
                max(min_, np.floor(mean - size)),
                min(max_, np.floor(mean + size)+1))  # Right border non-inclusive
        return inds, scipy.stats.norm.pdf(inds, loc=center, scale=sigma)
    size = sigma*sigma_number
    x_inds, x_vals = gaus_1d(mean_x, *ixrange)
    y_inds, y_vals = gaus_1d(mean_y, *iyrange)
    X, Y = np.meshgrid(y_inds, x_inds)
    values = np.outer(x_vals, y_vals)
    return X.astype(int), Y.astype(int), values
