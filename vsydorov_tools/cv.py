"""
Helpers for opencv functions
- Opening/closing files
- Rotations and stuff
All images returned in BGR
"""
import numpy as np
import time
import cv2
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple, List

log = logging.getLogger(__name__)


"""
Reading videos (imageio)
"""


def imageio_naive_enumerate(reader, framenumbers):
    for i in framenumbers:
        frame_BGR = reader.get_data(i)[:, :, ::-1]
        yield (i, frame_BGR)


def imageio_sample(reader, framenumbers):
    sorted_framenumber = np.unique(framenumbers)
    reader.set_image_index(sorted_framenumber[0])
    frames_BGR = {}
    for i in sorted_framenumber:
        frames_BGR[i] = reader.get_data(i)[:, :, ::-1]
    sampled_BGR = []
    for i in framenumbers:
        sampled_BGR.append(frames_BGR[i])
    return sampled_BGR


"""
Reading videos
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


def video_open_get_frame(video_path, framenumber):
    with video_capture_open(video_path) as cap:
        cap.set(cv2.CAP_PROP_POS_FRAMES, framenumber)
        ret, frame_BGR = cap.retrieve()
        if ret == 0:
            raise OSError(f"Can't read frame {framenumber} from {video_path}")
    return frame_BGR


def video_getHW(cap) -> Tuple[int, int]:
    return (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))


def video_naive_enumerate(cap, framenumbers):
    for i in framenumbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame_BGR = cap.retrieve()
        yield (i, frame_BGR)


class VideoCaptureError(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
        # self.frame_number = frame_number


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


def read_whole_video(path):
    with video_capture_open(path) as vcap:
        framecount = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = np.array(
            video_sample(vcap, np.arange(framecount)),
            debug_filename=path)
    return frames


"""
Writing videos

video_path = vt_cv.suffix_helper(video_path, 'XVID')
with vt_cv.video_writer_open(
        video_path, sizeWH, 10, 'XVID') as vout:
    ...
    vout.write(im)
"""
# These values work for me
FOURCC_TO_CONTAINER = {
        'VP90': '.webm',
        'XVID': '.avi',
        'MJPG': '.avi',
        'H264': '.mp4',
        'MP4V': '.mp4'
}


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
