import cv2
import pprint
import time
import numpy as np
import logging
import re
import platform
import subprocess
from datetime import datetime
from pathlib import Path

from typing import (  # NOQA
        Iterable, List, Dict, Any)

from vsydorov_tools import small, cv as vt_cv

log = logging.getLogger(__name__)


def qsave_video(filepath, X_BGR):
    X_BGR = np.array(X_BGR)
    assert len(X_BGR.shape) == 4
    H, W = X_BGR.shape[1:3]
    video_path = vt_cv.suffix_helper(filepath, 'XVID')
    with vt_cv.video_writer_open(video_path, (W, H), 10, 'XVID') as vout:
        for frame in X_BGR:
            vout.write(frame)


def cv_put_box_with_text(
        image: np.ndarray,
        box_ltrd: Iterable[float],
        # Rectangle params
        rec_color=(255, 255, 255),  # White
        rec_thickness=4,
        # Text params
        text=None,
        text_size=0.6,
        text_color=None,
        text_thickness=2,
        text_position='left_down'
            ) -> np.ndarray:
    """
    Overwrites in place
    """

    l, t, r, d = map(int, box_ltrd)
    cv2.rectangle(image, (l, t), (r, d),
            color=rec_color,
            thickness=rec_thickness)

    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    if text:
        if text_color is None:
            text_color = rec_color
        # Text Positioning

        retval, baseline = cv2.getTextSize(
                text, fontFace, text_size, text_thickness)
        if text_position == 'left_down':
            text_pos = (l, d-5)
        elif text_position == 'left_up':
            text_pos = (l, t-5)
        elif text_position == 'right_down':
            text_pos = (r-retval[0], d-5)
        elif text_position == 'right_up':
            text_pos = (r-retval[0], t-5)
        else:
            raise ValueError('Wrong text position')
        cv2.putText(
                image,
                text,
                text_pos,
                fontFace=fontFace,
                fontScale=text_size,
                color=text_color,
                thickness=text_thickness)
    return image


class Averager(object):
    """
    Taken from kensh code. Also seen in Gunnar's code
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.last = 0.0
        self.avg = 0.0
        self._sum = 0.0
        self._count = 0.0

    def update(self, value, weight=1):
        self.last = value
        self._sum += value * weight
        self._count += weight
        self.avg = self._sum / self._count

    def __repr__(self):
        return 'Averager[{:.4f} (A: {:.4f})]'.format(self.last, self.avg)


class TicToc(object):
    def __init__(self, names):
        self.names = names
        self.meters = {k: Averager() for k in self.names}
        self.end_times = {k: np.NINF for k in self.names}

    def tic(self, *names):
        for name in names:
            self.end_times[name] = time.time()

    def toc(self, *names):
        for name in names:
            self.meters[name].update(time.time() - self.end_times[name])

    def reset(self, *names):
        if names is None:
            names = self.names
        for name in names:
            self.meters[name].reset()
            self.end_times[name] = np.NINF

    @property
    def time_str(self):
        time_strs = []
        for time_name in self.names:
            time_strs.append('{n}: {m.last:.2f}({m.avg:.2f})s'.format(
                n=time_name, m=self.meters[time_name]))
        time_str = 'Time['+' '.join(time_strs)+']'
        return time_str


def platform_info():
    platform_string = f'Node: {platform.node()}'
    oar_jid = subprocess.run('echo $OAR_JOB_ID', shell=True,
            stdout=subprocess.PIPE).stdout.decode().strip()
    platform_string += ' OAR_JOB_ID: {}'.format(
            oar_jid if len(oar_jid) else 'None')
    platform_string += f' System: {platform.system()} {platform.version()}'
    return platform_string


def get_experiment_id_string():
    import random
    import string
    time_now = datetime.now()
    str_time = time_now.strftime('%Y-%m-%d_%H-%M-%S')
    str_ms = time_now.strftime('%f')
    str_rnd = str_ms[:3] + ''.join(random.choices(
        string.ascii_uppercase, k=3))
    str_node = platform.node()
    return f'{str_time}_{str_rnd}_{str_node}'


def leqn_split(arr, N):
    """Divide 1d np array into batches of len <= N"""
    return np.array_split(arr, (len(arr)-1)//N + 1)


def weighted_array_split(X, weights, N):
    approx_weight_per_split = weights.sum() // N
    approx_split_indices = approx_weight_per_split * np.arange(1, N)
    split_indices = np.searchsorted(
            weights.cumsum(), approx_split_indices)
    X_split = np.array_split(X, split_indices)
    return X_split


def gather_check_all_present(gather_paths, filenames):
    # Check missing
    missing_paths = []
    for path in gather_paths:
        for filename in filenames:
            fpath = Path(path)/filename
            if not fpath.exists():
                missing_paths.append(fpath)
    if len(missing_paths):
        log.error('Some paths are MISSING:\n{}'.format(
            pprint.pformat(missing_paths)))
        return False
    return True


def get_subfolders(folder, subfolder_names=['out', 'temp']):
    return [small.mkdir(folder/name) for name in subfolder_names]

def tqdm_str(pbar, ninc=0):
    if pbar is None:
        tqdm_str = ''
    else:
        tqdm_str = 'TQDM[' + pbar.format_meter(
                pbar.n + ninc, pbar.total,
                pbar._time()-pbar.start_t) + ']'
    return tqdm_str


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
            ACTION |= check_step_sslice(i, self.sslice)
            if self.iters is not None:
                ACTION |= (i - self._i_last) >= self.iters

        if self.seconds is not None:
            time_since_last = time.perf_counter() - self._time_last
            ACTION |= time_since_last >= self.seconds
        return ACTION

# SSLICE


def check_step_sslice(
        step: int,
        period_sslice: str) -> bool:
    """
    Check whether step matches SSLICE spec

    SSLICE spec: '(more_runs):(run_limit):(period)'
      - more_runs: (csv list of runs when we should fire)
      - run_limit: [MIN],[MAX] (inclusive, don't fire beyond)
      - period: (fire at period intervals)
    """
    spec_re = r'^([\d,]*):((?:\d*,\d*)?):([\d]*)$'
    match = re.fullmatch(spec_re, period_sslice)
    if match is None:
        raise ValueError(f'Invalid spec {period_sslice}')
    _more_runs, run_limit, _period = match.groups()

    if len(run_limit):
        lmin, lmax = run_limit.split(',')
        if lmin and step < int(lmin):
            return False
        if lmax and step > int(lmax):
            return False
    if _more_runs and step in map(int, _more_runs.split(',')):
        return True
    if _period and step and (step % int(_period) == 0):
        return True
    return False


def get_period_actions(step: int, period_specs: Dict[str, str]):
    period_actions = {}
    for action, period_spec in period_specs.items():
        period_actions[action] = check_step_sslice(step, period_spec)
    return period_actions
