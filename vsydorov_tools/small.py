"""
Module with small snippets
"""
import sys
import logging
import pickle
import numpy as np  # type: ignore
import string
import random
import platform
from datetime import datetime

from pathlib import Path
from timeit import default_timer as timer
from contextlib import contextmanager
from typing import Union, Callable, TypeVar

log = logging.getLogger(__name__)

T = TypeVar('T')


def is_venv():
    # https://stackoverflow.com/questions/1871549/determine-if-python-is-running-inside-virtualenv
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


def mkdir(directory: Path) -> Path:
    """
        Python 3.5 pathlib shortcut to mkdir -p
        Fails if parent is created by other process in the middle of the call
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def add_pypath(path):
    path = str(path)  # To cover pathlib strings
    if path not in sys.path:
        sys.path.insert(0, path)


def get_experiment_id_string():
    time_now = datetime.now()
    str_time = time_now.strftime('%Y-%m-%d_%H-%M-%S')
    str_ms = time_now.strftime('%f')
    str_rnd = str_ms[:3] + ''.join(random.choices(
        string.ascii_uppercase, k=3))
    str_node = platform.node()
    return f'{str_time}_{str_rnd}_{str_node}'


# //// Context Managers ////

# Backwards compatability
from vsydorov_tools.log import logging_disabled  # NOQA


@contextmanager
def np_printoptions(*args, **kwargs):
    """Temporary set numpy printoptions
    with np_printoptions(precision=3, suppress=True):
    http://stackoverflow.com/questions/2891790/pretty-printing-of-numpy-array
    """
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)


class QTimer(object):
    """
    Example:

    with QTimer('Creating roidbs took %(time) sec'):
        or
    with QTimer('Long task'):
    """
    def __init__(self, message=None, enabled=True):
        self.message = message
        self.enabled = enabled

    def __enter__(self):
        self.start = timer()
        # // Get parent stack info
        f = sys._getframe(1)  # Get parent frame
        if hasattr(f, "f_code"):
            co = f.f_code
            _file = co.co_filename.split('/')[-1]
            _line = f.f_lineno
            _function = co.co_name
        else:
            _file, _line, _function = (
                    "(unknown file)", 0, "(unknown function)")
        self.stack_info_str = f'{_file}({_line}){_function}: '
        return self

    def __exit__(self, *args):
        self.end = timer()
        self.time = self.end - self.start

        if self.message and self.enabled:
            message = self.message
            if '%(time)' not in message:
                message += ' took %(time) sec'
            message = self.stack_info_str + message.replace('%(time)',
                f'{self.time:.2f}')
            log.info(message)


"""
    Magic for saving computation
    * Avoid passing big arguments
"""


def compute_or_load_pkl(
        filepath: Union[str, Path],
        function: Callable[..., T],
        *args, **kwargs) -> T:
    """
    Bread and butter of checkpoints
    - If filepath exists - try to load it
    - If filepath does not exist or failed to load - launch function
    """
    filepath = Path(filepath)
    start = timer()
    try:
        with filepath.open('rb') as f:
            pkl_bytes = f.read()
            pkl = pickle.loads(pkl_bytes)
        log.info(f'Unpickled {filepath} in {timer() - start:.2f}s')
    except (EOFError, FileNotFoundError) as e:
        log.debug(f'Caught "{e}" error, Computing {function}')
        pkl = function(*args, **kwargs)
        with filepath.open('wb') as f:
            pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)
        log.info(f'Computed and pickled {function} to {filepath} in {timer() - start:.2f}s')
    return pkl


def compute_or_load_pkl_silently(
        filepath: Union[str, Path],
        function: Callable[..., T],
        *args, **kwargs) -> T:
    """Implementation without outputs"""
    try:
        with Path(filepath).open('rb') as f:
            pkl = pickle.load(f)
    except (EOFError, FileNotFoundError):
        pkl = function(*args, **kwargs)
        with Path(filepath).open('wb') as f:
            pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)
    return pkl


def stash(
        active=True,
        silent=False
            ) -> Callable:
    """
    Returns 'compute_or_load' function. Maybe. Hacky.
    If active == False function with 'compute_or_load' signature is called.
    """
    def compute(filepath, function, *args, **kwargs):
        return function(*args, **kwargs)
    if active:
        if silent:
            return compute_or_load_pkl_silently
        else:
            return compute_or_load_pkl
    else:
        return compute


def stash2(stash_to, active=True, silent=False) -> Callable:
    if silent:
        c_pkl_func = compute_or_load_pkl_silently
    else:
        c_pkl_func = compute_or_load_pkl
    if active:
        def stash_func(function, *args, **kwargs):
            return c_pkl_func(stash_to, function, *args, **kwargs)
    else:
        def stash_func(function, *args, **kwargs):
            return function(*args, **kwargs)
    return stash_func


"""
Reasonable pkl routines
"""


def load_pkl(filepath):
    with Path(filepath).resolve().open('rb') as f:
        pkl = pickle.load(f)
    return pkl


def load_pkl_whichever(*filepaths):
    """Try to unpickle any file from the list"""
    for filepath in filepaths:
        try:
            with Path(filepath).resolve().open('rb') as f:
                pkl = pickle.load(f)
            return pkl
        except (FileNotFoundError, IsADirectoryError) as err:
            log.info(f'Skipping.. Failed to load {filepath}. Error {err}')
    raise FileNotFoundError('Failed to load pkl from a list of files', list(
        filepaths))


def load_py2_pkl(filepath):
    with Path(filepath).resolve().open('rb') as f:
        pkl = pickle.load(f, encoding='latin1')
    return pkl


def save_pkl(filepath, pkl):
    with Path(filepath).open('wb') as f:
        pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)
