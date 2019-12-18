"""
Version 0.001
"""
from __future__ import absolute_import

import itertools
import io
import joblib
import math
import sys
import yaml
import copy
import numpy as np  # type: ignore
import functools
import logging
import subprocess
import pickle
import collections
import tqdm

from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from pathlib import Path
from timeit import default_timer as timer  # type: ignore
from typing import Union, Any, NamedTuple, List, Tuple, Callable, TypeVar, Iterator, Iterable, Sequence  # NOQA
from itertools import islice

log = logging.getLogger(__name__)
T = TypeVar('T')

# // Stable //

# //// Nested Dicts ////


def set_dd(d, key, value, sep='.', soft=False):
    """Dynamic assignment to nested dictionary
    http://stackoverflow.com/questions/21297475/set-a-value-deep-in-a-dict-dynamically"""
    dd = d
    keys = key.split(sep)
    latest = keys.pop()
    for k in keys:
        dd = dd.setdefault(k, {})
    if soft:
        dd.setdefault(latest, value)
    else:
        dd[latest] = value


def flatten_nested_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_nested_dict(flat_dict, sep='.'):
    nested = {}
    for k, v in flat_dict.items():
        set_dd(nested, k, v, sep)
    return nested


def gir_merge_dicts(user, default):
    """Girschik's dict merge from F-RCNN python implementation"""
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = gir_merge_dicts(user[k], v)
    return user


def cfg_inherit_defaults(str_default, cfg, strict=False):
    """
    Universal function for setting up default configurations
    """
    cf = flatten_nested_dict(cfg)
    cf_default = flatten_nested_dict(yaml.load(str_default))

    keys_cf = np.array(list(cf.keys()))
    keys_cf_default = np.array(list(cf_default.keys()))

    BAD_STUFF = []
    DEFAULTS_ASSIGNED = []

    # Are there new keys that were not present in default?
    keys_without_defaults = keys_cf[~np.in1d(keys_cf, keys_cf_default)]
    if len(keys_without_defaults):
        with np_printoptions(linewidth=256):
            BAD_STUFF.append('    Config keys {} are missing default values'.format(keys_without_defaults))

    # Are there defaults that are not covered by new keys?
    defaults_without_keys = keys_cf_default[~np.in1d(keys_cf_default, keys_cf)]
    if len(defaults_without_keys):
        for k in defaults_without_keys:
            cf[k] = cf_default[k]
            DEFAULTS_ASSIGNED.append(f'    {k} -> {cf[k]}')

    # Are there None values in final config?
    if None in cf.values():
        BAD_STUFF.append('Config keys {} have "None" value after default merge'
                .format([k for k, v in cf.items() if v is None]))

    # // Afterprocessing
    if len(BAD_STUFF):
        BAD_STUFF_STR = 'Strict config inheritance not possible:\n{}'.format('\n'.join(BAD_STUFF))
        # Strict mode will throw in case of bad stuff
        if strict:
            raise ValueError(BAD_STUFF_STR)
        else:
            log.warning(BAD_STUFF_STR)
    if len(DEFAULTS_ASSIGNED):
        DEFAULTS_ASSIGNED_STR = 'We assigned some defaults:\n{}'.format('\n'.join(DEFAULTS_ASSIGNED))
        # Strict mode will warn in case of defaults assignment
        if strict:
            log.warning(DEFAULTS_ASSIGNED_STR)
        else:
            log.info(DEFAULTS_ASSIGNED_STR)

    return unflatten_nested_dict(cf)


def set_cfg_defaults_strict(str_default, cfg):
    """
    DEPRECATED

    Make sure all 'cfg' key correspond to default config keys
        If new key not in str_default found - throw
    Throw if empty(~) values found
    """
    log.info('Deprecated function (set_cfg_defaults_strict). Please use cfg_inherit_defaults')
    cf = flatten_nested_dict(cfg)
    cf_default = flatten_nested_dict(yaml.load(str_default))

    k_cf = np.array(list(cf.keys()))
    k_cf_default = np.array(list(cf_default.keys()))

    # Are there new keys not present in default? That would be bad
    present_in_default = np.in1d(k_cf, k_cf_default)
    assert present_in_default.all(), \
            f'Keys like {k_cf[np.logical_not(present_in_default)]} missing from default'

    # Are there default keys not present in config? Complain and put them in
    absent_defaults = np.logical_not(np.in1d(k_cf_default, k_cf))
    for k in k_cf_default[absent_defaults]:
        cf[k] = cf_default[k]
        log.warn(f'DEFAULTS: {k} -> {cf[k]}')

    # Are there None values? Throw
    assert None not in cf.values(), \
        f'Keys like {[k for k, v in cf.items() if v is None]} have None value. Not allowed'

    return unflatten_nested_dict(cf)


def set_cfg_defaults(str_default, cfg_):
    """
    DEPRECATED

    Makes sure specific default config keys exist
    If absent:
        - fill default value for the missing key
        - complain about it
    """
    log.info('Deprecated function (set_cfg_defaults). Please use cfg_inherit_defaults')
    cfg = copy.deepcopy(cfg_)
    cfg_default = yaml.load(str_default)
    merged_cfg = gir_merge_dicts(cfg, cfg_default)
    # Verbose description of assigned default values
    fcfg = flatten_nested_dict(cfg_)
    fmerged_cfg = flatten_nested_dict(merged_cfg)
    if fcfg.keys() == fmerged_cfg.keys():
        log.info('DEFAULTS [OK]: Defaults play no role')
    else:
        added_defaults = [(key, value) for key, value in fmerged_cfg.items() if key not in fcfg]
        mkey = max(map(lambda x: len(x[0]), added_defaults))
        log.info('DEFAULTS [NOTE]: Some defaults were added:\n' +
                indent_mstring(('\n'.join([f'{k:{mkey}s} -> {v}' for k, v in added_defaults]))))
    return merged_cfg


# //// Pickling and unpickling ////


"""
    Magic for saving computation
    * Do not pass big arguments
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
        # with filepath.open('rb') as f:
        #     pkl = pickle.load(f)
        with filepath.open('rb') as f:
            pkl_bytes = f.read()
            pkl = pickle.loads(pkl_bytes)
        log.info(f'Unpickled {filepath} in {timer() - start:.2f}s')
    except (EOFError, FileNotFoundError) as e:
        log.info(f'Caught "{e}" error, Computing {function}(*args, **kwargs)')
        pkl = function(*args, **kwargs)
        with filepath.open('wb') as f:
            pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)
        log.info(f'Computed and pickled to {filepath} in {timer() - start:.2f}s')
    return pkl


def compute_or_load_pkl_silently(
        filepath: Union[str, Path],
        function: Callable[..., T],
        *args, **kwargs) -> T:
    """Implementation without outputs"""
    try:
        with Path(filepath).open('rb') as f:
            pkl = pickle.load(f)
    except (EOFError, FileNotFoundError) as e:
        pkl = function(*args, **kwargs)
        with Path(filepath).open('wb') as f:
            pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)
    return pkl


def stash(
        active=True,
        silent=False
            ) -> Callable:
    """
    Returns 'compute_or_load' function. Maybe.
    If active == False function with 'compute_or_load' signature is called.
    P.S. I do very much realize it's a horrible ugly magick
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


# ////// Joblib stashing //////
def jstash(
        active=True,
        silent=False
            ) -> Callable:

    def compute(filepath, function, *args, **kwargs):
        return function(*args, **kwargs)

    def jcompute_or_load_silently(filepath, function, *args, **kwargs):
        try:
            dmp = joblib.load(filepath)
        except (EOFError, FileNotFoundError) as e:
            dmp = function(*args, **kwargs)
            joblib.dump(dmp, filepath)
        return dmp

    def jcompute_or_load(filepath, function, *args, **kwargs):
        start = timer()
        try:
            dmp = joblib.load(filepath)
            log.info(f'Joblib loaded {filepath} in {timer() - start:.2f}s')
        except (EOFError, FileNotFoundError) as e:
            log.info(f'Caught "{e}" error, Computing {function}(*args, **kwargs)')
            dmp = function(*args, **kwargs)
            joblib.dump(dmp, filepath)
            log.info(f'Computed and joblib dumped to {filepath} in {timer() - start:.2f}s')
        return dmp

    if active:
        if silent:
            return jcompute_or_load_silently
        else:
            return jcompute_or_load
    else:
        return compute


# ////// More reasonable pkl routines //////
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
            log.info(f'Skipping.. Failed to load {filepath}')
    raise FileNotFoundError('Failed to load pkl from a list of files', list(filepaths))


def load_py2_pkl(filepath):
    with Path(filepath).resolve().open('rb') as f:
        pkl = pickle.load(f, encoding='latin1')
    return pkl


def save_pkl(filepath, pkl):
    with Path(filepath).open('wb') as f:
        pickle.dump(pkl, f, pickle.HIGHEST_PROTOCOL)


# //// Directory management /////


def mkdir(directory: Path) -> Path:
    """
        Python 3.5 pathlib shortcut to mkdir -p

        Fails if parent is created by other process in the middle of the call
    """
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def mkdir_force(directory: Path) -> Path:
    """
        Repeatedly call mkdir -p until path exists

        Can cause infinite loops
    """
    while True:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            break
        except (FileExistsError, FileNotFoundError) as e:
            log.debug('Caught {}, trying again'.format(e))
    return directory


def force_symlink(linkname, where):
    """ Force symlink creation. If symlink to wrong place - fail """
    linkname = Path(linkname)
    if linkname.is_symlink():
        r_link = linkname.resolve()
        r_where = where.resolve()
        assert r_link == r_where, \
                'Symlink exists, but points to wrong place {} instead of {}'.format(
                    r_link, r_where)
    else:
        for i in range(256):
            try:
                linkname.symlink_to(where)
                break
            except (FileExistsError, FileNotFoundError) as e:
                log.debug('Try {}: Caught {}, trying again'.format(i, e))
            finally:
                log.debug('Managed at try {}'.format(i))


def get_work_subfolder(
        workfolder,
        subfolder,
        allowed_work_subfolders=['out', 'vis', 'temp', 'log']):
    """ Check if allowed name, create if missing """

    if str(subfolder) not in allowed_work_subfolders:
        raise ValueError('Subfolder not allowed {}'.format(subfolder))
    subfolder_path = workfolder/subfolder
    mkdir(subfolder_path)
    return subfolder_path


def get_work_subfolders(workfolder):
    out = get_work_subfolder(workfolder, 'out')
    temp = get_work_subfolder(workfolder, 'temp')
    return out, temp


def get_otv_subfolders(workfolder):
    log.warn('We are phasing out a separate vis folder. Use get_subfolders instead')
    out = get_work_subfolder(workfolder, 'out')
    temp = get_work_subfolder(workfolder, 'temp')
    vis = get_work_subfolder(workfolder, 'vis')
    return out, temp, vis


# //// Context Managers ////


@contextmanager
def logging_disabled(disable_level=logging.CRITICAL):
    """Temporarily disable logging inside context
    http://stackoverflow.com/questions/2266646/how-to-i-disable-and-re-enable-console-logging-in-python
    """
    logging.disable(disable_level)
    yield
    logging.disable(logging.NOTSET)


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
            _file, _line, _function = "(unknown file)", 0, "(unknown function)"
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


class LoggingCapturer(object):
    def __init__(self,
            loglevel=logging.DEBUG,
            pause_other_handlers=False):

        self.loglevel = loglevel
        self.pause_other_handlers = pause_other_handlers
        self._logger = logging.getLogger()

    def __enter__(self):
        self._log_capture_string = io.StringIO()
        if self.pause_other_handlers:
            self._other_handlers = self._logger.handlers.copy()
            for handle in self._logger.handlers:
                self._logger.removeHandler(handle)
        self._temporary_stream_handler = logging.StreamHandler(
                self._log_capture_string)
        self._temporary_stream_handler.setLevel(self.loglevel)
        self._logger.addHandler(self._temporary_stream_handler)
        return self

    def __exit__(self, *args):
        self.captured = self._log_capture_string.getvalue()
        self._logger.removeHandler(self._temporary_stream_handler)
        if self.pause_other_handlers:
            for handle in self._other_handlers:
                self._logger.addHandler(handle)


# //// Others ////


def leqn_split(arr, N):
    """Divide 1d np array into batches of len <= N"""
    return np.array_split(arr, (len(arr)-1)//N + 1)


# // Experimental //


class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None  # type: ignore

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)


@contextmanager
def stdout_redirect_to_tqdm():
    save_stdout = sys.stdout
    try:
        sys.stdout = DummyTqdmFile(sys.stdout)
        yield save_stdout
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout if necessary
    finally:
        sys.stdout = save_stdout


def indent_mstring(string, indent=4):
    """Indent multiline string"""
    return '\n'.join(map(lambda x: ' '*indent+x, string.split('\n')))


def enumerate_mstring(string, indent=4):
    estring = []
    splitted = string.split('\n')
    maxlen = math.floor(math.log(len(splitted), 10))+1
    for ind, line in enumerate(splitted):
        estring.append('{0:{1}d}{2}{3}'.format(
            ind+1, maxlen, ' '*indent, line))
    return '\n'.join(estring)


def series_to_list(series):
    return [(ind, line) for ind, line in series.iteritems()]


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)


def get_matlab_binary():
    matlab_bin = subprocess.check_output(
            'source ~/.bashrc; setenv; which matlab',
            shell=True, executable='/bin/bash').strip().decode()
    return matlab_bin


def SNOdict(cfg):
    """
    Return Sorted Nested OrderedDict

    Use me to get more or less unique representation of CONFIG nested dict
    Note: json module is VERY nice and returns a sorted representation of me
    """
    return OrderedDict([
        (k, SNOdict(cfg[k])
        if type(cfg[k]) is dict else cfg[k])
        for k in sorted(cfg)])


def quick_log_setup(level=logging.INFO):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
            "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def add_pypath(path):
    path = str(path)  # To cover pathlib strings
    if path not in sys.path:
        sys.path.insert(0, path)


def default_to_regular(d):
    """Convert defaultdict to dict
    http://stackoverflow.com/questions/26496831/how-to-convert-defaultdict-of-defaultdicts-of-defaultdicts-to-dict-of-dicts-o"""
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.iteritems()}
    return d


def lazy_property(function):
    """Force lazy evaluation of class properties with @lazy_property
    https://danijar.com/structuring-your-tensorflow-models/"""
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


def temporal_inside(
        range_to_cover: Tuple[int, int],
        covering_range: Tuple[int, int]):

    begin = max(range_to_cover[0], covering_range[0])
    end = min(range_to_cover[1], covering_range[1])
    inter = end-begin+1
    if inter <= 0:
        return 0.0
    else:
        return inter/(range_to_cover[1] - range_to_cover[0] + 1)


Numeric = TypeVar('Numeric', float, int)


def temporal_IOU(
        range_to_cover: Tuple[Numeric, Numeric],
        covering_range: Tuple[Numeric, Numeric]):

    begin = max(range_to_cover[0], covering_range[0])
    end = min(range_to_cover[1], covering_range[1])
    inter = end-begin+1
    if inter <= 0:
        return 0.0
    else:
        union = (range_to_cover[1] - range_to_cover[0] + 1) + \
                (covering_range[1] - covering_range[0] + 1) - inter
        return inter/union


def keyframe_range_shortcut(f_start, f_stop, step, gt_keyframe_numbers):
    f_range = range(f_start, f_stop, step)
    frame_numbers = sorted(set(f_range) | set(gt_keyframe_numbers))
    return frame_numbers

# // Metrics //


def get_voc_precision_recall(y_true_sorted):
    """
    Args:
        y_true_sorted: Array of type bool, sorted by score
    """
    if np.any(y_true_sorted):
        tp = np.cumsum(y_true_sorted)
        fp = np.cumsum(~y_true_sorted)
        rec_ = tp/np.sum(y_true_sorted)
        prec_ = tp/(tp+fp)
        rec_ = np.r_[0, rec_]
        prec_ = np.r_[1, prec_]
    else:
        rec_ = np.r_[0, np.nan]
        prec_ = np.r_[1, 0]
    # Reversed as per sklearn implementation
    return prec_[::-1], rec_[::-1]


def get_sklearn_like_precision_recall(detection_matched, proposal_prob):
    """
    Basically lightweight sklearn implementation
    """
    # Recover y_true_sorted
    proposal_prob_sorted_id = np.argsort(proposal_prob)[::-1]
    proposal_prob_sorted = proposal_prob[proposal_prob_sorted_id]
    y_true_sorted_ = detection_matched[proposal_prob_sorted_id]

    # Remove duplicates but concatenate value for the end of the curve
    distinct_value_indices = np.where(np.diff(proposal_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_sorted_.size-1]

    tp = np.cumsum(y_true_sorted_)[threshold_idxs]
    fp = 1 + threshold_idxs - tp
    rec_ = tp/tp[-1]
    prec_ = tp/(tp+fp)

    # Must stop as soon as recall hits last value
    # Also reverse output and add 0/1
    last_ind = tp.searchsorted(tp[-1])
    sl = slice(last_ind, None, -1)

    return np.r_[prec_[sl], 1], np.r_[rec_[sl], 0]

# // Ad-hoc Tables //


def quick_1row_table(**kwargs):
    """In 3.6+ kwargs order is preserved"""
    header, cells = zip(*kwargs.items())
    str = string_table([cells], header,
            ['{:.2f}' if isinstance(t, float) else'{}' for t in cells])
    return str


def string_table(
        table_rows: List[Iterable],
        header: List[str] = None,
        col_formats: Iterable[str] = itertools.repeat('{}'),
        col_alignments: Iterable[str] = itertools.repeat('<'),
        pad=0,
            ) -> str:
    """ Revisiting the string tables creation"""
    table_rows_s = [[cf.format(i)
        for i, cf in zip(row, col_formats)]
        for row in table_rows]
    if header is not None:
        table_rows_s = [header] + table_rows_s
    widths = map(lambda x: max(map(len, x)), zip(*table_rows_s))
    formats = [f'{{:{a}{w}}}' for w, a in zip(widths, col_alignments)]
    formats = [f'{f:^{pad+len(f)}}' for f in formats]  # Apply padding
    row_format = '|' + '|'.join(formats) + '|'
    table = [row_format.format(*row) for row in table_rows_s]
    return '\n'.join(table)


def df_to_table(df, pad=0):
    return string_table(
            df.reset_index().values,
            ['', ]+df.columns.tolist(), pad=pad)


# // Others //


def index_repr(index):
    return f'{min(index)}:{max(index)}({len(index)})'
