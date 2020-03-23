import cv2
import pandas as pd
import yaml
import time
import copy
import collections
import numpy as np
import logging
import re
import itertools
import platform
import subprocess
from abc import abstractmethod, ABC  # NOQA
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from typing import Iterable, List, Dict

from vsydorov_tools import small

log = logging.getLogger(__name__)


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
    result = cv2.rectangle(
            image,
            (l, t), (r, d),
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
    return result


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


def check_step_v2(
        step: int,
        period_spec: str):
    """
    Check whether step matches SPEC (simplified)
    - SPEC: '(more_runs):(run_limit):(period)'
        - more_runs: (csv list of runs when we should fire)
        - run_limit: [MIN],[MAX] (inclusive, don't fire beyond)
        - period: (fire at period intervals)
    """
    spec_re = r'^([\d,]*):((?:\d*,\d*)?):([\d]*)$'
    match = re.fullmatch(spec_re, period_spec)
    if match is None:
        raise ValueError(f'Invalid spec {period_spec}')
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


def get_period_actions(step: int, period_specs: Dict[str, str]):
    period_actions = {}
    for action, period_spec in period_specs.items():
        period_actions[action] = check_step_v2(step, period_spec)
    return period_actions


# == String tables ==


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
    widths = []
    for x in zip(*table_rows_s):
        widths.append(max([len(y) for y in x]))
    formats = [f'{{:{a}{w}}}' for w, a in zip(widths, col_alignments)]
    formats = [f'{f:^{pad+len(f)}}' for f in formats]  # Apply padding
    row_format = '|' + '|'.join(formats) + '|'
    table = [row_format.format(*row) for row in table_rows_s]
    return '\n'.join(table)


def df_to_table_v1(df, pad=0):
    return string_table(
            df.reset_index().values,
            ['', ]+df.columns.tolist(), pad=pad)


def df_to_table_v2(df: pd.DataFrame, indexname=None) -> str:
    # Header
    if indexname is None:
        indexname = df.index.name
    if indexname is None:
        indexname = 'index'
    header = [indexname, ] + [str(x) for x in df.columns]
    # Col formats
    col_formats = ['{}']
    for dt in df.dtypes:
        form = '{}'
        if dt in ['float32', 'float64']:
            form = '{:.2f}'
        col_formats.append(form)

    table = string_table(
            np.array(df.reset_index()),
            header=header,
            col_formats=col_formats,
            pad=2)
    return table


# = Configuration =


def get_subfolders(folder, subfolder_names=['out', 'temp']):
    return [small.mkdir(folder/name) for name in subfolder_names]


def flatten_nested_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def set_dd(d, key, value, sep='.', soft=False):
    """
    Dynamic assignment to nested dictionary
    http://stackoverflow.com/questions/21297475/set-a-value-deep-in-a-dict-dynamically
    """
    dd = d
    keys = key.split(sep)
    latest = keys.pop()
    for k in keys:
        dd = dd.setdefault(k, {})
    if soft:
        dd.setdefault(latest, value)
    else:
        dd[latest] = value


def unflatten_nested_dict(flat_dict, sep='.'):
    nested = {}
    for k, v in flat_dict.items():
        set_dd(nested, k, v, sep)
    return nested


def yml_cast_to_dict(merge_from):
    if isinstance(merge_from, str):
        merge_from = flatten_nested_dict(yaml.safe_load(merge_from))
    elif isinstance(merge_from, dict):
        pass
    else:
        raise ValueError('Unusual type {} for'
                ' merge_from'.format(type(merge_from)))
    return merge_from


def cfg_without_prefix(cf, prefix, keys=None):
    """
    Returns dict keys without prefix
        If keys is None - get all keys with that prefix
        If keys is set - get only matching ones
    """
    good_prefix = []
    for k in cf.keys():
        if k.startswith(prefix):
            good_prefix.append(k[len(prefix):])

    cf_no_prefix = {}
    if keys is None:
        matched_keys = good_prefix
    else:
        matched_keys = [k for k in keys if k in good_prefix]

    for k in matched_keys:
        value = cf.get(prefix+k)
        cf_no_prefix[k] = value
    return cf_no_prefix


class YConfig(object):
    """
    Helps with validation and default params

    - All configurations stored inside are flat
    """

    def __init__(self, cfg_dict):
        self.cf = flatten_nested_dict(cfg_dict)
        self.cf_defaults = {}
        self.typechecks = {}
        self.allowed_wo_defaults = []
        self.raise_without_defaults = True

    def set_defaults_handling(self,
            allowed_wo_defaults=[],
            raise_without_defaults=True):
        # Key substrings that are allowed to exist without defaults
        self.allowed_wo_defaults = allowed_wo_defaults
        self.raise_without_defaults = raise_without_defaults

    @staticmethod
    def _flat_merge(merge_into, merge_from, prefix, allow_overwrite):
        assert isinstance(prefix, str)
        for k, v in merge_from.items():
            key = f'{prefix}{k}'
            if key in merge_into and not allow_overwrite:
                raise ValueError('key {} already in {}'.format(
                    key, merge_into))
            merge_into[key] = v

    def remove_keys(self, keys):
        for key in keys:
            if key in self.cf_defaults:
                del self.cf_defaults[key]
            if key in self.cf:
                del self.cf[key]
            if key in self.typechecks:
                del self.typechecks[key]

    def set_defaults(self, merge_from, prefix='', allow_overwrite=False):
        merge_from_ = yml_cast_to_dict(merge_from)
        self._flat_merge(self.cf_defaults, merge_from_,
                prefix, allow_overwrite)

    def set_typecheck(self, merge_from, prefix='', allow_overwrite=False):
        merge_from_ = yml_cast_to_dict(merge_from)
        self._flat_merge(self.typechecks, merge_from_,
                prefix, allow_overwrite)

    def set_deftype(self, merge_from, prefix='', allow_overwrite=False):
        """ Both defaults and types """
        merge_from_ = yml_cast_to_dict(merge_from)
        # Split in def/type
        merge_from_def = {k: v[0] for k, v in merge_from_.items()}
        merge_from_type = {k: v[1] for k, v in merge_from_.items()}
        self._flat_merge(self.cf_defaults, merge_from_def,
                prefix, allow_overwrite)
        self._flat_merge(self.typechecks, merge_from_type,
                prefix, allow_overwrite)

    def _assign_defaults(self):
        # // Assign defaults
        cf_with_defaults = copy.deepcopy(self.cf)

        keys_cf = np.array(list(self.cf.keys()))
        keys_cf_default = np.array(list(self.cf_defaults.keys()))
        DEFAULTS_ASSIGNED = []

        # // Are there new keys that were not present in default?
        keys_without_defaults = keys_cf[~np.in1d(keys_cf, keys_cf_default)]
        # Take care of keys that were allowed
        allowed_keys_without_defaults = []
        forbidden_keys_without_defaults = []
        for k in keys_without_defaults:
            allowed = False
            for allowed_prefix in self.allowed_wo_defaults:
                if k.startswith(allowed_prefix):
                    allowed = True
            if allowed:
                allowed_keys_without_defaults.append(k)
            else:
                forbidden_keys_without_defaults.append(k)
        if len(allowed_keys_without_defaults):
            log.info('Some keys were allowed to '
                    'exist without defaults: {}'.format(
                        allowed_keys_without_defaults))
        # Complain about forbidden ones
        if len(forbidden_keys_without_defaults):
            for k in forbidden_keys_without_defaults:
                log.info(f'ERROR: Key {k} has no default value')
            if self.raise_without_defaults:
                raise ValueError('Keys without defaults')

        # Are there defaults that need to be assigned
        defaults_without_keys = keys_cf_default[~np.in1d(keys_cf_default, keys_cf)]
        if len(defaults_without_keys):
            for k in defaults_without_keys:
                old_value = cf_with_defaults.get(k)
                new_value = self.cf_defaults[k]
                cf_with_defaults[k] = new_value
                DEFAULTS_ASSIGNED.append((k, old_value, new_value))

        # Are there None values in final config?
        if None in cf_with_defaults.values():
            none_keys = [k for k, v in cf_with_defaults.items() if v is None]
            log.warning('Config keys {} have "None" value after default merge'
                    .format(none_keys))

        if len(DEFAULTS_ASSIGNED):
            DEFAULTS_TABLE = string_table(DEFAULTS_ASSIGNED,
                    header=['KEY', 'OLD', 'NEW'])
            DEFAULTS_ASSIGNED_STR = 'We assigned some defaults:\n{}'.format(
                    DEFAULTS_TABLE)
            log.info(DEFAULTS_ASSIGNED_STR)
        self.cf = cf_with_defaults

    def _check_types(self):
        cf = self.cf
        for k, v in self.typechecks.items():
            assert k in cf, f'Parsed key {k} not in {cf}'
            VALUE = cf[k]
            if isinstance(v, list):
                assert VALUE in v, f'Value {VALUE} for key {k} not in {v}'
            elif v in ['int', 'float', 'str', 'bool', 'list']:
                good_cls = eval(v)
                assert isinstance(VALUE, good_cls), \
                    f'Value {VALUE} for key {k} not of type {good_cls}'
            elif isinstance(v, collections.abc.Iterable) and 'VALUE' in v:
                assert eval(v) is True, \
                    f'Value {VALUE} for key {k} does not eval: {v}'
            elif v is None:
                # No check when none value
                continue
            else:
                raise NotImplementedError(f'Parsing {k} -> {v} is undefined')

    def _remove_ignored_fields(self):
        self.cf = {k: v for k, v in self.cf.items() if v!= '!ignore'}

    def parse(self):
        self._remove_ignored_fields()
        self._assign_defaults()
        self._check_types()
        return self.cf

    def without_prefix(self, prefix, keys=None):
        return cfg_without_prefix(self.cf, prefix, keys)


class Base_isaver(ABC):
    def __init__(self, folder, total):
        self._re_finished = (
            r'item_(?P<i>\d+)_of_(?P<N>\d+).finished')
        self._fmt_finished = 'item_{:04d}_of_{:04d}.finished'
        self._history_size = 3

        self._folder = folder
        self._total = total

    def _get_filenames(self, i) -> Dict[str, Path]:
        base_filenames = {
            'finished': self._fmt_finished.format(i, self._total)}
        base_filenames['pkl'] = Path(base_filenames['finished']).with_suffix('.pkl')
        filenames = {k: self._folder/v for k, v in base_filenames.items()}
        return filenames

    def _get_intermediate_files(self) -> Dict[int, Dict[str, Path]]:
        """Check re_finished, query existing filenames"""
        intermediate_files = {}
        for ffilename in self._folder.iterdir():
            matched = re.match(self._re_finished, ffilename.name)
            if matched:
                i = int(matched.groupdict()['i'])
                # Check if filenames exist
                filenames = self._get_filenames(i)
                all_exist = all([v.exists() for v in filenames.values()])
                assert ffilename == filenames['finished']
                if all_exist:
                    intermediate_files[i] = filenames
        return intermediate_files

    def _purge_intermediate_files(self):
        """Remove old saved states"""
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        inds_to_purge = np.sort(np.fromiter(
            intermediate_files.keys(), np.int))[:-self._history_size]
        files_purged = 0
        for ind in inds_to_purge:
            filenames = intermediate_files[ind]
            for filename in filenames.values():
                filename.unlink()
                files_purged += 1
        log.debug('Purged {} states, {} files'.format(
            len(inds_to_purge), files_purged))


def _tqdm_str(pbar, ninc=0):
    if pbar is None:
        tqdm_str = ''
    else:
        tqdm_str = 'TQDM[' + pbar.format_meter(
                pbar.n + ninc, pbar.total,
                pbar._time()-pbar.start_t) + ']'
    return tqdm_str


class Simple_isaver(Base_isaver):
    """
    Will process a list with a func
    """
    def __init__(self, folder, in_list, func,
            save_period='::25',
            log_interval_seconds=-1):
        # assert sys.version_info >= (3, 6), 'Dicts must keep insertion order'
        super().__init__(folder, len(in_list))
        self.in_list = in_list
        self.result = []
        self.func = func
        self._save_period = save_period
        self._log_interval_seconds = log_interval_seconds

    def _restore(self):
        intermediate_files: Dict[int, Dict[str, Path]] = \
                self._get_intermediate_files()
        start_i, ifiles = max(intermediate_files.items(),
                default=(-1, None))
        if ifiles is not None:
            restore_from = ifiles['pkl']
            self.result = small.load_pkl(restore_from)
            log.info('Restore from {}'.format(restore_from))
        return start_i

    def _save(self, i):
        ifiles = self._get_filenames(i)
        savepath = ifiles['pkl']
        small.save_pkl(savepath, self.result)
        ifiles['finished'].touch()

    def run(self):
        start_i = self._restore()
        run_range = np.arange(start_i+1, self._total)
        self._last_time = time.perf_counter()
        pbar = tqdm(run_range)
        for i in pbar:
            self.result.append(self.func(self.in_list[i]))
            if check_step_v2(i, self._save_period) or \
                    (i+1 == self._total):
                self._save(i)
                self._purge_intermediate_files()
            if self._log_interval_seconds != -1:
                if (time.perf_counter() - self._last_time) > \
                        self._log_interval_seconds:
                    log.info(_tqdm_str(pbar))
        return self.result
