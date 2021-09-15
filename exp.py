#!/usr/bin/env python3
"""
These functions supplement and extend Dervo basic functionality
At the same time Dervo is not required to make use of them
"""
import sys
import importlib
import collections
import yaml
import logging
import copy
import numpy as np
from pathlib import Path
from typing import (  # NOQA
            Optional, Iterable, List, Dict,
            Any, Union, Callable, TypeVar)

from docopt import docopt

import vst

log = logging.getLogger(__name__)


def get_subfolders(folder, subfolder_names=['out', 'temp']):
    return [vst.mkdir(folder/name) for name in subfolder_names]


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


def gir_merge_dicts(user, default):
    """Girschik's dict merge from F-RCNN python implementation"""
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = gir_merge_dicts(user[k], v)
    return user

def unflatten_nested_dict(flat_dict, sep='.', soft=False):
    nested = {}
    for k, v in flat_dict.items():
        set_dd(nested, k, v, sep, soft)
    return nested


def flatten_nested_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ConfigLoader(yaml.SafeLoader):
    pass


class Ydefault(yaml.YAMLObject):
    yaml_tag = '!def'
    argnames = ('default', 'values', 'typecheck', 'evalcheck')
    yaml_loader = [ConfigLoader]

    def __init__(self,
            default=None,
            values: Optional[List] = None,
            typecheck=None,
            evalcheck: str = None,):
        self.default = default
        self.values = values
        self.typecheck = typecheck
        self.evalcheck = evalcheck

    @classmethod
    def from_yaml(cls, loader, node):
        """
        If scalar: assume this is default
        If sequence: assume correspondence to
            [default, values, typecheck, evalcheck]
        if mapping: feed to the constructor directly
        """
        args = {}
        if isinstance(node, yaml.MappingNode):
            x = loader.construct_mapping(node, deep=True)
            for k, v in x.items():
                if k in cls.argnames:
                    args[k] = v
        elif isinstance(node, yaml.SequenceNode):
            x = loader.construct_sequence(node, deep=True)
            for k, v in zip(cls.argnames, x):
                if v is not None:
                    args[k] = v
        elif isinstance(node, yaml.ScalarNode):
            args['default'] = loader.construct_scalar(node)
        else:
            raise RuntimeError()
        ydef = Ydefault(**args)
        return ydef

    def __repr__(self):
        items = [str(self.default)]
        for arg in self.argnames[1:]:
            attr = getattr(self, arg, None)
            if attr is not None:
                items.append(f'{arg}: {attr}')
        s = 'Ydef[{}]'.format(', '.join(items))
        return s


# ConfigLoader.add_constructor('!def', Ydefault)


def _flat_config_merge(merge_into, merge_from, prefix, allow_overwrite):
    assert isinstance(prefix, str)
    for k, v in merge_from.items():
        key = f'{prefix}{k}'
        if key in merge_into and not allow_overwrite:
            raise ValueError('key {} already in {}'.format(
                key, merge_into))
        merge_into[key] = v


def _config_assign_defaults(cf, cf_defaults,
        allowed_wo_defaults=[],
        raise_without_defaults=True):
    # // Assign defaults
    cf_with_defaults = copy.deepcopy(cf)
    assert isinstance(allowed_wo_defaults, list), \
            'Wrong spec for allowed_wo_defaults'

    keys_cf = np.array(list(cf.keys()))
    keys_cf_default = np.array(list(cf_defaults.keys()))
    DEFAULTS_ASSIGNED = []

    # // Are there new keys that were not present in default?
    keys_without_defaults = keys_cf[~np.in1d(keys_cf, keys_cf_default)]
    # Take care of keys that were allowed
    allowed_keys_without_defaults = []
    forbidden_keys_without_defaults = []
    for k in keys_without_defaults:
        allowed = False
        for allowed_prefix in allowed_wo_defaults:
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
        if raise_without_defaults:
            raise ValueError('Keys without defaults')

    # Are there defaults that need to be assigned
    defaults_without_keys = keys_cf_default[~np.in1d(keys_cf_default, keys_cf)]
    if len(defaults_without_keys):
        for k in defaults_without_keys:
            old_value = cf_with_defaults.get(k)
            new_value = cf_defaults[k]
            cf_with_defaults[k] = new_value
            DEFAULTS_ASSIGNED.append((k, old_value, new_value))

    # Are there None values in final config?
    if None in cf_with_defaults.values():
        none_keys = [k for k, v in cf_with_defaults.items() if v is None]
        log.warning('Config keys {} have "None" value after default merge'
                .format(none_keys))

    if len(DEFAULTS_ASSIGNED):
        DEFAULTS_TABLE = vst.string_table(DEFAULTS_ASSIGNED,
                header=['KEY', 'OLD', 'NEW'])
        DEFAULTS_ASSIGNED_STR = 'We assigned some defaults:\n{}'.format(
                DEFAULTS_TABLE)
        log.info(DEFAULTS_ASSIGNED_STR)
    cf = cf_with_defaults
    return cf


class YConfig(object):
    """
    Improved, simplified version of YConfig
    - Helps with validation and default params
    - All configurations stored inside are flat
    """
    def __init__(
            self, cfg_dict,
            allowed_wo_defaults=[],
            raise_without_defaults=True
            ):
        """
         - allowed_wo_defaults - Key substrings that are allowed to exist
           without defaults
        """
        self.cf = flatten_nested_dict(cfg_dict)
        self.ydefaults = {}
        self.allowed_wo_defaults = allowed_wo_defaults
        self.raise_without_defaults = raise_without_defaults

    def set_defaults_yaml(self,
            merge_from: str, prefix='', allow_overwrite=False):
        """ Set defaults from YAML string """
        assert isinstance(merge_from, str)
        yaml_loaded = yaml.load(merge_from, ConfigLoader)
        if not yaml_loaded:
            return
        loaded_flat = flatten_nested_dict(yaml_loaded)
        # Convert everything to Ydefault
        for k, v in loaded_flat.items():
            if not isinstance(v, Ydefault):
                loaded_flat[k] = Ydefault(default=v)
        # Merge into Ydefaults
        _flat_config_merge(self.ydefaults, loaded_flat,
                prefix, allow_overwrite)

    @staticmethod
    def _check_types(cf, ydefaults):
        for k, v in ydefaults.items():
            assert k in cf, f'Parsed key {k} not in {cf}'
            VALUE = cf[k]
            # Values check
            if v.values is not None:
                assert VALUE in v.values, \
                        f'Value {VALUE} for key {k} not in {v.values}'
            # Typecheck
            if v.typecheck is not None:
                good_cls = eval(v.typecheck)
                assert isinstance(VALUE, good_cls), \
                    f'Value {VALUE} for key {k} not of type {good_cls}'
            # Evalcheck
            if v.evalcheck is not None:
                assert eval(v.evalcheck) is True, \
                    f'Value {VALUE} for key {k} does not eval: {v.evalcheck}'

    def parse(self):
        # remove ignored fields
        self.cf = {k: v for k, v in self.cf.items() if v!= '!ignore'}
        cf_defaults = {k: v.default for k, v in self.ydefaults.items()}
        self.cf = _config_assign_defaults(self.cf, cf_defaults,
                self.allowed_wo_defaults, self.raise_without_defaults)
        self._check_types(self.cf, self.ydefaults)
        return self.cf

    def without_prefix(self, prefix, flat=True):
        new_cf = {}
        for k, v in self.cf.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                new_cf[new_k] = v
        if not flat:
            new_cf = unflatten_nested_dict(new_cf, soft=True)
        return new_cf


class UniqueKeyLoader(yaml.SafeLoader):
    # https://gist.github.com/pypt/94d747fe5180851196eb#gistcomment-3401011
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            assert key not in mapping, f'Duplicate key ("{key}") in YAML'
            mapping.append(key)
        return super().construct_mapping(node, deep)


def yml_load(f):
    # We disallow duplicate keys
    cfg = yaml.load(f, UniqueKeyLoader)
    cfg = {} if cfg is None else cfg
    return cfg


def yml_from_file(filepath: Path):
    filepath = Path(filepath)
    try:
        with filepath.open('r') as f:
            return yml_load(f)
    except Exception as e:
        log.info(f'Could not load yml at {filepath}')
        raise e


# Launching dervo scripts


def resolve_clean_exp_path(path_: str) -> Path:
    path = Path(path_)
    assert path.exists(), f'Path must exists: {path}'
    if path.is_file():
        log.warning('File instead of dir was provided, using its parent instead')
        path = path.parent
    path = path.resolve()
    return path


def add_logging_filehandlers(workfolder):
    # Create two output files in /_log subfolder, start loggign
    assert isinstance(logging.getLogger().handlers[0],
            logging.StreamHandler), 'First handler should be StreamHandler'
    logfolder = vst.mkdir(workfolder/'_log')
    id_string = vst.get_experiment_id_string()
    logfilename_debug = vst.add_filehandler(
            logfolder/f'{id_string}.DEBUG.log', logging.DEBUG, 'extended')
    logfilename_info = vst.add_filehandler(
            logfolder/f'{id_string}.INFO.log', logging.INFO, 'short')
    return logfilename_debug, logfilename_info


def extend_path_reload_modules(actual_code_root):
    if actual_code_root is not None:
        # Extend pythonpath to allow importing certain modules
        sys.path.insert(0, str(actual_code_root))
    # Unload caches, to allow local version (if present) to take over
    importlib.invalidate_caches()
    # Reload vst and then submoduless (avoid issues with __init__ imports)
    # https://stackoverflow.com/questions/35640590/how-do-i-reload-a-python-submodule/51074507#51074507
    importlib.reload(vst)
    for k, v in list(sys.modules.items()):
        if k.startswith('vst'):
            log.debug(f'Reload {k} {v}')
            importlib.reload(v)


def import_routine(run_string):
    module_str, experiment_str = run_string.split(':')
    module = importlib.import_module(module_str)
    experiment_routine = getattr(module, experiment_str)
    return experiment_routine


def remove_first_loghandler_before_handling_error(err):
    # Remove first handler(StreamHandler to stderr) to avoid double clutter
    our_logger = logging.getLogger()
    assert len(our_logger.handlers), \
            'Logger handlers are empty for some reason'
    if isinstance(our_logger.handlers[0], logging.StreamHandler):
        our_logger.removeHandler(our_logger.handlers[0])
    log.exception("Fatal error in experiment routine")
    raise err


DERVO_DOC = """
Run dervo experiments without using the whole experimental system

Usage:
    exp.py folder <folder_path> [--nolog] [--] [<add_args> ...]
    exp.py manual --run <str> --cfg <path> [--workfolder <path>] [--code_root <path>] [--nolog] [--] [<add_args> ...]

Options:
    --nolog     Do not log to "workfolder/_log"

    Manual mode:
        --run <str>             Format: module.submodule:function
        --cfg <path>            .yml file containing experimental config
        --workfolder <path>     Workfolder for the experiment.
            Defaults to config folder if not set.
        --code_root <path>      Optional code root to append to the PYTHONPATH

"""


def dervo_run(args):
    # / Figure experimental configuration
    if args['folder']:
        # // Automatically pick up experiment from dervo output folder
        workfolder = resolve_clean_exp_path(args['<folder_path>'])
        # Read _final_cfg.yml that defines the experimental configuration
        ycfg = yml_from_file(workfolder/'_final_cfg.yml')
        # Cannibalize some values from _experiment meta
        actual_code_root = ycfg['_experiment']['code_root']
        run_string = ycfg['_experiment']['run']
    elif args['manual']:
        run_string = args['--run']
        cfg_path = Path(args['--cfg'])
        ycfg = yml_from_file(cfg_path)
        workfolder = vst.npath(args['--workfolder'])
        if workfolder is None:
            workfolder = cfg_path.parent
        actual_code_root = vst.npath(args['--code_root'])
    else:
        raise NotImplementedError()

    # Strip '_experiment meta' from ycfg if present
    if '_experiment' in ycfg:
        del ycfg['_experiment']

    if not args['--nolog']:
        # Create logfilehandlers
        add_logging_filehandlers(workfolder)

    # Deal with imports
    extend_path_reload_modules(actual_code_root)
    experiment_routine = import_routine(run_string)
    try:
        experiment_routine(workfolder, ycfg, args['<add_args>'])
    except Exception as err:
        if not args['--nolog']:
            remove_first_loghandler_before_handling_error(err)
        log.exception("Fatal error in experiment routine")
    log.info('- } Execute experiment routine')


if __name__ == '__main__':
    log = vst.reasonable_logging_setup(logging.INFO)
    dervo_run(docopt(DERVO_DOC))
