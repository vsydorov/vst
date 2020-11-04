import collections
import yaml
import logging
import copy
import numpy as np
from typing import (  # NOQA
            Optional, Iterable, List, Dict,
            Any, Union, Callable, TypeVar)

log = logging.getLogger(__name__)

from vst import small


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


class Ydefault(yaml.YAMLObject):
    yaml_tag = '!def'
    argnames = ('default', 'values', 'typecheck', 'evalcheck')

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
        DEFAULTS_TABLE = small.string_table(DEFAULTS_ASSIGNED,
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
        yaml_loaded = yaml.load(merge_from, yaml.Loader)
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

    def without_prefix(self, prefix):
        new_cf = {}
        for k, v in self.cf.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                new_cf[new_k] = v
        return new_cf
