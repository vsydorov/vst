#!/usr/bin/env python3
"""
A few helper functions for yaml parsing
"""

import collections
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def gir_merge_dicts(user, default):
    """Girschik's dict merge from F-RCNN python implementation"""
    if isinstance(user, dict) and isinstance(default, dict):
        for k, v in default.items():
            if k not in user:
                user[k] = v
            else:
                user[k] = gir_merge_dicts(user[k], v)
    return user


def set_dd(d, key, value, sep=".", soft=False):
    """Dynamic assignment to nested dictionary
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


def get_dd(d, key, sep="."):
    # Dynamic query from a nested dictonary
    dd = d
    keys = key.split(sep)
    latest = keys.pop()
    for k in keys:
        dd = dd[k]
    return dd[latest]


def unflatten_nested_dict(flat_dict, sep=".", soft=False):
    nested = {}
    for k, v in flat_dict.items():
        set_dd(nested, k, v, sep, soft)
    return nested


def flatten_nested_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def flatten_nested_dict_v2(d, parent_key="", sep=".", keys_to_ignore=[]):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if new_key in keys_to_ignore:
            items.append((new_key, v))
        elif isinstance(v, collections.abc.MutableMapping):
            items.extend(
                flatten_nested_dict_v2(
                    v, new_key, sep=sep, keys_to_ignore=keys_to_ignore
                ).items()
            )
        else:
            items.append((new_key, v))
    return dict(items)


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
        with filepath.open("r") as f:
            return yml_load(f)
    except Exception as e:
        log.info(f"Could not load yml at {filepath}")
        raise e
