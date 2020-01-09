import logging
from tqdm import tqdm
from abc import abstractmethod, ABC
import re
import collections
import pprint
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from mypy_extensions import TypedDict

from thes.tools import snippets
from thes.data.external_dataset import DatasetVOC2007
from demo_jan import ender_rcnn 

from vsydorov_tools import small

log = logging.getLogger(__name__)


def load_voc2007(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    imported_wein_tubes: [~, str]
    dataset:
        name: [~, ['voc2007']]
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    dataset = DatasetVOC2007()
    dataset.populate_from_folder(cf['dataset.cache_folder'])


def eval_endernewton_rcnn(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    imported_wein_tubes: [~, str]
    dataset:
        name: [~, ['voc2007']]
        cache_folder: [~, str]
    """)
    cf = cfg.parse()
    dataset = DatasetVOC2007()
    dataset.populate_from_folder(cf['dataset.cache_folder'])
