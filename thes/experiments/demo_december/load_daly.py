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

import torch
import torch.utils.data
import torch.backends.cudnn as cudnn

from thes.tools import snippets
from thes.data.external_dataset import DatasetDALY

from vsydorov_tools import small

log = logging.getLogger(__name__)


def load_philippe_tubes(workfolder, cfg_dict, add_args):
    dataset = DatasetDALY()
    dataset.precompute_to_folder()
    """
    dataset:
        cache_folder: [~, str]
    """
    import pudb; pudb.set_trace()  # XXX BREAKPOINT
    pass


def precompute_cache(workfolder, cfg_dict, add_args):
    out, = snippets.get_subfolders(workfolder, ['out'])
    cfg = snippets.YConfig(cfg_dict)
    cfg.set_deftype("""
    dataset: [~, ['daly']]
    """)
    cf = cfg.parse()

    if cf['dataset'] == 'daly':
        dataset = DatasetDALY()
    else:
        raise RuntimeError('Wrong dataset')
    dataset.precompute_to_folder(out)
