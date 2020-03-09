import argparse
import os
import yacs
import copy
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, cast

from vsydorov_tools import small

import torch

from detectron2.engine import (
        DefaultTrainer, hooks)
from detectron2.data import (
        DatasetCatalog, MetadataCatalog,
        build_detection_train_loader,
        build_detection_test_loader)
from detectron2.engine.defaults import DefaultPredictor
from detectron2.layers.nms import nms, batched_nms

from thes.data.external_dataset import (
        DatasetDALY, DALY_action_name, DALY_object_name)
from thes.tools import snippets
from thes.det2 import (
        launch_w_logging, simple_d2_setup,
        base_d2_frcnn_config, set_d2_cthresh, d2dict_gpu_scaling,
        D2DICT_GPU_SCALING_DEFAULTS,
        get_frame_without_crashing)
from thes.daly_d2 import (simplest_daly_to_datalist,
        DalyVideoDatasetMapper)
from thes.eval_tools import legacy_evaluation


log = logging.getLogger(__name__)


def train_d2_dalyobj_hacky(workfolder, cfg_dict, add_args):
    raise NotImplementedError('Refactoring in progress')


def eval_d2_dalyobj_hacky(workfolder, cfg_dict, add_args):
    raise NotImplementedError('Refactoring in progress')
