import yacs
import numpy as np

from detectron2.config import get_cfg
from fvcore.common.config import CfgNode
from detectron2.config.config import CfgNode as CfgNode_d2

from thes.tools import snippets


PRETRAINED_WEIGHTS_MODELPATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/horus/pytorch_model_zoo/pascal_voc_baseline/model_final_b1acc2.pkl'


# Base-RCNN-C4.yaml
YAML_Base_RCNN_C4 = """
MODEL:
  META_ARCHITECTURE: "GeneralizedRCNN"
  RPN:
    PRE_NMS_TOPK_TEST: 6000
    POST_NMS_TOPK_TEST: 1000
  ROI_HEADS:
    NAME: "Res5ROIHeads"
DATASETS:
  TRAIN: ("coco_2017_train",)
  TEST: ("coco_2017_val",)
SOLVER:
  IMS_PER_BATCH: 16
  BASE_LR: 0.02
  STEPS: (60000, 80000)
  MAX_ITER: 90000
INPUT:
  MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
VERSION: 2
"""

# faster_rcnn_R_50_C4.yaml
YAML_faster_rcnn_R_50_C4 = """
MODEL:
  WEIGHTS: "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
  MASK_ON: False
  RESNETS:
    DEPTH: 50
  ROI_HEADS:
    NUM_CLASSES: 20
INPUT:
  MIN_SIZE_TRAIN: (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
  MIN_SIZE_TEST: 800
DATASETS:
  TRAIN: ('voc_2007_trainval', 'voc_2012_trainval')
  TEST: ('voc_2007_test',)
SOLVER:
  STEPS: (12000, 16000)
  MAX_ITER: 18000  # 17.4 epochs
  WARMUP_ITERS: 100
"""


def base_d2_frcnn_config():
    d_cfg = get_cfg()
    # Base keys
    loaded_cfg = yacs.config.CfgNode.load_cfg(YAML_Base_RCNN_C4)
    d_cfg.merge_from_other_cfg(loaded_cfg)
    # FRCCN keys
    loaded_cfg = yacs.config.CfgNode.load_cfg(YAML_faster_rcnn_R_50_C4)
    d_cfg.merge_from_other_cfg(loaded_cfg)
    return d_cfg


def cf_to_cfgnode(cf):
    """ Flat config to detectron2 confignode """
    cn = CfgNode(
        snippets.unflatten_nested_dict(cf), [])
    return cn


def yacs_merge_additional_keys(d_cfg, cf_add_d2):
    d_cfg.merge_from_other_cfg(cf_to_cfgnode(cf_add_d2))
    return d_cfg


def set_d2_cthresh(d_cfg, CONFIDENCE_THRESHOLD=0.25):
    d_cfg.MODEL.RETINANET.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD
    d_cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = \
            CONFIDENCE_THRESHOLD


D2DICT_GPU_SCALING_DEFAULTS = """
gpu_scaling:
    enabled: True
    base:
        # more per gpu
        SOLVER.BASE_LR: 0.0025
        SOLVER.IMS_PER_BATCH: 2
        # less per gpu
        SOLVER.MAX_ITER: 18000
        SOLVER.STEPS: [12000, 16000]
"""


def d2dict_gpu_scaling(cf, cf_add_d2, num_gpus):
    imult = 8 / num_gpus
    prepared = {}
    prepared['SOLVER.BASE_LR'] = \
        cf['gpu_scaling.base.SOLVER.BASE_LR'] * num_gpus
    prepared['SOLVER.IMS_PER_BATCH'] = \
        cf['gpu_scaling.base.SOLVER.IMS_PER_BATCH'] * num_gpus
    prepared['SOLVER.MAX_ITER'] = \
        int(cf['gpu_scaling.base.SOLVER.MAX_ITER'] * imult)
    prepared['SOLVER.STEPS'] = tuple((np.array(
            cf['gpu_scaling.base.SOLVER.STEPS']
            ) * imult).astype(int).tolist())
    cf_add_d2.update(prepared)
    return cf_add_d2


def set_detectron_cfg_base(d2_output_dir, num_classes, seed):
    d_cfg = base_d2_frcnn_config()
    d_cfg.OUTPUT_DIR = d2_output_dir
    d_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    d_cfg.SEED = seed
    return d_cfg


def set_detectron_cfg_train(
        d_cfg, DATASET_NAME, cf_add_d2,
        freeze=True):
    d_cfg.MODEL.WEIGHTS = PRETRAINED_WEIGHTS_MODELPATH
    d_cfg.DATASETS.TRAIN = (DATASET_NAME, )
    d_cfg.DATASETS.TEST = ()
    yacs_merge_additional_keys(d_cfg, cf_add_d2)
    if freeze:
        d_cfg.freeze()
    return d_cfg


def set_detectron_cfg_test(
        d_cfg, DATASET_NAME,
        model_weights, conf_thresh, cf_add_d2,
        freeze=True):
    d_cfg.MODEL.WEIGHTS = model_weights
    d_cfg.DATASETS.TRAIN = ()
    d_cfg.DATASETS.TEST = (DATASET_NAME, )
    set_d2_cthresh(d_cfg, conf_thresh)
    yacs_merge_additional_keys(d_cfg, cf_add_d2)
    if freeze:
        d_cfg.freeze()
    return d_cfg
