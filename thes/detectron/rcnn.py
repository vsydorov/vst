import numpy as np
from types import MethodType

from vsydorov_tools import small

import torch
from torch.nn import functional as F

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as d2_transforms
from detectron2.structures import Boxes, Instances


from thes.detectron.cfg import (
    set_detectron_cfg_base, set_detectron_cfg_test,)
from thes.detectron.externals import (simple_d2_setup,)


def genrcnn_rcnn_roiscores_forward(self, batched_inputs):
    """
    Replacing detectron2/detectron2/modeling/meta_arch/rcnn.py (GeneralizedRCNN.forward)
    """
    assert not self.training
    images = self.preprocess_image(batched_inputs)
    features = self.backbone(images.tensor)
    assert "proposals" in batched_inputs[0]
    proposals = [x["proposals"].to(self.device) for x in batched_inputs]
    del images
    # Borrowed from detectron2/detectron2/modeling/roi_heads/roi_heads.py (Res5ROIHeads.forward)
    proposal_boxes = [x.proposal_boxes for x in proposals]
    box_features = self.roi_heads._shared_roi_transform(
        [features[f] for f in self.roi_heads.in_features], proposal_boxes
    )
    feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
    pred_class_logits, pred_proposal_deltas = \
            self.roi_heads.box_predictor(feature_pooled)
    pred_softmax = F.softmax(pred_class_logits, dim=-1)
    return pred_softmax


class D2_rcnn_helper(object):
    def __init__(self, cf, cf_add_d2, dataset, out):
        num_classes = len(dataset.action_names)
        TEST_DATASET_NAME = 'daly_objaction_test'

        # / Define d2 conf
        d2_output_dir = str(small.mkdir(out/'d2_output'))
        d_cfg = set_detectron_cfg_base(
                d2_output_dir, num_classes, cf['seed'])
        d_cfg = set_detectron_cfg_test(
                d_cfg, TEST_DATASET_NAME,
                cf['d2_rcnn.model'], cf['d2_rcnn.conf_thresh'], cf_add_d2,
                freeze=False)
        d_cfg.MODEL.PROPOSAL_GENERATOR.NAME = "PrecomputedProposals"
        d_cfg.freeze()

        # / Start d2
        simple_d2_setup(d_cfg)

        # Predictor without proposal generator
        model = build_model(d_cfg)
        model.eval()
        checkpointer = DetectionCheckpointer(model)

        checkpointer.load(d_cfg.MODEL.WEIGHTS)
        MIN_SIZE_TEST = d_cfg.INPUT.MIN_SIZE_TEST
        MAX_SIZE_TEST = d_cfg.INPUT.MAX_SIZE_TEST
        transform_gen = d2_transforms.ResizeShortestEdge(
            [MIN_SIZE_TEST, MIN_SIZE_TEST], MAX_SIZE_TEST)

        # Instance monkeypatching
        # https://stackoverflow.com/questions/50599045/python-replacing-a-function-within-a-class-of-a-module/50600307#50600307
        model.forward = MethodType(genrcnn_rcnn_roiscores_forward, model)

        self.d_cfg = d_cfg
        self.rcnn_roiscores_model = model
        self.cpu_device = torch.device("cpu")
        self.transform_gen = transform_gen

    def score_boxes(self, frame_BGR, boxes) -> np.ndarray:
        o_height, o_width = frame_BGR.shape[:2]
        got_transform = self.transform_gen.get_transform(frame_BGR)
        # Transform image
        image = got_transform.apply_image(frame_BGR)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        imshape = tuple(image.shape[1:3])
        # Transform box
        t_boxes = torch.as_tensor(boxes.astype("float32"))
        transformed_t_boxes = got_transform.apply_box(t_boxes)
        # Proposals w.r.t transformed imagesize
        proposal = Instances(imshape)
        tb_boxes = Boxes(transformed_t_boxes)
        proposal.proposal_boxes = tb_boxes
        inputs = {
                "image": image,
                "proposals": proposal,
                "height": o_height,
                "width": o_width}
        with torch.no_grad():
            pred_softmax = self.rcnn_roiscores_model([inputs])
        X = pred_softmax.to(self.cpu_device).numpy()
        # To conform to caffe style put background cls at 0th position
        X_caffelike = np.c_[X[:, -1:], X[:, :-1]]
        return X_caffelike
