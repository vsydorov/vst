import logging
import numpy as np
import cv2  # type: ignore
from collections import namedtuple
from typing import Union, Any, NamedTuple, List, Tuple, Callable, TypeVar, Iterator, Iterable, Sequence  # NOQA

log = logging.getLogger(__name__)

Box_tldr = namedtuple("Box_tldr", "t l d r")
Box_ltrd = namedtuple("Box_ltrd", "l t r d")
BoxTypeUnion = Any


def tbb_up(box, height_width):
    """
    Upscale to image shape, given relative 0-1 coordinates
    Args:
        box: ...
        height_width: [H, W, ...]
    """
    return type(box)(
            t=box.t*height_width[0],
            l=box.l*height_width[1],
            d=box.d*height_width[0],
            r=box.r*height_width[1])


def tbb_relative_rescale(box, scale):
    """ Scale box up/down so both sides are scaled"""
    x = (box.r-box.l+1)
    xfact = x*(scale-1)/2
    y = (box.d-box.t+1)
    yfact = y*(scale-1)/2
    return type(box)(
            t=box.t - yfact,
            l=box.l - xfact,
            d=box.d + yfact,
            r=box.r + xfact)


def tbb_area(box: BoxTypeUnion) -> float:
    return float((box.d-box.t+1)*(box.r-box.l+1))


def tbb_diag(box: Box_ltrd) -> float:
    """ Return diagonal of the box """
    return np.linalg.norm((box.d-box.t, box.r-box.l))


def tbb_inter(
        bb1: BoxTypeUnion,
        bb2: BoxTypeUnion
            ) -> BoxTypeUnion:
    """ Intersection of two bbs
    Returned bb has same type as first argument"""

    return type(bb1)(
            t=max(bb1.t, bb2.t),
            l=max(bb1.l, bb2.l),
            d=min(bb1.d, bb2.d),
            r=min(bb1.r, bb2.r))


def tbb_IOU(
        bb1: BoxTypeUnion,
        bb2: BoxTypeUnion
            ) -> float:

    inter = tbb_inter(bb1, bb2)
    if (inter.t >= inter.d) or (inter.l >= inter.r):
        return 0  # no intersection
    else:
        intersection_area = tbb_area(inter)
        union_area = tbb_area(bb1) + tbb_area(bb2) - intersection_area
        return float(intersection_area)/union_area


def tbb_inside(bbin, bbout):
    """ How much of the bbin area is in bbout
        bbin|bbout overlap divided by bbin area"""
    inter = tbb_inter(bbin, bbout)
    if (inter.t >= inter.d) or (inter.l >= inter.r):
        return 0  # no intersection
    else:
        intersection_area = tbb_area(inter)
        return float(intersection_area)/tbb_area(bbin)


def adjust_tbb_by_image_shape(image_shape, bb):
    """ Prevent bounding box from escaping the image coordinate space
        image -- (height, width, channels)
        bb - Box_* tuple
    """
    img_height, img_width, img_channels = image_shape
    values = np.array(bb)
    max_image_dims = [img_width if key in ['l', 'r'] else img_height for key in bb._fields]
    clipped_values = values.clip(min=0, max=max_image_dims).astype(int)
    clipped_bb = type(bb)(*clipped_values)
    return clipped_bb


def move_tbb(bb, x, y):
    return type(bb)(
            l=bb.l + x,
            t=bb.t + y,
            r=bb.r + x,
            d=bb.d + y)


def apply_tbb(bb, func):
    return type(bb)(
            l=func(bb.l),
            t=func(bb.t),
            r=func(bb.r),
            d=func(bb.d))


def safely_crop_image_by_tbb(image, bb):
    """
    image of shape [height, width, channels]
    """
    fixbb = adjust_tbb_by_image_shape(image.shape, bb)
    return image[
            fixbb.t:fixbb.d,
            fixbb.l:fixbb.r,
            :]


def tbb_cpu_nms(dets, scores, thresh):
    x1 = np.array([d.l for d in dets])
    y1 = np.array([d.t for d in dets])
    x2 = np.array([d.r for d in dets])
    y2 = np.array([d.d for d in dets])
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.array(scores).argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def get_cropbox_ltrd_from_center_cropside(center, cropside):
    """ center -- (x, y) tuple, center in image coordinates
        cropside -- float, cropbox side
    """
    cropping_center = np.array(center)[[0, 1, 0, 1]]
    cropping_tldr = np.array([-1, -1, +1, +1]) * cropside/2
    cropping_bbox = cropping_center + cropping_tldr
    cropbox = Box_ltrd(*cropping_bbox)
    return cropbox


def get_p3_from_bb(bb: BoxTypeUnion) -> np.ndarray:
    """ Get p3 basis from bounding box
    Args:
        bb - Box_* named tuple
    Returns:
        p3 basis:
        0 (0, 0) - 2 (X, 0)
        |
        1 (0, Y)
    """
    p3basis = np.array([
            [bb.l, bb.t],
            [bb.l, bb.d],
            [bb.r, bb.t]], dtype='float32')
    return p3basis


def pad1(p3):
    return np.pad(p3, ((0, 0), (0, 1)), 'constant', constant_values=1)


def rotate_p3(p3, angle):
    """
    We rotate p3 basis (centrally)
    angle - appears to be clockwise
        Why? X,Y confusion probably
    """
    p3center = (p3[2][0]/2, p3[1][1]/2)
    rot = cv2.getRotationMatrix2D(p3center, float(angle), 1.0)
    rotated_p3 = pad1(p3).dot(rot.T).astype('float32')
    return rotated_p3


def get_p3_from_center_cropside_angle(center, cropside, angle):
    """ Used to specify old basis """
    approx_box = get_cropbox_ltrd_from_center_cropside(center, cropside)  # Get bounding box as Box_ltrd
    approx_p3 = get_p3_from_bb(approx_box)  # Get p3 from this box
    rotated_approx_p3 = rotate_p3(approx_p3, 20)
    return rotated_approx_p3


def get_p3_from_height_width(height, width):
    """ Used to define new basis """
    return np.array([[0, 0], [0, height], [width, 0]], dtype='float32')
