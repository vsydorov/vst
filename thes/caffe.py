import logging
import os
import cv2
import numpy as np
from pathlib import Path

from vsydorov_tools import small

log = logging.getLogger(__name__)


# NICPHIL_RCNN_CAFFE_PATH = '/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/30_nicolas_rcnn_hackery/caffe/py-faster-rcnn/caffe-fast-rcnn/python'
NICPHIL_RCNN_CAFFE_PATH = Path('/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/40_recompiled_caffe/caffe/py-faster-rcnn/caffe-fast-rcnn/python')
NICPHIL_RCNN_MODEL_PATH = Path('/home/vsydorov/projects/deployed/2019_12_Thesis/links/scratch2/30_nicolas_rcnn_hackery/models')


def revive_nicolas_caffe():
    caffe_root = NICPHIL_RCNN_CAFFE_PATH
    small.add_pypath(caffe_root)
    os.environ['GLOG_minloglevel'] = '3'  # Stop caffe outputs
    import caffe  # type: ignore
    caffe.set_device(0)
    caffe.set_mode_gpu()
    return caffe


def nicolas_net():
    caffe = revive_nicolas_caffe()
    nico_root = NICPHIL_RCNN_MODEL_PATH
    model_weights = nico_root/'net_VGG16_iter_70000PHIL.caffemodel'
    model_proto = nico_root/'net_VGG16_test.prototxt'

    net = caffe.Net(str(model_proto), str(model_weights), caffe.TEST)
    return net


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def model_test_get_image_blob(im, PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True) - PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def get_scores_per_frame_RGB(
        net, tube, frames_u8,
        PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE):
    """
    Score per frame for NIC RGB
    """
    tube_scores = []
    for i, frame_BGR in enumerate(frames_u8):
        # Image resizing
        blob_, im_scale_factors = model_test_get_image_blob(
                frame_BGR, PIXEL_MEANS, TEST_SCALES, TEST_MAX_SIZE)
        blob = blob_.transpose(0, 3, 1, 2)  # 1, H, W, 3 --> 1, 3, H, W
        im_scale_factor = im_scale_factors[0]

        net.blobs['data'].reshape(*blob.shape)
        net.blobs['data'].data[...] = blob

        # Put in the sole ROI
        box = tube['boxes'][i]
        box5 = np.r_[0, box]
        box5 = box5 * im_scale_factor
        _, l, t_, r, d = map(int, box5)
        net.blobs['rois'].reshape(1, 5)
        net.blobs['rois'].data[...] = box5
        cls_prob = net.forward()['cls_prob']
        """
        import cv2
        imshow(np.clip(blob_[0] + PIXEL_MEANS, 1, 254)/255)
        im2 = cv2.rectangle(im.copy(), (l, t_), (r, d), (255, 255, 0), thickness=2)
        # printing
        """
        scores = cls_prob.flatten()
        tube_scores.append(scores)
    return tube_scores
