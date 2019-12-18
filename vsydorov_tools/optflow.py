import numpy as np
import sys
import logging
import cv2
log = logging.getLogger(__name__)


def read_flow_file(filename):
    """
    Source:
      - http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    Returns:
      - (X, Y, 2) ndarray with (u, v) flow channels
    """
    filename = str(filename)
    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            nBands = 2
            # print('Reading %d x %d flo file %s' % (w, h, filename))
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            tmp = np.reshape(data, (w*2, h), 'F')
            tmp = np.transpose(tmp)
            image = np.dstack((
                tmp[:, np.arange(1, w+1)*nBands-2],
                tmp[:, np.arange(1, w+1)*nBands-1]))
    return image


def example_circle(side=75):
    """
    OptFlow (u,v) circle like this:
             ^ '-V'
             |
             |
    '-U' <------> '+U'
             |
             |
             V '+V'
    """
    V, U = np.mgrid[-side:side, -side:side]
    zeros = U*U + V*V > side*side
    uv = np.dstack((U, V))
    uv[zeros] = 0
    uv = uv.astype('float32')
    return uv


def opencv_flow_to_bgr(flo):
    """
    Source:
      - https://stackoverflow.com/questions/28898346/visualize-optical-flow-with-color-model
      - https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html

    Returns:
      - BGR image
    """
    flo = flo.astype('float32')
    mag, ang = cv2.cartToPolar(flo[..., 0], flo[..., 1])
    hsv = np.zeros((*flo.shape[0:2], 3))
    hsv[:, :, 0] = ang*180/np.pi/2
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
    return bgr


"""
Middlebury colorwheel code

Source:
- https://github.com/Johswald/flow-code-python/blob/master/readFlowFile.py
"""


def makeColorwheel():
    #  color encoding scheme
    #   adapted from the color circle idea described at
    #   http://members.shaw.ca/quadibloc/other/colint.htm
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])  # r g b
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0, RY, 1)/RY)
    col += RY
    # YG
    colorwheel[col:YG+col, 0]= 255 - np.floor(255*np.arange(0, YG, 1)/YG)
    colorwheel[col:YG+col, 1] = 255
    col += YG
    # GC
    colorwheel[col:GC+col, 1]= 255
    colorwheel[col:GC+col, 2] = np.floor(255*np.arange(0, GC, 1)/GC)
    col += GC
    # CB
    colorwheel[col:CB+col, 1]= 255 - np.floor(255*np.arange(0, CB, 1)/CB)
    colorwheel[col:CB+col, 2] = 255
    col += CB
    # BM
    colorwheel[col:BM+col, 2]= 255
    colorwheel[col:BM+col, 0] = np.floor(255*np.arange(0, BM, 1)/BM)
    col += BM
    # MR
    colorwheel[col:MR+col, 2]= 255 - np.floor(255*np.arange(0, MR, 1)/MR)
    colorwheel[col:MR+col, 0] = 255
    return colorwheel


def computeColor(u, v):
    colorwheel = makeColorwheel()
    nan_u = np.isnan(u)
    nan_v = np.isnan(v)
    nan_u = np.where(nan_u)
    nan_v = np.where(nan_v)
    u[nan_u] = 0
    u[nan_v] = 0
    v[nan_u] = 0
    v[nan_v] = 0
    ncols = colorwheel.shape[0]
    radius = np.sqrt(u**2 + v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) /2 * (ncols-1)  # -1~1 maped to 1~ncols
    k0 = fk.astype(np.uint8)	 # 1, 2, ..., ncols
    k1 = k0+1
    k1[k1 == ncols] = 0
    f = fk - k0
    img = np.empty([k1.shape[0], k1.shape[1], 3])
    ncolors = colorwheel.shape[1]
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx]*(1-col[idx])
        col[~idx] *= 0.75  # out of range
        img[:, :, 2-i] = np.floor(255*col).astype(np.uint8)
    return img.astype(np.uint8)


def middlebury_flow_to_bgr(flow):
    """
    Returns: BGR image
    """
    eps = sys.float_info.epsilon
    UNKNOWN_FLOW_THRESH = 1e9
    # UNKNOWN_FLOW = 1e10
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    maxu = -999
    maxv = -999
    minu = 999
    minv = 999
    maxrad = -1
    # fix unknown flow
    greater_u = np.where(u > UNKNOWN_FLOW_THRESH)
    greater_v = np.where(v > UNKNOWN_FLOW_THRESH)
    u[greater_u] = 0
    u[greater_v] = 0
    v[greater_u] = 0
    v[greater_v] = 0
    maxu = max([maxu, np.amax(u)])
    minu = min([minu, np.amin(u)])
    maxv = max([maxv, np.amax(v)])
    minv = min([minv, np.amin(v)])
    rad = np.sqrt(
            np.multiply(u, u)+np.multiply(v, v))
    maxrad = max([maxrad, np.amax(rad)])
    log.debug((
        'max flow: {:.4f} flow range: '
        'u = {:.3f} .. {:.3f}; v = {:.3f} .. {:.3f}\n').format(
            maxrad, minu, maxu, minv, maxv))
    u = u/(maxrad+eps)
    v = v/(maxrad+eps)
    bgr = computeColor(u, v)
    return bgr
