import cv2
from scipy import misc
from matplotlib import pyplot as plt
import numpy as np
# from skimage.feature import match_template

from PIL import Image

def align_better(im_rgb):
    shape_y = min(im_rgb[0].shape[0], im_rgb[1].shape[0], im_rgb[2].shape[0])
    shape_x = min(im_rgb[0].shape[1], im_rgb[1].shape[1], im_rgb[2].shape[1])
    
    r = im_rgb[0][:shape_y,:shape_x]
    g = im_rgb[1][:shape_y,:shape_x]
    b = im_rgb[2][:shape_y,:shape_x]

    dst = np.zeros((shape_y, shape_x, 3), dtype=int)
    dst[:shape_y,:shape_x,0] = r

    res = cv2.matchTemplate(r, b[int(shape_y/5):int(4*shape_y/5), int(shape_x/5):int(4*shape_x/5)], cv2.TM_CCORR_NORMED)
    y, x = np.unravel_index(np.argmax(res), res.shape)
    
    offset_x = x-int(shape_x/5)
    offset_y = y-int(shape_y/5)

    dst_xi = offset_x
    dst_xf = shape_x - offset_x
    src_xi = 0
    dst_yi = offset_y
    dst_yf = shape_y - offset_y
    src_yi = 0

    if offset_x < 0:
        dst_xi = 0
        dst_xf = shape_x + offset_x
        src_xi = abs(offset_x)

    if offset_y < 0:
        dst_yi = 0
        dst_yf = shape_y + offset_y
        src_yi = abs(offset_y)

    src_xf = src_xi + (dst_xf - dst_xi)
    src_yf = src_yi + (dst_yf - dst_yi)   

    dst[dst_yi:dst_yf,dst_xi:dst_xf,2] = b[src_yi:src_yf, src_xi:src_xf]


    res = cv2.matchTemplate(r, g[int(shape_y/5):int(4*shape_y/5), int(shape_x/5):int(4*shape_x/5)], cv2.TM_CCORR_NORMED)
    y, x = np.unravel_index(np.argmax(res), res.shape)
    
    offset_x = x-int(shape_x/5)
    offset_y = y-int(shape_y/5)

    dst_xi = offset_x
    dst_xf = shape_x - offset_x
    src_xi = 0
    dst_yi = offset_y
    dst_yf = shape_y - offset_y
    src_yi = 0

    if offset_x < 0:
        dst_xi = 0
        dst_xf = shape_x + offset_x
        src_xi = abs(offset_x)

    if offset_y < 0:
        dst_yi = 0
        dst_yf = shape_y + offset_y
        src_yi = abs(offset_y)

    src_xf = src_xi + (dst_xf - dst_xi)
    src_yf = src_yi + (dst_yf - dst_yi)   

    dst[dst_yi:dst_yf,dst_xi:dst_xf,1] = g[src_yi:src_yf, src_xi:src_xf]

    # plt.imshow(dst)
    # plt.show()

    return dst