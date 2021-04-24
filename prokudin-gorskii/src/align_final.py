import cv2
from scipy import misc
from matplotlib import pyplot as plt
from matplotlib import patches
import numpy as np
import statistics
from sklearn import preprocessing
# from skimage.feature import match_template

from PIL import Image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def align_final(im_rgb):
    shape_y = min(im_rgb[0].shape[0], im_rgb[1].shape[0], im_rgb[2].shape[0])
    shape_x = min(im_rgb[0].shape[1], im_rgb[1].shape[1], im_rgb[2].shape[1])
    
    r = im_rgb[0][:shape_y,:shape_x]
    g = im_rgb[1][:shape_y,:shape_x]
    b = im_rgb[2][:shape_y,:shape_x]

    dst = np.zeros((shape_y, shape_x, 3), dtype=int)
    dst[:shape_y,:shape_x,0] = r

    res = cv2.matchTemplate(r, b[20:int(shape_y-20), 20:int(shape_x-20)], cv2.TM_CCORR_NORMED)
    y, x = np.unravel_index(np.argmax(res), res.shape)
    
    offset_x = x-20
    offset_y = y-20

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


    res = cv2.matchTemplate(r, g[20:int(shape_y-20), 20:int(shape_x-20)], cv2.TM_CCORR_NORMED)
    y, x = np.unravel_index(np.argmax(res), res.shape)
    
    offset_x = x-20
    offset_y = y-20

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

    #####################################
    #             CROPPING
    #####################################

    dst_comb = dst[:,:,0]*dst[:,:,1]*dst[:,:,2]

    # Calculate the standard deviation for each column and row
    row_data = np.zeros((shape_y), dtype=float)
    col_data = np.zeros((shape_x), dtype=float)

    for i in range(0, shape_y):
        row_data[i] = statistics.stdev(dst_comb[i,:])

    for i in range(0, shape_x):
        col_data[i] = statistics.stdev(dst_comb[:,i])
    
    # Calculate the squared difference in stddev between each n and n+1th row
    row_dev_diff = row_data[1:]-row_data[:-1]
    row_dev_diff *= row_dev_diff
    col_dev_diff = col_data[1:]-col_data[:-1]
    col_dev_diff *= col_dev_diff

    # Calculate the (squared) change in difference of stddev between rows
    # This will yield values representative of how rapidly the amount of information between rows changes
    row_dev_diff_diff = row_dev_diff[1:] - row_dev_diff[:-1]
    col_dev_diff_diff = col_dev_diff[1:] - col_dev_diff[:-1]

    top_i = 0
    top_f = 0
    left_i = 0
    left_f = 0
    right_i = 0
    right_f = shape_x
    bottom_i = 0
    bottom_f = shape_y

    top_max = max(row_dev_diff_diff[:int(shape_y/9)])
    bottom_max = max(row_dev_diff_diff[int(8*shape_y/9):shape_y-4])
    top_max_index = np.where(row_dev_diff_diff == top_max)[-1]+2
    bottom_max_index = np.where(row_dev_diff_diff == bottom_max)[-1]+2

    left_max = max(col_dev_diff_diff[:int(shape_x/9)])
    right_max = max(col_dev_diff_diff[int(8*shape_x/9):shape_x-4])
    left_max_index = np.where(col_dev_diff_diff == left_max)[-1]+2
    right_max_index = np.where(col_dev_diff_diff == right_max)[-1]+2

    top_f = top_max_index
    bottom_i = bottom_max_index
    left_f = left_max_index
    right_i = right_max_index


    #print(bottom_max_index)
    #print(top_max_index)
    #print(left_max_index)
    #print(right_max_index)

    # Create figure and axes
    #fig, ax = plt.subplots()

    # Display the image
    #ax.imshow(dst)
    #width = int(right_i-left_f)
    #height = int(bottom_i-top_f)
    # Create a Rectangle patch
    #rect = patches.Rectangle((left_f, top_f), width, height, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    #ax.add_patch(rect)

    #plt.show()
    res = dst[top_f[0]:bottom_i[0], left_f[0]:right_i[0],:]
    return res

