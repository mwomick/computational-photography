import matplotlib.pyplot as plt

import cv2

from skimage import color
from skimage.util import img_as_float

def manual_contrast(im, factor, offset):
    im = img_as_float(im)
    plt.subplot(2, 2, 1)
    _ = plt.imshow(im)
    plt.subplot(2, 2, 2)
    _ = plt.hist(im[:, :, 0].ravel(), bins = 256, color = 'Red', alpha = 0.33)
    _ = plt.hist(im[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.33)
    _ = plt.hist(im[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.33)
    _ = plt.xlim(0, 1.0) 
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Red', 'Green', 'Blue'])

    hsv = color.rgb2hsv(im)
    hsv[:, :, 2] += offset
    hsv[:, :, 2] *= factor
    im = color.hsv2rgb(hsv)

    plt.subplot(2, 2, 3)
    _ = plt.imshow(im)
    plt.subplot(2, 2, 4)
    _ = plt.hist(im[:, :, 0].ravel(), bins = 256, color = 'Red', alpha = 0.33)
    _ = plt.hist(im[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.33)
    _ = plt.hist(im[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.33)
    _ = plt.xlim(0, 1.0)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Red', 'Green', 'Blue'])
    plt.show()

    return im


def auto_contrast(im):
    plt.subplot(2, 2, 1)
    _ = plt.imshow(im)
    plt.subplot(2, 2, 2)
    _ = plt.hist(im[:, :, 0].ravel(), bins = 256, color = 'Red', alpha = 0.33)
    _ = plt.hist(im[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.33)
    _ = plt.hist(im[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.33)
    _ = plt.xlim(0, 255)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Red', 'Green', 'Blue'])

    H, S, V = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
    eq_V = cv2.equalizeHist(V)
    im = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)

    plt.subplot(2, 2, 3)
    _ = plt.imshow(im)
    plt.subplot(2, 2, 4)
    _ = plt.hist(im[:, :, 0].ravel(), bins = 256, color = 'Red', alpha = 0.33)
    _ = plt.hist(im[:, :, 1].ravel(), bins = 256, color = 'Green', alpha = 0.33)
    _ = plt.hist(im[:, :, 2].ravel(), bins = 256, color = 'Blue', alpha = 0.33)
    _ = plt.xlim(0, 255)
    _ = plt.xlabel('Intensity Value')
    _ = plt.ylabel('Count')
    _ = plt.legend(['Red', 'Green', 'Blue'])
    plt.show()

    return im