import numpy as np
from skimage import io, transform
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt

def get_laplacians(im):
    gaussians = get_gaussians(im)
    res = []
    for i in range(0, 7):
        upper = gaussians[i]
        lower = transform.resize(gaussians[i+1], (gaussians[i].shape[0], gaussians[i].shape[1]))
        diff = upper-lower
        res.append(diff)
    return res


def get_gaussians(im):
    res = []
    im_g = im/255.
    res.append(im_g)
    for i in range(0, 7):
        im_g = cv.GaussianBlur(im_g, (5,5), 2, sigmaY=2)
        im_g = transform.resize(im_g, (im_g.shape[0]//2, im_g.shape[1]//2))
        res.append(im_g)
    return res 

INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

src = io.imread(INPUT_ROOT_DIR + "bear_padded.png")[:,:,:3]
mask = io.imread(INPUT_ROOT_DIR + "bear_mask.png")[:,:,:3]
tar = io.imread(INPUT_ROOT_DIR + "swim.jpg")[:,:,:3]

mask_gaussians = get_gaussians(mask)
tar_laplacians = get_laplacians(tar)
src_laplacians = get_laplacians(src)
tar_gaussians = get_gaussians(tar)

pyramid = []
for i in range(0, 7):
    pyramid.append(mask_gaussians[i] * src_laplacians[i] + (1.-mask_gaussians[i]) * tar_laplacians[i])

com = transform.resize(tar_gaussians[7], (tar.shape[0], tar.shape[1]))
for i in reversed(range(0, 7)):
    com += transform.resize(pyramid[i], (tar.shape[0], tar.shape[1]))



plt.imshow(com)
plt.show()


