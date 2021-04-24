from PIL import Image
from cv2 import Sobel
from cv2 import CV_64F
import cv2
import numpy

# == DEBUGGING =========
# from matplotlib import pyplot as plt
# ======================

def crop(im):
    
    height, width  = im.shape

    sobelx = Sobel(im, CV_64F, 1, 0, ksize=5)
    sobely = Sobel(im, CV_64F, 0, 1, ksize=5)

    size = 299
    kernel_blur_x = numpy.zeros((size, size))
    kernel_blur_x[int((size-1)/2), :] = numpy.ones(size)
    kernel_blur_x = kernel_blur_x / size

    kernel_blur_y = numpy.zeros((size, size))
    kernel_blur_y[:, int((size-1)/2)] = numpy.ones(size)
    kernel_blur_y = kernel_blur_y / size
  
    sobely = cv2.filter2D(sobely, -1, kernel_blur_x)
    sobelx = cv2.filter2D(sobelx, -1, kernel_blur_y)

    sobely = numpy.multiply(sobely, sobely)
    sobelx = numpy.multiply(sobelx, sobelx)

    left = 0
    for i in range(0, 200):
        if(numpy.mean(sobelx[:, i]) > 1e7):
            left = i
            break

    right = 0
    for i in range(width-1, 200, -1):
        if(numpy.mean(sobelx[:, i]) > 1e7):
            right = i
            break 

    top = 0
    for i in range(0, height):
        if(numpy.mean(sobely[i, :]) > 1e7):
            top = i
            break

    bottom = height
    for i in range(top+250, height):
        if(numpy.mean(sobely[i, :]) > 1e7):
            bottom = i
            break

    #plt.imshow(im[top:bottom, left:right])
    #plt.show()

    return im[top:bottom, left:right]