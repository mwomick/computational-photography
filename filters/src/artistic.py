import numpy as np
from skimage import color
from skimage import filters
from math import sqrt
import cv2

filtr = [[0.025, 0.0625, 0.5, 0.0625, 0.025],[0.0125, 0.025, 0.25, 0.025, 0.0125]]

def neon(im):
    height, width, ch = im.shape
    im = filters.gaussian(im, sigma=0.7)
    imo = np.zeros((height, width, 3), dtype=np.uint8)
    gra = np.zeros((height, width, 1), dtype=np.int64)
    gra = color.rgb2gray(im)

    # thresh = 0.001

    for y in range(2, height-2):
        for x in range(2, width-2):
            gradX = np.dot(filtr[0], gra[y, x-2:x+3]) + np.dot(filtr[1], gra[y+1, x-2:x+3])-gra[y,x]
            gradY = np.sum(np.transpose(filtr)*gra[y-2:y+3, x:x+2])-gra[y, x]
            gradXY = sqrt(gradX*gradX + gradY*gradY)

            hsv = color.rgb2hsv(im[y,x])
            hsv[2] = gradXY*12          # perhaps the gradient could be normalized to the max/min/avg/std dev of gradient?
            hsv[1] = 1
            # if we have some combination of high value and low saturation,...
            lightness = sqrt(hsv[2]*hsv[2] + (1-hsv[1])*(1-hsv[1]))
            if(lightness > 0.5):
                hsv[1] = lightness
                hsv[2] = hsv[2]*(2-lightness)

            imo[y,x] = 255*color.hsv2rgb(hsv)
            # imo[y][x] = ( np.uint8(sepia_r), np.uint8(sepia_g) , np.uint8(sepia_b) )

    # wanna smear hue, but do we wanna smear lines?
    # imo = filters.gaussian(imo, sigma=0.1)
    hsv = color.rgb2hsv(imo)
    ghsv = filters.gaussian(hsv, sigma=0.4)
    hsv[:,:,0] = ghsv[:,:,0]
    imo = color.hsv2rgb(hsv)

    return imo


def comic(im):
    height, width, ch = im.shape
    im = filters.gaussian(im, sigma=0.7)
    imblur = filters.gaussian(im, sigma=1.5)

    imo = np.zeros((height, width, 3), dtype=np.uint8)
    gra = np.zeros((height, width, 1), dtype=np.int64)
    gra = color.rgb2gray(im)
    grad = np.zeros((height, width, 1), dtype=np.float)

    # make gradient matrix
    for y in range(2, height-2):
        for x in range(2, width-2):
            gradX = np.dot(filtr[0], gra[y, x-2:x+3]) + np.dot(filtr[1], gra[y+1, x-2:x+3])-gra[y,x]
            gradY = np.sum(np.transpose(filtr)*gra[y-2:y+3, x:x+2])-gra[y, x]
            gradXY = sqrt(gradX*gradX + gradY*gradY) 
            grad[y][x] = gradXY

    mean = np.mean(grad)
    stddev = sqrt(np.sum(np.multiply(gradXY - mean, gradXY - mean))/(height*width))
    thresh = mean+128*stddev

    for y in range(2, height-2):
        for x in range(2, width-2):
            hsv = color.rgb2hsv(imblur[y,x])
            if(grad[y,x] > thresh):
                factor = min(1,(grad[y,x]-mean)/(mean))
                val = hsv[2]
                hsv[2] = abs(val-.5)*(1-factor)
                hsv[1] = min(1, val+.1)*(1-factor)
            else:
                hsv[1] = abs(hsv[2]-.5)
                hsv[2] = min(1, hsv[2]+.1)

            imo[y,x] = 255*color.hsv2rgb(hsv)
            # imo[y][x] = ( np.uint8(sepia_r), np.uint8(sepia_g) , np.uint8(sepia_b) )

    return imo

def vignette(im):
    rows, cols = im.shape[:2]

    x_gauss = cv2.getGaussianKernel(cols, cols/2) 
    y_gauss = cv2.getGaussianKernel(rows, rows/2)
    kernel = y_gauss * x_gauss.T
    
    hsv = color.rgb2hsv(im)
    
    mask = 255 * kernel / np.linalg.norm(kernel)
    hsv[:,:,2] *= mask
    im = color.hsv2rgb(hsv)

    return im