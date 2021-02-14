from skimage import color
import numpy as np


def built_in_method(im):
    return color.rgb2gray(im)


def average_method(im):
    height, width, ch = im.shape
    imo = np.zeros((height, width, 3), dtype=int)

    for y in range(height):
        for x in range(width):
            gray = int((float(im[y][x][0]) + float(im[y][x][1]) + float(im[y][x][2]))/3.0)
            imo[y][x] = (gray, gray, gray)

    return imo


def weighted_method(im):
    height, width, ch = im.shape
    imo = np.zeros((height, width, 3), dtype=int)

    for y in range(height):
        for x in range(width):
            gray = int(im[y][x][0] * 0.299 + im[y][x][1] * 0.587 + im[y][x][2] * 0.114)
            imo[y][x] = (gray, gray, gray)

    return imo


def hsv_method(im):
    im = color.rgb2hsv(im)
    height, width, ch = im.shape
    imo = im
    for y in range(height):
        for x in range(width):
            imo[y][x][1] = 0
            
    return color.hsv2rgb(imo)