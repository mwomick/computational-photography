from skimage import color
import numpy as np

def built_in_method(im):
    return color.rgb2gray(color.rgba2rgb(im))

def formula_method(im):
    height, width, channels = im.shape
    imo = np.zeros((height, width, 3), dtype=int)

    for y in range(height):
        for x in range(width):
            gray = im[y][x][0] * 0.299 + im[y][x][1] * 0.587 + im[y][x][2] * 0.114
            imo[y][x] = (gray, gray, gray)

    return imo