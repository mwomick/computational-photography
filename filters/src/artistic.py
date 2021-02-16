import numpy as np
from skimage import color
from skimage import filters
from math import sqrt

filtr = [[0.025, 0.0625, 0.5, 0.0625, 0.025],[0.0125, 0.025, 0.25, 0.025, 0.0125]]

def bw_grad(im):
    height, width, ch = im.shape
    im = filters.gaussian(im, sigma=0.4)
    imo = np.zeros((height, width, 3), dtype=np.uint8)
    gra = np.zeros((height, width, 1), dtype=np.int64)
    gra = color.rgb2gray(im)

    thresh = 0.025

    for y in range(2, height-2):
        for x in range(2, width-2):
            gradX = np.dot(filtr[0], gra[y, x-2:x+3]) + np.dot(filtr[1], gra[y+1, x-2:x+3])-gra[y,x]
            gradY = np.sum(np.transpose(filtr)*gra[y-2:y+3, x:x+2])-gra[y, x]
            gradXY = sqrt(gradX*gradX + gradY*gradY) # is this better than abs(x + y)?
            if(gradXY > thresh):
                hsv = color.rgb2hsv(im[y,x])
                hsv[1] = 1
                hsv[2] = min(hsv[2]*1.5, 1)
                imo[y,x] = 255*color.hsv2rgb(hsv)
            # imo[y][x] = ( np.uint8(sepia_r), np.uint8(sepia_g) , np.uint8(sepia_b) )
            
    return imo
