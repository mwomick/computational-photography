import numpy as np
from skimage import color
from skimage import filters
from math import sqrt

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
            gradXY = sqrt(gradX*gradX + gradY*gradY) # is this so much better than abs(x + y)?
            #if(gradXY > thresh):

            # maybe we want to handle gray/white/bright colors differently?
            hsv = color.rgb2hsv(im[y,x])

            
            if(hsv[1] < 0.6 and hsv[2] > 0.5):
                hsv[1] *= 1.66666
            else:
                hsv[1] = 1

            hsv[2] = gradXY*12

            imo[y,x] = 255*color.hsv2rgb(hsv)
            # imo[y][x] = ( np.uint8(sepia_r), np.uint8(sepia_g) , np.uint8(sepia_b) )

    # wanna smear hue, but do we wanna smear lines?
    # imo = filters.gaussian(imo, sigma=0.1)
    hsv = color.rgb2hsv(imo)
    ghsv = filters.gaussian(hsv, sigma=0.4)
    hsv[:,:,0] = ghsv[:,:,0]
    imo = color.hsv2rgb(hsv)

    return imo
