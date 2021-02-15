import numpy as np
from skimage import color

filtr = [[0.025, 0.0625, 0.5, 0.0625, 0.025],[0.0125, 0.025, 0.25, 0.025, 0.0125]]

    # [[0.025, 0.0125],[0.0625, 0.025],[0.5, 0.025],[0.0625, 0.025],[0.025, 0.0125]]

def grad(im):
    im = color.rgb2gray(im)
    height, width = im.shape
    imo = np.zeros((height, width, 1), dtype=np.uint8)

    for y in range(2, height-2):
        for x in range(2, width-2):
            imo[y][x]=abs(np.dot(filtr[0], im[y, x-2:x+3]) + np.dot(filtr[1], im[y+1, x-2:x+3])-im[y,x])*255 + abs(np.sum(np.transpose(filtr)*im[y-2:y+3, x:x+2])-im[y, x])*255
            # imo[y][x] = ( np.uint8(sepia_r), np.uint8(sepia_g) , np.uint8(sepia_b) )
            
    return imo
