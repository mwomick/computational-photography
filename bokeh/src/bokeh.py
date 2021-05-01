import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from scipy import signal
import cv2 as cv

def disk_filter(r):
    y,x = np.ogrid[-r: r+1, -r: r+1]
    disk = x**2+y**2 <= r**2
    return(disk.astype(float)/(3.14*r*r))

def bokeh(image, mask):
    dst = np.zeros((image.shape[0], image.shape[1], 3), dtype=float)
    fore = np.array(image*((255-mask)/(255*255)), dtype=float)
    
    back = np.array(image*(mask/(255)), dtype=float)

    disk = disk_filter(50)
    back_blurred = cv.filter2D(back, 0, disk)/255.
    mask_blurred = cv.filter2D(mask, 0, disk)/255.

    e = np.nan_to_num(np.divide(back_blurred, mask_blurred))
    
    for y in range(fore.shape[0]):
        for x in range(fore.shape[1]):
            if mask[y,x].all() == 0:
                e[y,x] = fore[y,x]

    plt.imshow(e)

    plt.show()

INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

image = io.imread(INPUT_ROOT_DIR + "building.png")[:,:,:3]
mask = io.imread(INPUT_ROOT_DIR + "building_mask.png")[:,:,:3]

bokeh(image, mask)
