import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from scipy import signal
import cv2 as cv

def bokeh(image, mask):
    dst = np.zeros((image.shape[0], image.shape[1], 3), dtype=int)
    fore = np.array(image*((255-mask)/255), dtype=int)
    back = np.array(image*(mask/255), dtype=int)
    
    r = 10
    y,x = np.ogrid[-r: r+1, -r: r+1]
    disk = x**2+y**2 <= r**2
    disk = disk.astype(float)/(3.14*r*r)
    


    blurred = cv.filter2D(back, 0, disk)

    plt.imshow(blurred+fore)
    plt.show()

INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 


image = io.imread(INPUT_ROOT_DIR + "couple.png")[:,:,:3]
mask = io.imread(INPUT_ROOT_DIR + "couple_mask.png")[:,:,:3]

bokeh(image, mask)
