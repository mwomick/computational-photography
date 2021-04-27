#!/usr/bin/env python3.9

import multiprocessing

from skimage import io
import numpy as np
from skimage import img_as_ubyte
import os



INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

def align_basic():
    ims = ['monastery.jpg', 'cathedral.jpg', 'camel.jpg', 'chapel.jpg', 'courtyard.jpg', 'emir.jpg', 'gruppa.jpg', 'khan.jpg', 'nativity.jpg', 'railroad.jpg', 'settlers.jpg', 'urn.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "part1/" + f, img_as_ubyte(im)


def clean():
    for i in range(1, 3):
        os.system('rm data/out/part'+str(i)+'/*')

# clean()

if __name__ == '__main__':
    functions = [] # vignette, contrast, gray, sepia, balance, neon, 
    processes = []

    for f in functions:
        p = multiprocessing.Process(target=f)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()