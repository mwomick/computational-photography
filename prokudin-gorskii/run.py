#!/usr/bin/env python3.9

import multiprocessing

from skimage import io
import numpy as np
from skimage import img_as_ubyte
import os

from src.autocrop import crop
from src.align_basic import align_basic as basic 

INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

def preprocess(im):
    pass

def align_basic():
    ims = ['camel.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "part1/" + f, img_as_ubyte(basic(im)))


def clean():
    for i in range(1, 3):
        os.system('rm data/out/part'+str(i)+'/*')

# clean()

if __name__ == '__main__':
    functions = [align_basic] # vignette, contrast, gray, sepia, balance, neon, 
    processes = []

    for f in functions:
        p = multiprocessing.Process(target=f)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()