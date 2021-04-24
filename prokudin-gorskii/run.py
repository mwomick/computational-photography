#!/usr/bin/env python3.9

import multiprocessing

from skimage import io
import numpy as np
from skimage import img_as_ubyte
import os

from src.autocrop import crop
from src.align_basic import align_basic as basic
from src.align_better import align_better as better
from src.align_final import align_final as final

INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

def preprocess(im):
    height, width  = im.shape
    b = im
    g = im[int(height/4):height, 0:width]
    r = im[int(height/2):height, 0:width]
    b = crop(b)
    g = crop(g)
    r = crop(r)
    return((r, g, b))


def align_basic():
    ims = ['monastery.jpg', 'cathedral.jpg', 'camel.jpg', 'chapel.jpg', 'courtyard.jpg', 'emir.jpg', 'gruppa.jpg', 'khan.jpg', 'nativity.jpg', 'railroad.jpg', 'settlers.jpg', 'urn.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        r, g, b = preprocess(im)
        io.imsave(OUTPUT_ROOT_DIR + "part1/" + f, img_as_ubyte(basic((r,g,b))))


def align_better():
    ims = ['monastery.jpg', 'cathedral.jpg', 'camel.jpg', 'chapel.jpg', 'courtyard.jpg', 'emir.jpg', 'gruppa.jpg', 'khan.jpg', 'nativity.jpg', 'railroad.jpg', 'settlers.jpg', 'urn.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        r, g, b = preprocess(im)
        io.imsave(OUTPUT_ROOT_DIR + "part2/" + f, img_as_ubyte(better((r,g,b))))


def align_final():
    ims = ['monastery.jpg', 'cathedral.jpg', 'camel.jpg', 'chapel.jpg', 'courtyard.jpg', 'emir.jpg', 'gruppa.jpg', 'khan.jpg', 'nativity.jpg', 'railroad.jpg', 'settlers.jpg', 'urn.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        r, g, b = preprocess(im)
        # final((r, g, b))
        io.imsave(OUTPUT_ROOT_DIR + "part3/" + f, img_as_ubyte(final((r,g,b))))


def clean():
    for i in range(1, 3):
        os.system('rm data/out/part'+str(i)+'/*')

# clean()

if __name__ == '__main__':
    functions = [align_final] # vignette, contrast, gray, sepia, balance, neon, 
    processes = []

    for f in functions:
        p = multiprocessing.Process(target=f)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()