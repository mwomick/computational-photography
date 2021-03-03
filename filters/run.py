#!/usr/bin/env python3.9

import multiprocessing

from skimage import io
import numpy as np
from skimage import img_as_ubyte
import os

from src import grayscale as gra
from src import sepia as sep
from src import balancing as bal
from src import artistic as art 
from src import contrast as con


INPUT_ROOT_DIR = 'data/'
OUTPUT_ROOT_DIR = 'data/out/' 


def gray():
    ims = ['belltower.png', 'paris.jpg', 'venice1.jpg', 'snow1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method1/" + f, img_as_ubyte(gra.built_in_method(im)))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method2/" + f, img_as_ubyte(gra.average_method(im)))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method3/" + f, img_as_ubyte(gra.weighted_method(im)))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method4/" + f, img_as_ubyte(gra.hsv_method(im)))


def sepia():
    ims = ['belltower.png', 'paris.jpg', 'venice1.jpg', 'snow1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise2/" + f, img_as_ubyte(sep.sepia(im)))


def balance():
    ims = ['belltower.png', 'paris.jpg', 'venice2.jpg', 'snow2.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise3/" + f, img_as_ubyte(bal.balance(im)))


def contrast():
    ims = ['nicaragua1.jpg', 'paris.jpg', 'venice3.jpg', 'snow3.jpg']
    factors = [1.3, 1.1, 3.3, 2.0]
    offsets = [-0.2, -0.1, -0.2, -0.5]
    for i in range(0, len(ims)):
        im = io.imread(INPUT_ROOT_DIR + ims[i])
        con.manual_contrast(im, factors[i], offsets[i])
        con.auto_contrast(im)


def neon():
    ims = ['belltower.png', 'paris.jpg', 'venice1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise5/" + "neon_" + f, img_as_ubyte(art.neon(im)))
    

def comic():
    ims = ['belltower.png', 'paris.jpg', 'venice1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise5/" + "comic_" + f, img_as_ubyte(art.comic(im)))


def vignette():
    ims = ['belltower.png', 'paris.jpg', 'venice1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise5/" + "vign_" + f, img_as_ubyte(art.vignette(im)))


def clean():
    for i in range(1, 4):
        os.system('rm data/out/exercise1/method'+str(i)+'/*')
    for i in range(2, 5):
        os.system('rm data/out/exercise'+str(i)+'/*')


# clean()

if __name__ == '__main__':
    functions = [vignette] #, contrast, gray, sepia, balance, neon, comic]
    processes = []

    for f in functions:
        p = multiprocessing.Process(target=f)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()