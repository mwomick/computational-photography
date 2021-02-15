import multiprocessing

from skimage import io
import numpy as np
from skimage import img_as_ubyte

from src import grayscale as gra
from src import sepia as sep
from src import balancing as bal
from src import artistic as art 

INPUT_ROOT_DIR = 'data/'
OUTPUT_ROOT_DIR = 'data/out/' 


def gray():
    ims = ['belltower.png', 'venice1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method1/" + f, img_as_ubyte(gra.built_in_method(im)))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method2/" + f, img_as_ubyte(gra.average_method(im)))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method3/" + f, img_as_ubyte(gra.weighted_method(im)))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method4/" + f, img_as_ubyte(gra.hsv_method(im)))


def sepia():
    ims = ['belltower.png', 'venice1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise2/" + f, img_as_ubyte(sep.sepia(im)))


def balance():
    ims = ['belltower.png', 'venice1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise3/" + f, img_as_ubyte(bal.balance(im)))


def artistic():
    ims = ['belltower.png', 'venice1.jpg', 'swirl.png']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        io.imsave(OUTPUT_ROOT_DIR + "exercise4/" + f, img_as_ubyte(art.grad(im)))


if __name__ == '__main__':
    functions = [gray, sepia, balance, artistic]
    processes = []

    for f in functions:
        p = multiprocessing.Process(target=f)
        processes.append(p)
        p.start()

    for p in processes:
        p.join()