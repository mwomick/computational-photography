from skimage import io
import numpy as np
from skimage import img_as_ubyte

from src import grayscale as gra

INPUT_ROOT_DIR = 'data/'
OUTPUT_ROOT_DIR = 'data/out/' 


def gray():
    ims = ['belltower.jpg', 'venice1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        imo = gra.built_in_method(im)
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method1/" + f, img_as_ubyte(gra.built_in_method(im)))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method2/" + f, img_as_ubyte(gra.average_method(im)))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method3/" + f, img_as_ubyte(gra.weighted_method(im)))


gray()