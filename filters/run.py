from skimage import io
import numpy as np

from src import grayscale as gra

INPUT_ROOT_DIR = 'data/'
OUTPUT_ROOT_DIR = 'data/out/' 


def gray():
    ims = ['belltower.png', 'venice1.jpg']

    for f in ims:
        im = io.imread(INPUT_ROOT_DIR + f)
        imo = gra.built_in_method(im)
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method1/" + f, gra.built_in_method(im))
        io.imsave(OUTPUT_ROOT_DIR + "exercise1/method2/" + f, gra.formula_method(im))


gray()