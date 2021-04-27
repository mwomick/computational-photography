from os import listdir

from skimage import io, color, img_as_ubyte
from PIL import Image
from matplotlib import pyplot as plt
import time

from src.mosaic import mosaic
import numpy as np

INPUT_ROOT_DIR = 'data/in/val2017/'
OUTPUT_ROOT_DIR = 'data/out/'

template_name = "belltower.png"

dataset_names = listdir(INPUT_ROOT_DIR)

dataset = []
for f in dataset_names[:5500]:
    e = Image.open(INPUT_ROOT_DIR + f)
    dataset.append(e)
template = Image.open(INPUT_ROOT_DIR + template_name)

show = mosaic(template, dataset, 25, 25)
io.imsave(OUTPUT_ROOT_DIR + "mosaic-"+str(int(time.time()))+".jpg", img_as_ubyte(show))