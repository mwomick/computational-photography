from os import listdir
import time

from skimage import io, color, img_as_ubyte
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from progress.bar import IncrementalBar

from src.mosaic import mosaic
from config import DATASET_ROOT_DIR, OUTPUT_ROOT_DIR, TEMPLATE_PATH, DATASET_MAX, TILE_WIDTH, TILE_HEIGHT


def run(template_path, set_dir, tile_width, tile_height):
    template = Image.open(template_path)
    files = listdir(set_dir)[:DATASET_MAX]
    resized = []
    resize_bar = IncrementalBar('Resizing collection', max=len(files))
    for f in files:
        im = Image.open(set_dir + f)
        try:
            e = np.array(im.resize((tile_width, tile_height)))
            if e.shape[2] == 3:
                resized.append(e)
        except Exception as e:
            pass
        finally:
            im.close()
            resize_bar.next()
    resize_bar.finish()

    show = mosaic(template, resized, tile_width, tile_height)
    io.imsave(OUTPUT_ROOT_DIR + "mosaic-"+str(int(time.time()))+".jpg", img_as_ubyte(show))


run(TEMPLATE_PATH, DATASET_ROOT_DIR, TILE_WIDTH, TILE_HEIGHT)