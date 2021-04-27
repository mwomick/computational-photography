import threading

from PIL import Image
import numpy as np
import cv2        
import scipy.stats  as st
from progress.bar import IncrementalBar

MAX_THREADS = 150

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def imadjust(im):
    H, S, V = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
    eq_V = cv2.equalizeHist(V)
    im = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
    return im


def gkern(kern_width=60, kern_height=60, nsig=3):
    """Returns a 2D Gaussian kernel."""
    y = np.linspace(-nsig, nsig, kern_height+1)
    x = np.linspace(-nsig, nsig, kern_width+1)
    kern1a = np.diff(st.norm.cdf(y))
    kern1b = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1a, kern1b)
    return (kern2d/kern2d.sum())


def optimal_image(template, dataset):
    gfilter = gkern(kern_width=template.shape[1], kern_height=template.shape[0])
    template_r = np.array(template[:,:,0]) * gfilter
    template_g = np.array(template[:,:,1]) * gfilter
    template_b = np.array(template[:,:,2]) * gfilter

    tot_diffs = []
    for i in range(0, len(dataset)):
        e = dataset[i]
        r = e[:,:,0] * gfilter - template_r
        r *= r
        g = e[:,:,1] * gfilter - template_g
        g *= g
        b = e[:,:,2] * gfilter - template_b
        b *= b
        color_diff = r.mean() + g.mean()+ b.mean()
        
        tot_diffs.append(color_diff)

    min_diff = min(tot_diffs)
    min_indx = tot_diffs.index(min_diff)
    return dataset[min_indx]


def place_tile(dst, xi, yi, width, height, im, dataset):
    try:
        _im = np.array(optimal_image(im[yi:yi+height, xi:xi+width], dataset))
        dst[yi*height+int(height/2):yi*height+height+int(height/2), xi*width+int(width/2):xi*width+width+int(width/2)] = _im[:height,:width]
    except Exception as e:
        dst[yi:yi+height, xi:xi+width, :] = 0
        print(e)


def mosaic(img, dataset, tile_width, tile_height):
    image = np.array(img)

    output = np.zeros((tile_height*image.shape[0], tile_width*image.shape[1], 3), dtype=int)
    resized = []
    resize_bar = IncrementalBar('Resizing collection', max=len(dataset))
    for im in dataset:
        resize_bar.next()
        e = np.array(im.resize((tile_width, tile_height)))
        try:
            if e.shape[2] == 3:
                resized.append(e)
        except Exception:
            pass
    resize_bar.finish()


    queue = []
    bar = IncrementalBar('Mosaicking', max=image.shape[0]-tile_height)
    for i in range(0, image.shape[0]-tile_height):
        bar.next()
        for j in range(0, image.shape[1]-tile_width):
            thread = threading.Thread(target=place_tile, args=(output, j, i, tile_width, tile_height, image, resized,))
            thread.start()
            queue.append(thread)
            if len(queue) > MAX_THREADS:
                for thread in queue:
                    thread.join()
                queue.clear()
        for thread in queue:
            thread.join()
        queue.clear()
    bar.finish()
    return output