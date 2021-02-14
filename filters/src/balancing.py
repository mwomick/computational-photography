import numpy as np


def balance(im):
    overall_avg = np.mean(im)
    height, width, ch = im.shape
    imo = np.zeros((height, width, 3), dtype=np.uint8)

    rgb = np.zeros((3), dtype=np.int32)

    for y in range(height):
        for x in range(width):
            rgb[0] += im[y][x][0]
            rgb[1] += im[y][x][1]
            rgb[2] += im[y][x][2]

    rgb = rgb/(width*height)
    multiplier = overall_avg * 1/rgb

    for y in range(height):
        for x in range(width):
            tmp = multiplier * im[y][x]
            imo[y][x][0] = np.uint8(min(tmp[0], 255))
            imo[y][x][1] = np.uint8(min(tmp[1], 255))
            imo[y][x][2] = np.uint8(min(tmp[2], 255))

    return imo