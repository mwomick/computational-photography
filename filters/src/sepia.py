import numpy as np

def sepia(im):
    height, width, ch = im.shape
    imo = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            r = float(im[y][x][0])
            g = float(im[y][x][1])
            b = float(im[y][x][2])

            sepia_r = min(r*0.393 + g*0.769 + b*0.189, 255)
            sepia_g = min(r*0.349 + g*0.686 + b*0.168, 255)
            sepia_b = min(r*0.272 + g*0.534 + b*0.131, 255)
            
            imo[y][x] = ( np.uint8(sepia_r), np.uint8(sepia_g) , np.uint8(sepia_b) )

    return imo