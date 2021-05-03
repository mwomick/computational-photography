from skimage import io
import matplotlib.pyplot as plt
import cv2 as cv

def cut_and_paste(im_dst, im_src, im_mask):
    mask_blurred = cv.GaussianBlur(im_mask,(21,21),5, sigmaY=5)

    for y in range(im_dst.shape[0]):
        for x in range(im_dst.shape[1]):
            im_dst[y, x] = mask_blurred[y,x]*(im_src[y, x]/255) + (255-mask_blurred[y,x])*(im_dst[y,x]/255)
    plt.imshow(im_dst)
    plt.show()


INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

dst = io.imread(INPUT_ROOT_DIR + "swim.jpg")[:,:,:3]
src = io.imread(INPUT_ROOT_DIR + "bear_padded.png")[:,:,:3]
mask = io.imread(INPUT_ROOT_DIR + "bear_mask.png")[:,:,:3]

cut_and_paste(dst, src, mask)