from skimage import io
import matplotlib.pyplot as plt

def cut_and_paste(im_dst, im_src, im_mask):
    for y in range(im_mask.shape[0]):
        for x in range(im_mask.shape[1]):
            if im_mask[y, x].all() == 1:
                im_dst[y, x] = im_src[y, x]
    plt.imshow(im_dst)
    plt.show()


INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

dst = io.imread(INPUT_ROOT_DIR + "swim.jpg")[:,:,:3]
src = io.imread(INPUT_ROOT_DIR + "bear_padded.png")[:,:,:3]
mask = io.imread(INPUT_ROOT_DIR + "bear_mask.png")[:,:,:3]

cut_and_paste(dst, src, mask)