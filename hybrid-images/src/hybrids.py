import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import convolve2d
import scipy.stats as st
import utils

def norm(mat):
    minn = np.min(mat)
    maxx = np.max(mat)
    res = (mat-minn)/(maxx-minn)
    return res


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

lo = cv.imread(INPUT_ROOT_DIR + "derek.jpg")[:,:,::-1]
hi = cv.imread(INPUT_ROOT_DIR + "nutmeg.jpg")[:,:,::-1]

# plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(a)))))
# plt.show()
pts_1 = utils.prompt_eye_selection(lo)
pts_2 = utils.prompt_eye_selection(hi)

lo, hi = utils.align_images(lo, hi, pts_1, pts_2)

hi_fft_r = np.fft.fftshift(np.fft.fft2(hi[:,:,0]))
hi_fft_g = np.fft.fftshift(np.fft.fft2(hi[:,:,1]))
hi_fft_b = np.fft.fftshift(np.fft.fft2(hi[:,:,2]))

hi_kern = matlab_style_gauss2D(shape=(hi.shape[0], hi.shape[1]), sigma=6)
hi_kern = hi_kern/np.max(hi_kern)
hi_kern = 1 - hi_kern

plt.imshow(np.log(np.abs(hi_fft_r)))
plt.show()

hi_conv_r = np.multiply(hi_fft_r, hi_kern)
hi_conv_g = np.multiply(hi_fft_g, hi_kern)
hi_conv_b = np.multiply(hi_fft_b, hi_kern)

hi_finv_r = np.abs(np.fft.ifft2(np.fft.ifftshift(hi_conv_r)))
hi_finv_g = np.abs(np.fft.ifft2(np.fft.ifftshift(hi_conv_g)))
hi_finv_b = np.abs(np.fft.ifft2(np.fft.ifftshift(hi_conv_b)))

# hi_norm = norm(hi_outi)
hi_result = np.zeros((hi.shape[0], hi.shape[1], 3))
hi_result[:,:,0] = hi_finv_r
hi_result[:,:,1] = hi_finv_g
hi_result[:,:,2] = hi_finv_b

plt.imshow(norm(hi_result))
plt.show()
# =====================================================================
lo_fft_r = np.fft.fftshift(np.fft.fft2(lo[:,:,0]))
lo_fft_g = np.fft.fftshift(np.fft.fft2(lo[:,:,1]))
lo_fft_b = np.fft.fftshift(np.fft.fft2(lo[:,:,2]))

lo_kern = matlab_style_gauss2D(shape=(lo.shape[0], lo.shape[1]), sigma=25)
lo_kern = lo_kern/np.max(lo_kern)

plt.imshow(np.log(np.abs(lo_fft_r)))
plt.show()

lo_conv_r = np.multiply(lo_fft_r, lo_kern)
lo_conv_g = np.multiply(lo_fft_g, lo_kern)
lo_conv_b = np.multiply(lo_fft_b, lo_kern)

lo_finv_r = np.abs(np.fft.ifft2(np.fft.ifftshift(lo_conv_r+hi_conv_r)))
lo_finv_g = np.abs(np.fft.ifft2(np.fft.ifftshift(lo_conv_g+hi_conv_g)))
lo_finv_b = np.abs(np.fft.ifft2(np.fft.ifftshift(lo_conv_b+hi_conv_b)))

# hi_norm = norm(hi_outi)
lo_result = np.zeros((lo.shape[0], lo.shape[1], 3))
lo_result[:,:,0] = lo_finv_r
lo_result[:,:,1] = lo_finv_g
lo_result[:,:,2] = lo_finv_b



e = utils.interactive_crop(norm(lo_result))

plt.imshow(e)
plt.show()
