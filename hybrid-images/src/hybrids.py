import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.signal import convolve2d

import utils

INPUT_ROOT_DIR = 'data/in/'
OUTPUT_ROOT_DIR = 'data/out/' 

lo = cv.imread(INPUT_ROOT_DIR + "derek.jpg",flags=cv.IMREAD_GRAYSCALE)
hi = cv.imread(INPUT_ROOT_DIR + "nutmeg.jpg",flags=cv.IMREAD_GRAYSCALE)

pts_1 = utils.prompt_eye_selection(lo)
pts_2 = utils.prompt_eye_selection(hi)

al_lo, al_hi = utils.align_images(INPUT_ROOT_DIR + "derek.jpg", INPUT_ROOT_DIR + "nutmeg.jpg", pts_1, pts_2)

hi_fft = np.fft.fftshift(np.fft.fft2(al_hi))
gkern = utils.gaussian_kernel(.82, 3)
hi_blurred = convolve2d(hi_fft, gkern, mode='same')
hi_out = hi_fft - hi_blurred

lo_fft = np.fft.fftshift(np.fft.fft2(al_lo))
gkern = utils.gaussian_kernel(.42, 3)
lo_blurred = convolve2d(lo_fft, gkern, mode='same') + hi_out
lo_outi = np.abs(np.fft.ifft2(np.fft.ifftshift(lo_blurred)))

plt.imshow(lo_outi, cmap='gray')
plt.show()
# plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(a)))))
# plt.show()
