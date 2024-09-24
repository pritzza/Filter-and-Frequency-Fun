import matplotlib.pyplot as plt
from align_image_code import align_images
import cv2 as cv
import numpy as np

def to_01(arr):
    if (arr.max() > 1.0):
        arr = arr / 255.0
    return arr

def low_pass(im, sigma):
    G = cv.getGaussianKernel(ksize=int(sigma * 3), sigma=sigma)
    G2D = G @ G.T  # 2D gaussian from outer product of 1D kernels
    low_freq = cv.filter2D(im, -1, G2D)
    return low_freq

def high_pass(im, sigma):
    low_freq = low_pass(im, sigma)
    high_freq = im - low_freq
    return high_freq

# im1 high pass, im2 low pass
def hybrid_image(im1, im2, sigma1, sigma2):
    high_freq = high_pass(im1, sigma1)
    low_freq = low_pass(im2, sigma2)
    hybrid = (low_freq + high_freq) * 255
    hybrid = np.clip(hybrid, 0, 255).astype(np.uint8)
    return hybrid
    
def laplacian_stack(im, n):
    return [im for _ in range(n)]

def gaussian_stack(im, n, sigma):
    gs_stack  = laplacian_stack(im, n)
    for layer in gs_stack:
        layer = low_pass(layer, sigma * 2 ** n)
    return gs_stack

def do_the_thing():
    # First load images

    # high sf
    im1 = plt.imread('res/nutmeg.jpg')

    # low sf
    im2 = plt.imread('res/DerekPicture.jpg')

    while im1.shape[0] * im1.shape[1] > 1000*1000:
        im1 = cv.resize(im1, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    while im2.shape[0] * im2.shape[1] > 1000*1000:
        im2 = cv.resize(im2, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)


    im1 = to_01(im1)
    im2 = to_01(im2)
    
    if im1.shape[2] > 3:
        im1 = cv.cvtColor(im1, cv.COLOR_RGBA2RGB)

    if im2.shape[2] > 3:
        im2 = cv.cvtColor(im2, cv.COLOR_RGBA2RGB)

    # Next align images (this code is provided, but may be improved)
    im1_aligned, im2_aligned = align_images(im1, im2)

    ## You will provide the code below. Sigma1 and sigma2 are arbitrary 
    ## cutoff values for the high and low frequencies
    sigma1 = 15
    sigma2 = 15
    hybrid = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)

    # Display the hybrid image with proper scaling
    plt.figure()
    plt.imshow(hybrid, cmap='gray', vmin=0, vmax=255)
    plt.show()


    # show frequency domain
    gray_image = cv.cvtColor(hybrid, cv.COLOR_RGB2GRAY)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image)))))

    # show the low and high frequency images
    high_freq = high_pass(im1_aligned, sigma2)
    low_freq = low_pass(im2_aligned, sigma1)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Low Frequency Image")
    plt.imshow(low_freq, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("High Frequency Image")
    plt.imshow(high_freq, cmap='gray')
    plt.show()

do_the_thing()
