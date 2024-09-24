import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

def normalize_array(arr):
    min_val, max_val = np.min(arr), np.max(arr)
    return (arr - min_val) / (max_val - min_val) if min_val != max_val else np.zeros_like(arr)

def normalize_to_8bit(arr):
    return (normalize_array(arr) * 255).astype(np.uint8)

def display_images(image_title_pairs):
    n = len(image_title_pairs)
    plt.figure(figsize=(4*n, 2*n))
    for i, (image, title) in enumerate(image_title_pairs, start=1):
        plt.subplot(1, n, i)
        plt.imshow(normalize_to_8bit(image), cmap='gray')
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        cv.imwrite(title + ".jpg", image)
    plt.show()

def do_the_thing():
    img = cv.imread('res/cameraman.png', cv.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    Dx = np.array([[1, -1]])
    Dy = np.array([[1], [-1]])
    horizontal_edges = convolve2d(img, Dx, mode='same')
    vertical_edges = convolve2d(img, Dy, mode='same')
    grad_mag = np.sqrt(horizontal_edges**2 + vertical_edges**2)

    _, bin_grad = cv.threshold(normalize_to_8bit(grad_mag), 30, 255, cv.THRESH_BINARY)

    sigma = 2
    ksize = int(6 * sigma) | 1  # ensure ksize is odd
    G = cv.getGaussianKernel(ksize=ksize, sigma=sigma)
    G2D = G @ G.T  # 2D Gaussian from outer product of 1D kernels
    
    # two convolution passes (blur then differentiate)
    smooth_img = cv.filter2D(img, -1, G2D)
    s_horizontal_edges = cv.filter2D(smooth_img, -1, Dx)
    s_vertical_edges = cv.filter2D(smooth_img, -1, Dy)
    s_grad_mag = np.sqrt(s_horizontal_edges**2 + s_vertical_edges**2)
    _, s_bin_grad = cv.threshold(normalize_to_8bit(s_grad_mag), 127, 255, cv.THRESH_BINARY) 

    # single pass (combine kernels)
    DxoG = convolve2d(G2D, Dx, mode='same')
    DyoG = convolve2d(G2D, Dy, mode='same')
    
    dog_horizontal_edges = cv.filter2D(img, -1, DxoG)
    dog_vertical_edges = cv.filter2D(img, -1, DyoG)
    dog_grad_mag = np.sqrt(dog_horizontal_edges**2 + dog_vertical_edges**2)
    _, dog_bin_grad = cv.threshold(normalize_to_8bit(dog_grad_mag), 127, 255, cv.THRESH_BINARY)

    images = [
        (bin_grad, 'Finite Difference Binary Gradient'),
        (s_bin_grad, 'Smoothed Binary Gradient'),
        (dog_bin_grad, 'DoG Binary Gradient')
    ]

    display_images(images)

do_the_thing()
