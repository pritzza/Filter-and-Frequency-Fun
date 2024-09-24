import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

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

    images = [
        #(img, 'Original'),
        (horizontal_edges, 'Horizontal Edges'),
        (vertical_edges, 'Vertical Edges'),
        #(grad_mag, 'Gradient Magnitude'),
        #(bin_grad, 'Binary Gradient')
    ]

    display_images(images)

do_the_thing()
