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
        cv.imwrite(title + ".jpg", image)

    plt.show()

def sharpen(img, sharpness, G2D):
    r, g, b = cv.split(img)
    channels = [r,g,b]
    new_channels = []

    for im in channels:
        low_freq = convolve2d(im, G2D, mode='same')
        high_freq = im - low_freq
        sharpened = im + high_freq * sharpness
        sharpened = np.clip(sharpened, 0, 255)
        new_channels.append(sharpened)

    return cv.merge(new_channels)

def do_the_thing():
    img = cv.imread('res/taj.jpg')
    assert img is not None, "file could not be read, check with os.path.exists()"

    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    G = cv.getGaussianKernel(ksize=15, sigma=2)
    G2D = G @ G.T  # 2D gaussian from outer product of 1D kernels

    sharpness = 1
    num_iterations = 15
    sharpened_images = [img]

    for i in range(num_iterations + 1):
        sharpened_img = sharpen(sharpened_images[-1], sharpness, G2D)
        sharpened_images.append(sharpened_img)

    high_freq = sharpened_images[0] - sharpened_images[1]

    images = [
        (high_freq, 'High Frequency Image'),
        (sharpened_images[0], 'Original'),
        (sharpened_images[1], 'Sharpened'),
        (sharpened_images[14], 'Deep Fried'),
    ]

    display_images(images)

do_the_thing()
