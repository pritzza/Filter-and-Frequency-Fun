import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

def normalize_arr(arr):
    # Normalize the Laplacian image to range [0, 1]
    return cv.normalize(arr, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

def to_01(arr):
    if (arr.max() > 1.0):
        arr = arr / 255.0
    return arr

def collapse(stack):
    return np.sum(stack, axis=0)

def low_pass(im, sigma):
    G = cv.getGaussianKernel(ksize=int(6 * sigma), sigma=sigma)
    G2D = G @ G.T 
    low_freq = cv.filter2D(im, -1, G2D)
    return low_freq

def gaussian_stack(im, n, sigma):
    gs_stack = [im]  # start with the original image
    for i in range(1, n):
        sigma_i = sigma * (2 ** i)  # increase sigma at each level
        blurred_im = low_pass(gs_stack[-1], sigma_i)
        gs_stack.append(blurred_im)
    return gs_stack

def laplacian_stack(gaussian_stack):
    ls_stack = []
    for i in range(len(gaussian_stack) - 1):
        laplacian = gaussian_stack[i] - gaussian_stack[i + 1]  # subtract consecutive levels
        ls_stack.append(laplacian)
    ls_stack.append(gaussian_stack[-1])  # append the last Gaussian level
    return ls_stack

def display_stack(stack, title):
    plt.figure(figsize=(12, 6))
    for i, layer in enumerate(stack):
        plt.subplot(1, len(stack), i + 1)
        plt.imshow(layer, vmin=0, vmax=1)
        plt.title(f'{title} {i + 1}')
        plt.axis('off')
        im_layer = cv.normalize(layer, None, 0, 255, cv.NORM_MINMAX)
        im_layer = cv.cvtColor(im_layer.astype(np.uint8), cv.COLOR_BGR2RGB)
        cv.imwrite(f'res/{title}_{i}.jpg', im_layer)
    plt.show()

def display_img(img):
    plt.imshow(img, vmin=0, vmax=1)
    plt.show()

def create_half_mask(size):
    mask = np.zeros(size, dtype=np.float32)
    mask[:, size[0]//2:] = 1.
    return mask

def visualize_stacks(gs_stack, ls_stack):
    display_stack(gs_stack, 'Gaussian')
    display_stack(ls_stack, 'Laplacian')
    normalized_laplacian = [normalize_arr(x) for x in ls_stack]
    display_stack(normalized_laplacian, 'Normalized Laplacian')
    reconstructed = collapse(ls_stack)
    display_img(reconstructed)

def blend_images(im1, im2, mask):

    layers = 6
    sigma = 3

    g1 = gaussian_stack(im1, layers, sigma)
    g2 = gaussian_stack(im2, layers, sigma)
    
    l1 = laplacian_stack(g1)
    l2 = laplacian_stack(g2)

    #visualize_stacks(g1, l1)
    #visualize_stacks(g2, l2)

    masks = gaussian_stack(mask, layers, sigma)

    #display_stack(masks, 'Masks')

    for i in range(len(masks)):
        l1[i] = l1[i] * masks[i]
        l2[i] = l2[i] * (1. - masks[i]) # inverted

    #display_stack(l1, 'Masked L1')
    #display_stack(l2, 'Masked L2')

    masked_im1 = collapse(l1)
    masked_im2 = collapse(l2)

    blended = masked_im1 + masked_im2

    b = cv.normalize(blended, None, 0, 255, cv.NORM_MINMAX)
    b = cv.cvtColor(b.astype(np.uint8), cv.COLOR_BGR2RGB)  
    cv.imwrite('res/blended.jpg', b)

    return blended

def orapple():
    im2 = plt.imread('res/apple.jpeg') / 255.
    im1 = plt.imread('res/orange.jpeg') / 255.
    mask = create_half_mask(im1.shape)

    blended = blend_images(im1, im2, mask)
    display_img(blended)

def do_the_thing():

    im2 = plt.imread('res/snake.png')
    im1 = plt.imread('res/pikmin3.png')
    mask = plt.imread('res/pikmin3_mask.png')
    
    im1 = to_01(im1)
    im2 = to_01(im2)
    mask = to_01(mask)

    mask = 1.0 - mask
    
    if im1.shape[2] > 3:
        im1 = cv.cvtColor(im1, cv.COLOR_RGBA2RGB)

    if im2.shape[2] > 3:
        im2 = cv.cvtColor(im2, cv.COLOR_RGBA2RGB)

    if mask.shape[2] > 3:
        mask = cv.cvtColor(mask, cv.COLOR_RGBA2RGB)

    while im1.shape[0] * im1.shape[1] > 1000*1000:
        im1 = cv.resize(im1, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)
        mask = cv.resize(mask, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv.INTER_AREA)

    im2 = cv.resize(im2, (im1.shape[1], im1.shape[0]), interpolation=cv.INTER_AREA)

    blended = blend_images(im1, im2, mask)

    display_img(blended)

    return

    apple_gaussian = gaussian_stack(im1, layers, sigma)
    orange_gaussian = gaussian_stack(im2, layers, sigma)
    
    apple_laplacian = laplacian_stack(apple_gaussian)
    normalized_apple_laplacian = [normalize_arr(x) for x in apple_laplacian]

    orange_laplacian = laplacian_stack(orange_gaussian)
    normalized_orange_laplacian = [normalize_arr(x) for x in orange_laplacian]

    reconstructed_apple = collapse(apple_laplacian)
    reconstructed_orange = collapse(orange_laplacian)

    mask = create_half_mask(reconstructed_apple.shape)
    half_apple = reconstructed_apple * mask

    display_img(half_apple)

    display_stack(normalized_apple_laplacian, 'Normalized Apple Laplacian')
    display_stack(normalized_orange_laplacian, 'Normalized Orange Laplacian')

    visualize_stacks(apple_gaussian, apple_laplacian)
    visualize_stacks(orange_gaussian, orange_laplacian)


do_the_thing()
