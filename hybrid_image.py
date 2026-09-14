# Imports the print function from newer versions of python
from __future__ import print_function


# The Random module implements pseudo-random number generators
from builtins import range
import random
import os

# Numpy is the main package for scientific computing with Python.
# This will be one of our most used libraries in this class
import numpy as np

# The Time library helps us time code runtimes
import time

# PIL (Pillow) is a useful library for opening, manipulating, and saving images
from PIL import Image

# skimage (Scikit-Image) is a library for image processing
from skimage import color, io, filters
from skimage.feature import corner_peaks

# Matplotlib is a useful plotting library for python
import matplotlib.pyplot as plt
# This code is to make matplotlib figures appear inline in the
# notebook rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# PART 1 ##############################################################

def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))

    return kernel / np.sum(kernel)

def edge_pad(image, pad_height, pad_width):
    if len(image.shape) > 2:
        return np.pad(
            image,
            ((pad_height, pad_height), (pad_width, pad_width), (0,0)),
            mode="edge"
        )
    else:
        return np.pad(
                    image,
                    ((pad_height, pad_height), (pad_width, pad_width)),
                    mode="edge"
                )

def conv_2D(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Inputs:
        image: Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: numpy array of shape (Hk, Wk). Both Hk and Wk must be odd.

    Returns:
        out: numpy array of shape (Hi, Wi) or (Hi, Wi, colors).
    """

    Hk, Wk = kernel.shape
    pad_height = Hk // 2
    pad_width = Wk // 2

    if len(image.shape) > 2:
        Hi, Wi, colors = image.shape
        out = np.zeros((Hi, Wi, colors), dtype=image.dtype)
        padded_image = edge_pad(image, pad_height, pad_width)

        for c in range(colors):
            for i in range(Hi):
                for j in range(Wi):
                    neighborhood = padded_image[i:i+Hk, j:j+Wk, c]
                    out[i, j, c] = np.sum(neighborhood * kernel)
    else:
        Hi, Wi = image.shape
        out = np.zeros((Hi, Wi), dtype=image.dtype)
        padded_image = edge_pad(image, pad_height, pad_width)

        for i in range(Hi):
            for j in range(Wi):
                neighborhood = padded_image[i:i+Hk, j:j+Wk]
                out[i, j] = np.sum(neighborhood * kernel)

    return out

def gaussian_filter(img, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    return conv_2D(img, kernel) 

def hybrid_image(imgA, sizeA, sigmaA, imgB, sizeB, sigmaB, alpha):
    lowA = gaussian_filter(imgA, sizeA, sigmaA)
    lowB = gaussian_filter(imgB, sizeB, sigmaB)
    highB = imgB = lowB
    hybrid = lowA + alpha*highB
    return lowA, highB, hybrid

# PART 2 ##############################################################

#Alignment 1: Pixel Matching

#Alignment 2: Edge Detection

def partial_x(img):
    """ Computes partial x-derivative of input grayscale img.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    derivkernel = np.array([[-1,1]])
    out = conv_2D(img,derivkernel)

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    derivkernel = np.array([[-1,1]])
    derivkernel = derivkernel.reshape(-1,1)
    out = conv_2D(img,derivkernel)

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    Gx = partial_x(img)
    Gy = partial_y(img)

    G = np.sqrt(Gx*Gx + Gy*Gy)
    theta = np.arctan2(Gy, Gx)
    theta = np.degrees(theta)

    return G, theta

def shift_gradient(imgA, imgB, max_shift):
    # Convert RGB images to grayscale
    if len(imgA.shape) == 3:
        grayA = color.rgb2gray(imgA)
    else:
        grayA = imgA

    if len(imgB.shape) == 3:
        grayB = color.rgb2gray(imgB)
    else:
        grayB = imgB

    # Compute gradient magnitude for each image
    gradA, thetaA = gradient(grayA)
    gradB, thetaB = gradient(grayB)

    H, W = grayA.shape

    best_error = float('inf')
    best_dx = 0
    best_dy = 0

    # Try every possible translation
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):

            if dx >= 0:
                A_x1, A_x2 = 0, W - dx
                B_x1, B_x2 = dx, W
            else:
                A_x1, A_x2 = -dx, W
                B_x1, B_x2 = 0, W + dx

            if dy >= 0:
                A_y1, A_y2 = 0, H - dy
                B_y1, B_y2 = dy, H
            else:
                A_y1, A_y2 = -dy, H
                B_y1, B_y2 = 0, H + dy

            # Get overlapping regions
            A = gradA[A_y1:A_y2, A_x1:A_x2]
            B = gradB[B_y1:B_y2, B_x1:B_x2]

            # Compare gradient magnitudes
            error = np.mean((A - B) ** 2)

            if error < best_error:
                best_error = error
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy

folder = "Dataset/easy"

images = [
    f for f in os.listdir(folder)
]

imgA, imgB = random.sample(images, 2)

imgA = np.array(Image.open(os.path.join(folder, imgA))).astype(float) / 255.0
plt.savefig("test1.png", bbox_inches="tight")
imgB = np.array(Image.open(os.path.join(folder, imgB))).astype(float) / 255.0
plt.savefig("test2.png", bbox_inches="tight")

print("Image A:", imgA)
print("Image B:", imgB)


dx2, dy2 = shift_gradient(imgA, imgB, max_shift=20)

print("Gradient-based method:")
print("dx =", dx2, "dy =", dy2)
