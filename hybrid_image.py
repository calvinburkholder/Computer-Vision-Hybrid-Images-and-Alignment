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
    highB = imgB - lowB
    hybrid = lowA + alpha*highB
    return lowA, highB, hybrid

def subsample(img, size, sigma):
    blurred = gaussian_filter(img, size, sigma)
    subsampled = blurred[::2, ::2]
    return subsampled


# PART 2 ##############################################################

#Alignment 1: Pixel Matching

def shift_pixel(imgA, imgB, max_shift):
    
    grayA = color.rgb2gray(imgA)
    grayB = color.rgb2gray(imgB)

    H, W = grayA.shape

    best_error = float('inf')
    best_dx = 0
    best_dy = 0

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

            A = grayA[A_y1:A_y2, A_x1:A_x2]
            B = grayB[B_y1:B_y2, B_x1:B_x2]

            error = np.mean((A - B) ** 2)

            if error < best_error:
                best_error = error
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy


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

    return G

def shift_gradient(imgA, imgB, max_shift):
    
    grayA = color.rgb2gray(imgA)
    grayB = color.rgb2gray(imgB)

    gradA = gradient(grayA)
    gradB = gradient(grayB)

    H, W = grayA.shape

    best_error = float('inf')
    best_dx = 0
    best_dy = 0

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

            A = gradA[A_y1:A_y2, A_x1:A_x2]
            B = gradB[B_y1:B_y2, B_x1:B_x2]

            error = np.mean((A - B) ** 2)

            if error < best_error:
                best_error = error
                best_dx = dx
                best_dy = dy

    return best_dx, best_dy

def shift_image(img, dx, dy):
    shifted = np.zeros_like(img)

    H,W = img.shape[:2]

    if dx >= 0:
        src_x1 = 0
        src_x2 = W - dx
        dst_x1 = dx
        dst_x2 = W
    else:
        src_x1 = -dx
        src_x2 = W
        dst_x1 = 0
        dst_x2 = W + dx

    if dy >= 0:
        src_y1 = 0
        src_y2 = H - dy
        dst_y1 = dy
        dst_y2 = H
    else:
        src_y1 = -dy
        src_y2 = H
        dst_y1 = 0
        dst_y2 = H + dy

    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = \
        img[src_y1:src_y2, src_x1:src_x2]

    return shifted

folder_aligned = "dataset/aligned"

images_aligned = [
    f for f in os.listdir(folder_aligned)
]

folder_easy = "dataset/easy"

images_easy = [
    f for f in os.listdir(folder_easy)
]

folder_large = "dataset/large"

images_large = [
    f for f in os.listdir(folder_large)
]

folder_medium = "dataset/medium"

images_medium = [
    f for f in os.listdir(folder_medium)
]

escape = 0
while escape == 0:
    print()
    print("Welcome to Hybrid Images and Alignment")
    print()
    print("0. Exit")
    print("1. Pre-Aligned Images")
    print("2. Un-Aligned Images")
    selection = input("Please select a function: ")
    print()

    if selection == "1":
        while escape == 0:
            print("0. Go Back")
            print("1. Random Images")
            print("2. Select Images")
            selection = input("Please select an option: ")
            print()
            if selection == "1":
                imgA, imgB = random.sample(images_aligned, 2)
                imgA = np.array(Image.open(os.path.join(folder_aligned, imgA))).astype(float) / 255.0
                imgB = np.array(Image.open(os.path.join(folder_aligned, imgB))).astype(float) / 255.0

                low_sd = float(input("Select a sigma for image A/low-frequency: "))
                high_sd = float(input("Select a sigma for image B/high frequency: "))
                weight = float(input("Select a weight: "))

                lowA, highB, hybrid = hybrid_image(imgA,31,low_sd,imgB,31,high_sd,weight)

                hybrid_subsample = subsample(hybrid, 11, 5)

                fig = plt.figure(figsize=(10, 4))

                ax1 = fig.add_axes([0.02, 0.15, 0.22, 0.7])
                ax1.imshow(lowA)
                ax1.set_title("lowA")
                ax1.axis("off")

                ax2 = fig.add_axes([0.26, 0.15, 0.22, 0.7])
                ax2.imshow(highB)
                ax2.set_title("highB")
                ax2.axis("off")

                ax3 = fig.add_axes([0.50, 0.15, 0.22, 0.7])
                ax3.imshow(hybrid)
                ax3.set_title("hybrid big")
                ax3.axis("off")

                ax4 = fig.add_axes([0.78, 0.30, 0.07, 0.2])
                ax4.imshow(hybrid_subsample)
                ax4.set_title("hybrid small")
                ax4.axis("off")
                
                plt.savefig("hybrid.png", bbox_inches="tight")

                print()
                continue
            elif selection == "2":
                print("Select two numbers from 1-100 as 3 digits (EX: 021)")
                imgA_selection = input("Select an image A number: ")
                imgA = np.array(Image.open(os.path.join(folder_aligned, "face_0"+imgA_selection+".png"))).astype(float) / 255.0
                imgB_selection = input("Select an image B number: ")
                imgB = np.array(Image.open(os.path.join(folder_aligned, "face_0"+imgB_selection+".png"))).astype(float) / 255.0

                low_sd = float(input("Select a sigma for image A/low-frequency: "))
                high_sd = float(input("Select a sigma for image B/high frequency: "))
                weight = float(input("Select a weight: "))

                lowA, highB, hybrid = hybrid_image(imgA,31,low_sd,imgB,31,high_sd,weight)

                hybrid_subsample = subsample(hybrid, 11, 5)

                fig = plt.figure(figsize=(10, 4))

                ax1 = fig.add_axes([0.02, 0.15, 0.22, 0.7])
                ax1.imshow(lowA)
                ax1.set_title("lowA")
                ax1.axis("off")

                ax2 = fig.add_axes([0.26, 0.15, 0.22, 0.7])
                ax2.imshow(highB)
                ax2.set_title("highB")
                ax2.axis("off")

                ax3 = fig.add_axes([0.50, 0.15, 0.22, 0.7])
                ax3.imshow(hybrid)
                ax3.set_title("hybrid big")
                ax3.axis("off")

                ax4 = fig.add_axes([0.78, 0.30, 0.07, 0.2])
                ax4.imshow(hybrid_subsample)
                ax4.set_title("hybrid small")
                ax4.axis("off")
                
                plt.savefig("hybrid.png", bbox_inches="tight")

                print()          
                continue
            elif selection == "0":
                break
            else:
                print("Invalid Selection. Please try again.")
                continue

    elif selection == "2":
        while escape == 0:
            print()
            print("0. Go Back")
            print("1. Easy")
            print("2. Medium")
            print("3. Large")
            selection = input("Please select an alignment difficulty: ")
            print()
            if selection == "0":
                break
            elif selection == "1":
                while escape == 0:
                    print("0. Go Back")
                    print("1. Random Images")
                    print("2. Select Images")
                    selection = input("Please select an option: ")
                    print()
                    if selection == "1":
                        imgA, imgB = random.sample(images_easy, 2)
                        imgA = np.array(Image.open(os.path.join(folder_aligned, imgA))).astype(float) / 255.0
                        imgB = np.array(Image.open(os.path.join(folder_easy, imgB))).astype(float) / 255.0

                        dx_pixel, dy_pixel = shift_pixel(imgA, imgB, 10)
                        dx_gradient, dy_gradient = shift_gradient(imgA, imgB, 10)

                        print("Pixel method:")
                        print("dx =", dx_pixel, "dy =", dy_pixel)

                        print("gradient method:")
                        print("dx =", dx_gradient, "dy =", dy_gradient)

                        alignB_pixel = shift_image(imgB, dx_pixel, dy_pixel)
                        alignB_gradient =  shift_image(imgB, dx_gradient, dy_gradient)

                        low_sd = float(input("Select a sigma for image A/low-frequency: "))
                        high_sd = float(input("Select a sigma for image B/high frequency: "))
                        weight = float(input("Select a weight: "))

                        lowA, highB, hybrid_pixel = hybrid_image(imgA, 31, low_sd, alignB_pixel, 31, high_sd, weight)
                        lowA, highB, hybrid_gradient = hybrid_image(imgA, 31, low_sd, alignB_gradient, 31, high_sd, weight)

                        hybrid_pixel_subsample = subsample(hybrid_pixel, 11, 5)
                        hybrid_gradient_subsample = subsample(hybrid_gradient, 11, 5)

                        fig = plt.figure(figsize=(12, 4))

                        ax1 = fig.add_axes([0.02, 0.15, 0.22, 0.7])
                        ax1.imshow(lowA)
                        ax1.set_title("lowA")
                        ax1.axis("off")

                        ax2 = fig.add_axes([0.20, 0.15, 0.22, 0.7])
                        ax2.imshow(highB)
                        ax2.set_title("highB")
                        ax2.axis("off")

                        ax3 = fig.add_axes([0.40, 0.15, 0.22, 0.7])
                        ax3.imshow(hybrid_pixel)
                        ax3.set_title("hybrid pixel big")
                        ax3.axis("off")

                        ax4 = fig.add_axes([0.62, 0.30, 0.07, 0.2])
                        ax4.imshow(hybrid_pixel_subsample)
                        ax4.set_title("hybrid pixel small")
                        ax4.axis("off")

                        ax5 = fig.add_axes([0.7, 0.15, 0.22, 0.7])
                        ax5.imshow(hybrid_gradient)
                        ax5.set_title("hybrid gradient big")
                        ax5.axis("off")

                        ax6 = fig.add_axes([0.95, 0.30, 0.07, 0.2])
                        ax6.imshow(hybrid_gradient_subsample)
                        ax6.set_title("hybrid gradient small")
                        ax6.axis("off")

                        
                        plt.savefig("hybrid.png", bbox_inches="tight")

                        print()
                        continue
                    elif selection == "2":
                        print("Select two numbers from 1-100 as 3 digits (EX: 021)")
                        imgA_selection = input("Select an image A number: ")
                        imgA = np.array(Image.open(os.path.join(folder_aligned, "face_0"+imgA_selection+".png"))).astype(float) / 255.0
                        imgB_selection = input("Select an image B number: ")
                        imgB = np.array(Image.open(os.path.join(folder_easy, "face_0"+imgB_selection+".png"))).astype(float) / 255.0

                        dx_pixel, dy_pixel = shift_pixel(imgA, imgB, 10)
                        dx_gradient, dy_gradient = shift_gradient(imgA, imgB, 10)

                        print("Pixel method:")
                        print("dx =", dx_pixel, "dy =", dy_pixel)

                        print("gradient method:")
                        print("dx =", dx_gradient, "dy =", dy_gradient)

                        alignB_pixel = shift_image(imgB, dx_pixel, dy_pixel)
                        alignB_gradient =  shift_image(imgB, dx_gradient, dy_gradient)

                        low_sd = float(input("Select a sigma for image A/low-frequency: "))
                        high_sd = float(input("Select a sigma for image B/high frequency: "))
                        weight = float(input("Select a weight: "))

                        lowA, highB, hybrid_pixel = hybrid_image(imgA, 31, low_sd, alignB_pixel, 31, high_sd, weight)
                        lowA, highB, hybrid_gradient = hybrid_image(imgA, 31, low_sd, alignB_gradient, 31, high_sd, weight)

                        hybrid_pixel_subsample = subsample(hybrid_pixel, 11, 5)
                        hybrid_gradient_subsample = subsample(hybrid_gradient, 11, 5)

                        fig = plt.figure(figsize=(12, 4))

                        ax1 = fig.add_axes([0.02, 0.15, 0.22, 0.7])
                        ax1.imshow(lowA)
                        ax1.set_title("lowA")
                        ax1.axis("off")

                        ax2 = fig.add_axes([0.20, 0.15, 0.22, 0.7])
                        ax2.imshow(highB)
                        ax2.set_title("highB")
                        ax2.axis("off")

                        ax3 = fig.add_axes([0.40, 0.15, 0.22, 0.7])
                        ax3.imshow(hybrid_pixel)
                        ax3.set_title("hybrid pixel big")
                        ax3.axis("off")

                        ax4 = fig.add_axes([0.62, 0.30, 0.07, 0.2])
                        ax4.imshow(hybrid_pixel_subsample)
                        ax4.set_title("hybrid pixel small")
                        ax4.axis("off")

                        ax5 = fig.add_axes([0.7, 0.15, 0.22, 0.7])
                        ax5.imshow(hybrid_gradient)
                        ax5.set_title("hybrid gradient big")
                        ax5.axis("off")

                        ax6 = fig.add_axes([0.95, 0.30, 0.07, 0.2])
                        ax6.imshow(hybrid_gradient_subsample)
                        ax6.set_title("hybrid gradient small")
                        ax6.axis("off")

                        
                        plt.savefig("hybrid.png", bbox_inches="tight")

                        print()
                        continue
                    elif selection == "0":
                        break
                    else:
                        print("Invalid Selection. Please try again.")
                        continue
            elif selection == "2":
                while escape == 0:
                    print("0. Go Back")
                    print("1. Random Images")
                    print("2. Select Images")
                    selection = input("Please select an option: ")
                    print()
                    if selection == "1":
                        imgA, imgB = random.sample(images_medium, 2)
                        imgA = np.array(Image.open(os.path.join(folder_aligned, imgA))).astype(float) / 255.0
                        imgB = np.array(Image.open(os.path.join(folder_medium, imgB))).astype(float) / 255.0

                        dx_pixel, dy_pixel = shift_pixel(imgA, imgB, 25)
                        dx_gradient, dy_gradient = shift_gradient(imgA, imgB, 25)

                        print("Pixel method:")
                        print("dx =", dx_pixel, "dy =", dy_pixel)

                        print("gradient method:")
                        print("dx =", dx_gradient, "dy =", dy_gradient)

                        alignB_pixel = shift_image(imgB, dx_pixel, dy_pixel)
                        alignB_gradient =  shift_image(imgB, dx_gradient, dy_gradient)

                        low_sd = float(input("Select a sigma for image A/low-frequency: "))
                        high_sd = float(input("Select a sigma for image B/high frequency: "))
                        weight = float(input("Select a weight: "))

                        lowA, highB, hybrid_pixel = hybrid_image(imgA, 31, low_sd, alignB_pixel, 31, high_sd, weight)
                        lowA, highB, hybrid_gradient = hybrid_image(imgA, 31, low_sd, alignB_gradient, 31, high_sd, weight)

                        hybrid_pixel_subsample = subsample(hybrid_pixel, 11, 5)
                        hybrid_gradient_subsample = subsample(hybrid_gradient, 11, 5)

                        fig = plt.figure(figsize=(12, 4))

                        ax1 = fig.add_axes([0.02, 0.15, 0.22, 0.7])
                        ax1.imshow(lowA)
                        ax1.set_title("lowA")
                        ax1.axis("off")

                        ax2 = fig.add_axes([0.20, 0.15, 0.22, 0.7])
                        ax2.imshow(highB)
                        ax2.set_title("highB")
                        ax2.axis("off")

                        ax3 = fig.add_axes([0.40, 0.15, 0.22, 0.7])
                        ax3.imshow(hybrid_pixel)
                        ax3.set_title("hybrid pixel big")
                        ax3.axis("off")

                        ax4 = fig.add_axes([0.62, 0.30, 0.07, 0.2])
                        ax4.imshow(hybrid_pixel_subsample)
                        ax4.set_title("hybrid pixel small")
                        ax4.axis("off")

                        ax5 = fig.add_axes([0.7, 0.15, 0.22, 0.7])
                        ax5.imshow(hybrid_gradient)
                        ax5.set_title("hybrid gradient big")
                        ax5.axis("off")

                        ax6 = fig.add_axes([0.95, 0.30, 0.07, 0.2])
                        ax6.imshow(hybrid_gradient_subsample)
                        ax6.set_title("hybrid gradient small")
                        ax6.axis("off")

                        
                        plt.savefig("hybrid.png", bbox_inches="tight")

                        print()
                        continue
                    elif selection == "2":
                        print("Select two numbers from 1-100 as 3 digits (EX: 021)")
                        imgA_selection = input("Select an image A number: ")
                        imgA = np.array(Image.open(os.path.join(folder_aligned, "face_0"+imgA_selection+".png"))).astype(float) / 255.0
                        imgB_selection = input("Select an image B number: ")
                        imgB = np.array(Image.open(os.path.join(folder_medium, "face_0"+imgB_selection+".png"))).astype(float) / 255.0

                        dx_pixel, dy_pixel = shift_pixel(imgA, imgB, 25)
                        dx_gradient, dy_gradient = shift_gradient(imgA, imgB, 25)

                        print("Pixel method:")
                        print("dx =", dx_pixel, "dy =", dy_pixel)

                        print("gradient method:")
                        print("dx =", dx_gradient, "dy =", dy_gradient)

                        alignB_pixel = shift_image(imgB, dx_pixel, dy_pixel)
                        alignB_gradient =  shift_image(imgB, dx_gradient, dy_gradient)

                        low_sd = float(input("Select a sigma for image A/low-frequency: "))
                        high_sd = float(input("Select a sigma for image B/high frequency: "))
                        weight = float(input("Select a weight: "))

                        lowA, highB, hybrid_pixel = hybrid_image(imgA, 31, low_sd, alignB_pixel, 31, high_sd, weight)
                        lowA, highB, hybrid_gradient = hybrid_image(imgA, 31, low_sd, alignB_gradient, 31, high_sd, weight)

                        hybrid_pixel_subsample = subsample(hybrid_pixel, 11, 5)
                        hybrid_gradient_subsample = subsample(hybrid_gradient, 11, 5)

                        fig = plt.figure(figsize=(12, 4))

                        ax1 = fig.add_axes([0.02, 0.15, 0.22, 0.7])
                        ax1.imshow(lowA)
                        ax1.set_title("lowA")
                        ax1.axis("off")

                        ax2 = fig.add_axes([0.20, 0.15, 0.22, 0.7])
                        ax2.imshow(highB)
                        ax2.set_title("highB")
                        ax2.axis("off")

                        ax3 = fig.add_axes([0.40, 0.15, 0.22, 0.7])
                        ax3.imshow(hybrid_pixel)
                        ax3.set_title("hybrid pixel big")
                        ax3.axis("off")

                        ax4 = fig.add_axes([0.62, 0.30, 0.07, 0.2])
                        ax4.imshow(hybrid_pixel_subsample)
                        ax4.set_title("hybrid pixel small")
                        ax4.axis("off")

                        ax5 = fig.add_axes([0.7, 0.15, 0.22, 0.7])
                        ax5.imshow(hybrid_gradient)
                        ax5.set_title("hybrid gradient big")
                        ax5.axis("off")

                        ax6 = fig.add_axes([0.95, 0.30, 0.07, 0.2])
                        ax6.imshow(hybrid_gradient_subsample)
                        ax6.set_title("hybrid gradient small")
                        ax6.axis("off")

                        
                        plt.savefig("hybrid.png", bbox_inches="tight")

                        print()
                        continue
                    elif selection == "0":
                        break
                    else:
                        print("Invalid Selection. Please try again.")
                        continue
            elif selection == "3":
                while escape == 0:
                    print("0. Go Back")
                    print("1. Random Images")
                    print("2. Select Images")
                    selection = input("Please select an option: ")
                    print()
                    if selection == "1":
                        imgA, imgB = random.sample(images_large, 2)
                        imgA = np.array(Image.open(os.path.join(folder_aligned, imgA))).astype(float) / 255.0
                        imgB = np.array(Image.open(os.path.join(folder_large, imgB))).astype(float) / 255.0

                        dx_pixel, dy_pixel = shift_pixel(imgA, imgB, 40)
                        dx_gradient, dy_gradient = shift_gradient(imgA, imgB, 40)

                        print("Pixel method:")
                        print("dx =", dx_pixel, "dy =", dy_pixel)

                        print("gradient method:")
                        print("dx =", dx_gradient, "dy =", dy_gradient)

                        alignB_pixel = shift_image(imgB, dx_pixel, dy_pixel)
                        alignB_gradient =  shift_image(imgB, dx_gradient, dy_gradient)

                        low_sd = float(input("Select a sigma for image A/low-frequency: "))
                        high_sd = float(input("Select a sigma for image B/high frequency: "))
                        weight = float(input("Select a weight: "))

                        lowA, highB, hybrid_pixel = hybrid_image(imgA, 31, low_sd, alignB_pixel, 31, high_sd, weight)
                        lowA, highB, hybrid_gradient = hybrid_image(imgA, 31, low_sd, alignB_gradient, 31, high_sd, weight)

                        hybrid_pixel_subsample = subsample(hybrid_pixel, 11, 5)
                        hybrid_gradient_subsample = subsample(hybrid_gradient, 11, 5)

                        fig = plt.figure(figsize=(12, 4))

                        ax1 = fig.add_axes([0.02, 0.15, 0.22, 0.7])
                        ax1.imshow(lowA)
                        ax1.set_title("lowA")
                        ax1.axis("off")

                        ax2 = fig.add_axes([0.20, 0.15, 0.22, 0.7])
                        ax2.imshow(highB)
                        ax2.set_title("highB")
                        ax2.axis("off")

                        ax3 = fig.add_axes([0.40, 0.15, 0.22, 0.7])
                        ax3.imshow(hybrid_pixel)
                        ax3.set_title("hybrid pixel big")
                        ax3.axis("off")

                        ax4 = fig.add_axes([0.62, 0.30, 0.07, 0.2])
                        ax4.imshow(hybrid_pixel_subsample)
                        ax4.set_title("hybrid pixel small")
                        ax4.axis("off")

                        ax5 = fig.add_axes([0.7, 0.15, 0.22, 0.7])
                        ax5.imshow(hybrid_gradient)
                        ax5.set_title("hybrid gradient big")
                        ax5.axis("off")

                        ax6 = fig.add_axes([0.95, 0.30, 0.07, 0.2])
                        ax6.imshow(hybrid_gradient_subsample)
                        ax6.set_title("hybrid gradient small")
                        ax6.axis("off")

                        
                        plt.savefig("hybrid.png", bbox_inches="tight")

                        print()
                        continue
                    elif selection == "2":
                        print("Select two numbers from 1-100 as 3 digits (EX: 021)")
                        imgA_selection = input("Select an image A number: ")
                        imgA = np.array(Image.open(os.path.join(folder_aligned, "face_0"+imgA_selection+".png"))).astype(float) / 255.0
                        imgB_selection = input("Select an image B number: ")
                        imgB = np.array(Image.open(os.path.join(folder_large, "face_0"+imgB_selection+".png"))).astype(float) / 255.0

                        dx_pixel, dy_pixel = shift_pixel(imgA, imgB, 40)
                        dx_gradient, dy_gradient = shift_gradient(imgA, imgB, 40)

                        print("Pixel method:")
                        print("dx =", dx_pixel, "dy =", dy_pixel)

                        print("gradient method:")
                        print("dx =", dx_gradient, "dy =", dy_gradient)

                        alignB_pixel = shift_image(imgB, dx_pixel, dy_pixel)
                        alignB_gradient =  shift_image(imgB, dx_gradient, dy_gradient)

                        low_sd = float(input("Select a sigma for image A/low-frequency: "))
                        high_sd = float(input("Select a sigma for image B/high frequency: "))
                        weight = float(input("Select a weight: "))

                        lowA, highB, hybrid_pixel = hybrid_image(imgA, 31, low_sd, alignB_pixel, 31, high_sd, weight)
                        lowA, highB, hybrid_gradient = hybrid_image(imgA, 31, low_sd, alignB_gradient, 31, high_sd, weight)

                        hybrid_pixel_subsample = subsample(hybrid_pixel, 11, 5)
                        hybrid_gradient_subsample = subsample(hybrid_gradient, 11, 5)

                        fig = plt.figure(figsize=(12, 4))

                        ax1 = fig.add_axes([0.02, 0.15, 0.22, 0.7])
                        ax1.imshow(lowA)
                        ax1.set_title("lowA")
                        ax1.axis("off")

                        ax2 = fig.add_axes([0.20, 0.15, 0.22, 0.7])
                        ax2.imshow(highB)
                        ax2.set_title("highB")
                        ax2.axis("off")

                        ax3 = fig.add_axes([0.40, 0.15, 0.22, 0.7])
                        ax3.imshow(hybrid_pixel)
                        ax3.set_title("hybrid pixel big")
                        ax3.axis("off")

                        ax4 = fig.add_axes([0.62, 0.30, 0.07, 0.2])
                        ax4.imshow(hybrid_pixel_subsample)
                        ax4.set_title("hybrid pixel small")
                        ax4.axis("off")

                        ax5 = fig.add_axes([0.7, 0.15, 0.22, 0.7])
                        ax5.imshow(hybrid_gradient)
                        ax5.set_title("hybrid gradient big")
                        ax5.axis("off")

                        ax6 = fig.add_axes([0.95, 0.30, 0.07, 0.2])
                        ax6.imshow(hybrid_gradient_subsample)
                        ax6.set_title("hybrid gradient small")
                        ax6.axis("off")

                        
                        plt.savefig("hybrid.png", bbox_inches="tight")

                        print()
                        continue
                    elif selection == "0":
                        break
                    else:
                        print("Invalid Selection. Please try again.")
                        continue                
            else:
                print("Invalid Selection. Please try again.")
                continue

    elif selection == "0":
        break
    else:
        print("Invalid Selection. Please try again.")
        continue
