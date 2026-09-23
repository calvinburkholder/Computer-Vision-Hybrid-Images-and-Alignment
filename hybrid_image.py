from __future__ import print_function
from builtins import range
import os
import argparse
 
import numpy as np
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt
 
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
 
DATASET_ROOT = "dataset"
DIFFICULTIES = {
    "aligned": 0,
    "easy": 10,
    "medium": 25,
    "large": 40,
}
 
 
# ---------------------------------------------------------------------------
# PART 1: Hybrid images
# ---------------------------------------------------------------------------
 
def gaussian_kernel(size, sigma):
    
    kernel = np.zeros((size, size))
    center = size // 2

    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return kernel / np.sum(kernel)
 
 
def kernel_size_for_sigma(sigma, min_size=5):
    size = 2 * int(np.ceil(3 * sigma)) + 1
    return max(size, min_size)
 
 
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

def hybrid_image(imgA, sigmaA, imgB, sigmaB, alpha):
    sizeA = kernel_size_for_sigma(sigmaA)
    sizeB = kernel_size_for_sigma(sigmaB)
    lowA = gaussian_filter(imgA, sizeA, sigmaA)
    lowB = gaussian_filter(imgB, sizeB, sigmaB)
    highB = imgB - lowB
    hybrid = lowA + alpha*highB
    return lowA, highB, hybrid

def subsample(img, sigma):
    size = kernel_size_for_sigma(sigma)
    blurred = gaussian_filter(img, size, sigma)
    subsampled = blurred[::2, ::2]
    return subsampled
 
 
# ---------------------------------------------------------------------------
# PART 2: Alignment
# ---------------------------------------------------------------------------
 
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
    theta = np.degrees(np.arctan2(Gy, Gx))

    G = np.sqrt(Gx*Gx + Gy*Gy)

    return G, theta

def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W), dtype=G.dtype)

    # Round the gradient direction to the nearest 45 degrees
    # This maps angles to 0, 45, 90, 135, 180, 225, 270, 315, 360
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = theta % 360 # Ensure angles are strictly < 360

    for i in range(1, H - 1): # Iterate from 1 to H-2 to avoid border issues for neighbors
        for j in range(1, W - 1): # Iterate from 1 to W-2 to avoid border issues for neighbors
            angle = theta[i, j]
            q = 0 # neighbor in one direction
            r = 0 # neighbor in opposite direction

            # Horizontal direction (0, 180 degrees)
            if (angle == 0) or (angle == 180):
                q = G[i+1,j] ### FILL IN HERE
                r = G[i-1,j] ### FILL IN HERE
            # Diagonal direction (45, 225 degrees)
            elif (angle == 45) or (angle == 225):
                q = G[i+1,j+1]### FILL IN HERE
                r = G[i-1,j-1]### FILL IN HERE
            # Vertical direction (90, 270 degrees)
            elif (angle == 90) or (angle == 270):
                q = G[i,j+1]### FILL IN HERE
                r = G[i,j-1]### FILL IN HERE
            # Diagonal direction (135, 315 degrees)
            elif (angle == 135) or (angle == 315):
                q = G[i+1,j-1]### FILL IN HERE
                r = G[i-1,j+1]### FILL IN HERE


            if G[i,j] > q and G[i,j] > r: ### FILL IN CONDITIONAL HERE:
                out[i, j] = G[i, j]
            # else: out[i, j] remains 0 as initialized for non-maxima
            else: out[i,j] = 0.0

    return out
  

def shift_nms(imgA, imgB, max_shift):

    grayA = color.rgb2gray(imgA)
    grayB = color.rgb2gray(imgB)

    Ga, thetaA = gradient(grayA)
    Gb, thetaB = gradient(grayB)

    edgesA = non_maximum_suppression(Ga, thetaA)
    edgesB = non_maximum_suppression(Gb, thetaB)

    H,W = edgesA.shape

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

            A = edgesA[A_y1:A_y2, A_x1:A_x2]
            B = edgesB[B_y1:B_y2, B_x1:B_x2]

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

    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]

    return shifted
 
 
# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
 
def _load(folder, num):
    path = os.path.join(folder, "face_0"+num+".png")
    return np.array(Image.open(path)).astype(float) / 255.0
 
 
def _list_images(folder):
    return sorted(f for f in os.listdir(folder))
 
 
def _aligned_folder():
    return os.path.join(DATASET_ROOT, "aligned")
 
 
def _difficulty_folder(difficulty):
    return os.path.join(DATASET_ROOT, difficulty)
 
 
 

def run_alignment_experiment(diff, num1, num2, low_sd, high_sd, weight,
                              max_shift=None, method="both", out_path="hybrid.png"):

    assert method in ("pixel", "nms", "both"), f"method must be 'pixel', 'nms', or 'both', got {method!r}"
    use_pixel = method in ("pixel", "both")
    use_nms = method in ("nms", "both")
 
    if max_shift is None:
        max_shift = DIFFICULTIES[diff]
 
    imgA = _load(_aligned_folder(), num1)
    imgB = _load(_difficulty_folder(diff), num2)
 
    dx_pixel = dy_pixel = dx_nms = dy_nms = None
    lowA = highB = None
    panels = []
 
    if use_pixel:
        dx_pixel, dy_pixel = shift_pixel(imgA, imgB, max_shift)
        alignB_pixel = shift_image(imgB, -dx_pixel, -dy_pixel)
        lowA, highB, hybrid_pixel = hybrid_image(imgA, low_sd, alignB_pixel, high_sd, weight)
        hybrid_pixel_small = subsample(hybrid_pixel, sigma=5)
        panels += [
            (hybrid_pixel, "hybrid (pixel-aligned)"),
            (hybrid_pixel_small, "hybrid (pixel) small"),
        ]
 
    if use_nms:
        dx_nms, dy_nms = shift_nms(imgA, imgB, max_shift)
        alignB_nms = shift_image(imgB, -dx_nms, -dy_nms)
        lowA2, highB2, hybrid_nms = hybrid_image(imgA, low_sd, alignB_nms, high_sd, weight)
        if lowA is None:  # method="nms" only -- pixel branch didn't run
            lowA, highB = lowA2, highB2
        hybrid_nms_small = subsample(hybrid_nms, sigma=5)
        panels += [
            (hybrid_nms, "hybrid (NMS-edge-aligned)"),
            (hybrid_nms_small, "hybrid (NMS) small"),
        ]
 
    panels = [(lowA, "lowA"), (highB, "highB")] + panels
 
    small_panel_shrink = 0.65
 
    inches_per_pixel = 3.0 / max(img.shape[1] for img, _ in panels)
    pad_inch = 0.35
    title_pad_inch = 0.3
 
    def panel_scale(title):
        return inches_per_pixel * (small_panel_shrink if "small" in title else 1.0)
 
    panel_w_inch = [img.shape[1] * panel_scale(title) for img, title in panels]
    panel_h_inch = [img.shape[0] * panel_scale(title) for img, title in panels]
 
    fig_w = sum(panel_w_inch) + pad_inch * (len(panels) + 1)
    fig_h = max(panel_h_inch) + title_pad_inch + 0.6  # + room for suptitle
 
    fig = plt.figure(figsize=(fig_w, fig_h))
 
    x = pad_inch
    for (img, title), w_inch, h_inch in zip(panels, panel_w_inch, panel_h_inch):
        left = x / fig_w
        width = w_inch / fig_w
        height = h_inch / fig_h
        bottom = 0.08  # baseline-align all panels along the bottom
 
        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
 
        x += w_inch + pad_inch
 
    shift_parts = []
    if use_pixel:
        shift_parts.append(f"pixel dx,dy=({dx_pixel},{dy_pixel})")
    if use_nms:
        shift_parts.append(f"nms dx,dy=({dx_nms},{dy_nms})")
    fig.suptitle(
        f"{diff}/{num1, num2}  |  " + "  ".join(shift_parts),
        fontsize=10,
    )
    fig.savefig(out_path, bbox_inches="tight")
 
    return {
        "difficulty": diff,
        "method": method,
        "num1": num1,
        "num2": num2,
        "dx_pixel": dx_pixel, "dy_pixel": dy_pixel,
        "dx_nms": dx_nms, "dy_nms": dy_nms,
        "out_path": out_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid images + Alignment")
    parser.add_argument("--diff", choices=DIFFICULTIES.keys())
    parser.add_argument("--num1")
    parser.add_argument("--num2")
    parser.add_argument("--method", choices=["pixel", "nms", "both"], default="both")
    parser.add_argument("--low-sd", type=float, default=5.0)
    parser.add_argument("--high-sd", type=float, default=5.0)
    parser.add_argument("--weight", type=float, default=1.0)
    parser.add_argument("--max-shift", type=int, default=None)
    parser.add_argument("--out", default="hybrid.png")
    args = parser.parse_args()
 
    res = run_alignment_experiment(
        diff=args.diff, num1=args.num1, num2=args.num2, method=args.method,
        low_sd=args.low_sd, high_sd=args.high_sd, weight=args.weight,
        max_shift=args.max_shift, out_path=args.out)
    print(res)
 