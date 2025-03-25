import cv2
import numpy as np
from scipy.ndimage import label

def analyze_blur(image_path, block_size=32, variance_threshold=100, portrait_threshold_percent=15):
    # Load the image and convert to grayscale
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to read.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Calculate the number of blocks, adjusting for any remaining pixels
    rows = (h + block_size - 1) // block_size
    cols = (w + block_size - 1) // block_size

    # Initialize the blur mask
    blur_mask = np.zeros((rows, cols), dtype=np.uint8)

    # Process each block to determine if it's blurry
    for i in range(rows):
        for j in range(cols):
            y_start = i * block_size
            y_end = min(y_start + block_size, h)
            x_start = j * block_size
            x_end = min(x_start + block_size, w)
            block = gray[y_start:y_end, x_start:x_end]
            if block.size == 0:
                continue
            laplacian = cv2.Laplacian(block, cv2.CV_64F)
            variance = laplacian.var()
            if variance < variance_threshold:
                blur_mask[i, j] = 1

    # Calculate blur ratio
    total_blocks = rows * cols
    blur_ratio = (np.sum(blur_mask) / total_blocks) * 100 if total_blocks else 0.0

    # Generate blur map
    blur_map = np.zeros_like(gray)
    for i in range(rows):
        for j in range(cols):
            y_start = i * block_size
            y_end = min(y_start + block_size, h)
            x_start = j * block_size
            x_end = min(x_start + block_size, w)
            if blur_mask[i, j]:
                blur_map[y_start:y_end, x_start:x_end] = 255

    # Classify the blur
    non_blur_mask = 1 - blur_mask
    labeled, num_features = label(non_blur_mask)
    max_non_blur_blocks = 0
    if num_features > 0:
        sizes = np.bincount(labeled.ravel())
        if len(sizes) > 1:
            max_non_blur_blocks = np.max(sizes[1:])

    portrait_threshold = (portrait_threshold_percent / 100) * total_blocks
    category = "Unknown"
    if max_non_blur_blocks >= portrait_threshold and blur_ratio >= 30:
        category = "Portrait-like"
    else:
        if blur_ratio < 20:
            category = "Clear"
        elif blur_ratio < 50:
            category = "Slightly Blurry"
        elif blur_ratio < 80:
            category = "Partially Blurry"
        else:
            category = "Very Blurry"

    # Overall blur assessment
    overall_blur = blur_ratio >= 50

    return {
        'blur_ratio': blur_ratio,
        'category': category,
        'blur_map': blur_map,
        'overall_blur': overall_blur
    }