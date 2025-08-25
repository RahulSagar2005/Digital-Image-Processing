import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Arithmetic Coding Functions
# ----------------------------
def arithmetic_encode(block, prob_table):
    low, high = 0.0, 1.0
    for pixel in block:
        range_ = high - low
        high = low + range_ * prob_table[pixel + 1]
        low = low + range_ * prob_table[pixel]
    return (low + high) / 2

def build_probability_table(img):
    hist = np.bincount(img.flatten(), minlength=256)
    pdf = hist / np.sum(hist)
    cdf = np.cumsum(pdf)
    cdf = np.insert(cdf, 0, 0.0)  # prepend 0
    return cdf

# ----------------------------
# Main Script
# ----------------------------

# Load image in grayscale
img = cv2.imread("jpegandarthmeticencoding/lena5.jpg", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise ValueError("Image not loaded. Check the path.")

block_size = 8  # block size
rows, cols = img.shape
prob_table = build_probability_table(img)

# Encode image in blocks
encoded_blocks = []
for r in range(0, rows, block_size):
    for c in range(0, cols, block_size):
        block = img[r:r+block_size, c:c+block_size].flatten()
        code = arithmetic_encode(block, prob_table)
        encoded_blocks.append(code)

encoded_blocks = np.array(encoded_blocks)

# ----------------------------
# Visualization
# ----------------------------

# Original Image
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original Lena Image")
plt.axis("off")

# Middle Image: Block-coded map
codes_img = encoded_blocks.reshape(rows//block_size, cols//block_size)
codes_img_norm = (codes_img - codes_img.min()) / (codes_img.max() - codes_img.min())
codes_img_norm = (codes_img_norm * 255).astype(np.uint8)

# Upscale to Lena size
codes_img_upscaled = cv2.resize(codes_img_norm, (cols, rows), interpolation=cv2.INTER_CUBIC)

plt.subplot(1, 3, 2)
plt.imshow(codes_img_upscaled, cmap='gray')
plt.title("Arithmetic Coded Block Representation")
plt.axis("off")

# Histogram of Encoded Values
plt.subplot(1, 3, 3)
plt.hist(encoded_blocks, bins=50, color="blue", alpha=0.7)
plt.title("Distribution of Encoded Blocks")
plt.xlabel("Encoded value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
