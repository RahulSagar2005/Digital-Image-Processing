import cv2
import os
from collections import Counter
import heapq

# -----------------------------
# STEP 1: Load Image
# -----------------------------
os.makedirs("outputs/task3", exist_ok=True)

# Load image in grayscale (pixel values 0–255)
img = cv2.imread("lena5.jpg", cv2.IMREAD_GRAYSCALE)

# Flatten image into a single list of pixel values
flat = img.flatten()

# Count frequency of each pixel value
freq = Counter(flat)

# -----------------------------
# STEP 2: Shannon–Fano
# -----------------------------
def shannon_fano(symbols):
    """Recursive Shannon–Fano coding"""
    if len(symbols) <= 1:
        if len(symbols) == 1:
            return {symbols[0][0]: "0"}
        else:
            return {}

    # Find total frequency
    total = 0
    for _, f in symbols:
        total += f

    # Find split point (where half frequency is reached)
    acc = 0
    split = 0
    for i, (sym, f) in enumerate(symbols):
        acc += f
        if acc >= total / 2:
            split = i
            break

    # Divide into left and right parts
    left = shannon_fano(symbols[:split + 1])
    right = shannon_fano(symbols[split + 1:])

    # Add "0" in front of left codes
    for k in left:
        left[k] = "0" + left[k]
    # Add "1" in front of right codes
    for k in right:
        right[k] = "1" + right[k]

    # Merge and return
    codes = {}
    codes.update(left)
    codes.update(right)
    return codes


# Sort symbols by frequency (high to low)
symbols_sorted = sorted(freq.items(), key=lambda x: -x[1])
sf_codes = shannon_fano(symbols_sorted)

# ✅ Build encoded string efficiently
sf_bits_list = []
for p in flat:
    sf_bits_list.append(sf_codes[p])
sf_encoded = ''.join(sf_bits_list)

# Save encoded Shannon–Fano result (⚠ very large file!)
with open("outputs/task3/shannon_fano.txt", "w") as f:
    f.write(sf_encoded)


# -----------------------------
# STEP 3: Huffman
# -----------------------------
class Node:
    def __init__(self, sym, freq):
        self.sym = sym
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def huffman(freq):
    """Huffman coding"""
    heap = []
    for s, f in freq.items():
        heap.append(Node(s, f))
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)

        parent = Node(None, n1.freq + n2.freq)
        parent.left = n1
        parent.right = n2

        heapq.heappush(heap, parent)

    root = heap[0]
    codes = {}

    def gen(node, code=""):
        if node:
            if node.sym is not None:
                codes[node.sym] = code
            gen(node.left, code + "0")
            gen(node.right, code + "1")

    gen(root)
    return codes


hf_codes = huffman(freq)

# ✅ Build encoded string efficiently
hf_bits_list = []
for p in flat:
    hf_bits_list.append(hf_codes[p])
hf_encoded = ''.join(hf_bits_list)

# Save encoded Huffman result (⚠ very large file!)
with open("outputs/task3/huffman.txt", "w") as f:
    f.write(hf_encoded)


# -----------------------------
# STEP 4: Compression Statistics
# -----------------------------
original_bits = len(flat) * 8
sf_bits = len(sf_encoded)
hf_bits = len(hf_encoded)

print("✅ Encoding done! Check outputs/task3/\n")

print("Original bits:", original_bits)
print("Shannon-Fano bits:", sf_bits)
print("Huffman bits:", hf_bits)

print("Shannon-Fano Compression Ratio:", round(original_bits / sf_bits, 3))
print("Huffman Compression Ratio:", round(original_bits / hf_bits, 3))
