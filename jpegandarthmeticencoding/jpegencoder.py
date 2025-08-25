import numpy as np
import cv2
from scipy.fftpack import dct, idct
from collections import Counter
import heapq
import matplotlib.pyplot as plt

# =========================
# Quantization Table
# =========================
Q50_LUMA = np.array([
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,22,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
], dtype=np.int32)

def scale_quant_table(Q50, quality):
    quality = max(1, min(100, int(quality)))
    if quality == 50:
        QX = Q50
    elif quality > 50:
        QX = np.round(Q50 * ((100 - quality) / 50.0))
    else:
        QX = np.round(Q50 * (50.0 / quality))
    QX[QX < 1] = 1
    QX[QX > 255] = 255
    return QX.astype(np.int32)

# =========================
# DCT / IDCT Helpers
# =========================
def pad_to_block(img, block=8):
    h, w = img.shape
    nh = (h + block - 1) // block * block
    nw = (w + block - 1) // block * block
    pad_img = np.zeros((nh, nw), dtype=img.dtype)
    pad_img[:h, :w] = img
    if nh > h:
        pad_img[h:nh, :w] = img[h-1:h, :]
    if nw > w:
        pad_img[:, w:nw] = pad_img[:, w-1:w]
    return pad_img, h, w

def unpad_from_block(img, h, w):
    return img[:h, :w]

def block_dct_quant(channel, Q):
    img = channel.astype(np.float32) - 128
    pad, h, w = pad_to_block(img)
    out = np.zeros_like(pad, dtype=np.int32)
    for y in range(0, pad.shape[0], 8):
        for x in range(0, pad.shape[1], 8):
            blk = pad[y:y+8, x:x+8]
            d = dct(dct(blk.T, norm='ortho').T, norm='ortho')
            out[y:y+8, x:x+8] = np.round(d / Q).astype(np.int32)
    return out, (h, w)

def block_idct_dequant(qcoeff, Q, shape_hw):
    h, w = shape_hw
    out = np.zeros_like(qcoeff, dtype=np.float32)
    for y in range(0, qcoeff.shape[0], 8):
        for x in range(0, qcoeff.shape[1], 8):
            blk = qcoeff[y:y+8, x:x+8].astype(np.float32) * Q
            s = idct(idct(blk.T, norm='ortho').T, norm='ortho')
            out[y:y+8, x:x+8] = s
    img = np.clip(out + 128, 0, 255)
    return unpad_from_block(img, h, w).astype(np.uint8)

# =========================
# Zigzag + Huffman helpers
# =========================
ZIGZAG_IDX = np.array([
 [0, 1, 5, 6,14,15,27,28],
 [2, 4, 7,13,16,26,29,42],
 [3, 8,12,17,25,30,41,43],
 [9,11,18,24,31,40,44,53],
 [10,19,23,32,39,45,52,54],
 [20,22,33,38,46,51,55,60],
 [21,34,37,47,50,56,59,61],
 [35,36,48,49,57,58,62,63]
], dtype=np.int32).flatten()

def zigzag(block8x8):
    return block8x8.flatten()[ZIGZAG_IDX]

def inv_zigzag(vec64):
    out = np.zeros((8,8), dtype=vec64.dtype)
    flat = out.flatten()
    flat[ZIGZAG_IDX] = vec64
    return out

def category(val):
    v = int(abs(val))
    if v == 0: return 0
    return int(np.floor(np.log2(v))) + 1

def value_to_bits(val, size):
    if size == 0: return ""
    if val >= 0:
        return format(val, '0{}b'.format(size))
    return format((1 << size) - 1 + val, '0{}b'.format(size))

class BitWriter:
    def __init__(self): self.bits = []
    def write_bits(self, bitstring):
        if bitstring: self.bits.append(bitstring)
    def get_bytes(self):
        bitstr = "".join(self.bits)
        pad = (8 - (len(bitstr) % 8)) % 8
        bitstr += "0"*pad
        data = bytearray()
        for i in range(0,len(bitstr),8):
            data.append(int(bitstr[i:i+8],2))
        return bytes(data), pad

EOB = 0x00
ZRL = 0xF0

def rle_ac(vec63):
    out=[]
    run=0
    for v in vec63:
        if v==0:
            run+=1
            if run==16: out.append((15,0,0)); run=0
        else:
            s=category(v)
            out.append((run,s,int(v)))
            run=0
    out.append((0,0,0))
    return out

def build_huffman_code(freqs):
    if not freqs: return {}
    heap = []
    counter = 0
    for sym, f in freqs.items():
        heap.append([f, counter, (sym, "")])
        counter += 1
    heapq.heapify(heap)
    if len(heap) == 1:
        sym = heap[0][2][0]
        return {sym: "0"}
    while len(heap) > 1:
        f1, _, t1 = heapq.heappop(heap)
        f2, _, t2 = heapq.heappop(heap)
        t1 = assign_prefix(t1, '0')
        t2 = assign_prefix(t2, '1')
        merged = [t1, t2]
        heapq.heappush(heap, [f1+f2, counter, merged])
        counter += 1
    codes = {}
    flatten_codes(heap[0][2], codes)
    return codes

def assign_prefix(tree, bit):
    if isinstance(tree, tuple):
        sym, code = tree
        return (sym, bit + code)
    else:
        left, right = tree
        return [assign_prefix(left, bit), assign_prefix(right, bit)]

def flatten_codes(tree, outdict):
    if isinstance(tree, tuple):
        sym, code = tree
        outdict[sym] = code
    else:
        left, right = tree
        flatten_codes(left, outdict)
        flatten_codes(right, outdict)

def build_symbol_stream(blocks_zz):
    dc_syms, dc_extras, ac_syms, ac_extras = [],[],[],[]
    prev_dc=0
    for coeffs in blocks_zz:
        dc = int(coeffs[0])
        diff = dc - prev_dc
        prev_dc = dc
        sz = category(diff)
        dc_syms.append(sz)
        dc_extras.append(value_to_bits(diff,sz))
        ac_rle = rle_ac(coeffs[1:])
        for run, sz_ac, val in ac_rle:
            if run==0 and sz_ac==0:
                ac_syms.append(EOB); ac_extras.append("")
            elif run==15 and sz_ac==0:
                ac_syms.append(ZRL); ac_extras.append("")
            else:
                ac_syms.append((run<<4)|sz_ac)
                ac_extras.append(value_to_bits(val,sz_ac))
    return dc_syms, dc_extras, ac_syms, ac_extras

def encode_channel(qcoeff, Q):
    blocks_zz = [zigzag(qcoeff[y:y+8,x:x+8]) for y in range(0,qcoeff.shape[0],8) for x in range(0,qcoeff.shape[1],8)]
    dc_syms, dc_extra, ac_syms, ac_extra = build_symbol_stream(blocks_zz)
    dc_codes = build_huffman_code(Counter(dc_syms))
    ac_codes = build_huffman_code(Counter(ac_syms))
    bw = BitWriter()
    for s,ex in zip(dc_syms,dc_extra):
        bw.write_bits(dc_codes[s])
        bw.write_bits(ex)
    for s,ex in zip(ac_syms,ac_extra):
        bw.write_bits(ac_codes[s])
        bw.write_bits(ex)
    data,pad = bw.get_bytes()
    return {"qcoeff":qcoeff,"Q":Q,"data":data,"pad":pad,"num_blocks":len(blocks_zz)}

# =========================
# JPEG Encode/Decode
# =========================
def encode_grayscale_full(img_gray, quality=50):
    Q = scale_quant_table(Q50_LUMA, quality)
    qcoeff, shape_hw = block_dct_quant(img_gray, Q)
    enc = encode_channel(qcoeff, Q)
    enc["orig_shape"]=img_gray.shape
    enc["compressed_vis"] = block_idct_dequant(qcoeff, Q, qcoeff.shape)
    enc["compression_ratio"] = img_gray.size*8 / (len(enc["data"])*8)
    return enc

def decode_grayscale_full(enc):
    return block_idct_dequant(enc["qcoeff"],enc["Q"],enc["qcoeff"].shape)

def encode_color_full(img_rgb, quality=50):
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycrcb[...,0], ycrcb[...,1], ycrcb[...,2]
    Cr_sub = cv2.resize(Cr,(Cr.shape[1]//2,Cr.shape[0]//2),interpolation=cv2.INTER_AREA)
    Cb_sub = cv2.resize(Cb,(Cb.shape[1]//2, Cb.shape[0]//2),interpolation=cv2.INTER_AREA)
    encY = encode_grayscale_full(Y,quality)
    encCr = encode_grayscale_full(Cr_sub,quality)
    encCb = encode_grayscale_full(Cb_sub,quality)
    return {"Y":encY,"Cr":encCr,"Cb":encCb,"subsample_420":True,"orig_shape":img_rgb.shape}

def decode_color_full(enc):
    decY = decode_grayscale_full(enc["Y"])
    decCr = decode_grayscale_full(enc["Cr"])
    decCb = decode_grayscale_full(enc["Cb"])
    H,W = decY.shape
    decCr = cv2.resize(decCr,(W,H),interpolation=cv2.INTER_LINEAR)
    decCb = cv2.resize(decCb,(W,H),interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(np.stack([decY,decCr,decCb],axis=-1).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

def compressed_color_vis(enc):
    decY = block_idct_dequant(enc["Y"]["qcoeff"], enc["Y"]["Q"], enc["Y"]["qcoeff"].shape)
    decCr = block_idct_dequant(enc["Cr"]["qcoeff"], enc["Cr"]["Q"], enc["Cr"]["qcoeff"].shape)
    decCb = block_idct_dequant(enc["Cb"]["qcoeff"], enc["Cb"]["Q"], enc["Cb"]["qcoeff"].shape)
    H, W = decY.shape
    decCr = cv2.resize(decCr, (W, H), interpolation=cv2.INTER_LINEAR)
    decCb = cv2.resize(decCb, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(np.stack([decY, decCr, decCb], axis=-1).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# =========================
# PSNR
# =========================
def compute_psnr(orig, recon):
    mse = np.mean((orig.astype(np.float32) - recon.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

# =========================
# Utility
# =========================
def size_in_bytes(arr):
    return arr.size if arr.ndim==2 else arr.size

def compressed_size_in_bytes(enc, color=False):
    if color:
        return (len(enc["Y"]["data"]) + len(enc["Cr"]["data"]) + len(enc["Cb"]["data"]))
    else:
        return len(enc["data"])

# =========================
# Main
# =========================
if __name__=="__main__":
    img_path = "jpegandarthmeticencoding/lena5.jpg"
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None: raise FileNotFoundError(f"{img_path} not found")
    color = cv2.imread(img_path, cv2.IMREAD_COLOR)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    quality = 50

    # Grayscale
    enc_g = encode_grayscale_full(gray, quality=quality)
    dec_g = decode_grayscale_full(enc_g)
    psnr_g = compute_psnr(gray, dec_g)

    orig_size_g = gray.size / 1024
    comp_size_g = compressed_size_in_bytes(enc_g) / 1024
    recon_size_g = dec_g.size / 1024

    # Color
    enc_c = encode_color_full(color, quality=quality)
    dec_c = decode_color_full(enc_c)
    raw_bits = color.size * 8
    comp_bits = len(enc_c["Y"]["data"])*8 + len(enc_c["Cr"]["data"])*8 + len(enc_c["Cb"]["data"])*8
    cr_color = raw_bits / comp_bits
    psnr_c = compute_psnr(color, dec_c)

    orig_size_c = color.size / 1024
    comp_size_c = compressed_size_in_bytes(enc_c, color=True) / 1024
    recon_size_c = dec_c.size / 1024

    # ======================
    # Show Grayscale
    # ======================
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.imshow(gray, cmap="gray")
    plt.title(f"Original\n{gray.shape[0]}x{gray.shape[1]}\n{orig_size_g:.1f} KB")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.imshow(enc_g["compressed_vis"], cmap="gray")
    plt.title(f"Compressed\n{gray.shape[0]}x{gray.shape[1]}\n{comp_size_g:.1f} KB\nCR≈{enc_g['compression_ratio']:.1f}×")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(dec_g, cmap="gray")
    plt.title(f"Reconstructed\n{gray.shape[0]}x{gray.shape[1]}\n{recon_size_g:.1f} KB\nPSNR {psnr_g:.1f} dB")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # ======================
    # Show Color
    # ======================
    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.imshow(color)
    plt.title(f"Original\n{color.shape[0]}x{color.shape[1]}x{color.shape[2]}\n{orig_size_c:.1f} KB")
    plt.axis("off")

    plt.subplot(1,3,2)
    comp_vis = compressed_color_vis(enc_c)
    plt.imshow(comp_vis)
    plt.title(f"Compressed (Color)\n{color.shape[0]}x{color.shape[1]}x3\n{comp_size_c:.1f} KB\nCR≈{cr_color:.1f}×")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(dec_c)
    plt.title(f"Reconstructed\n{color.shape[0]}x{color.shape[1]}x3\n{recon_size_c:.1f} KB\nPSNR {psnr_c:.1f} dB")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
