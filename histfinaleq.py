import cv2
import numpy as np
img=cv2.imread("lena5.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("grascale image",img_gray)
print("\n 2d matrix",img_gray)
im1=img_gray
imggray_size=im1.shape
m=imggray_size[0]
n=imggray_size[1]
print(m,n)
mn = m * n
im2 = [pixel for row in im1 for pixel in row]
im3=sorted(im2)
min_val=im3[0]
max_val=im3[len(im3)-1]
unique_pixels = []
for px in im2:
    if px not in unique_pixels:
        unique_pixels.append(px)
unique_pixels.sort()
count = []
for val in unique_pixels:
    c = 0
    for px in im2:
        if px == val:
            c += 1
    count.append(c)
cs = []
s = 0
for c in count:
    s += c
    cs.append(s)
hv = []
cdf_min = cs[0]
for cdf_v in cs:
    h_v = round(((cdf_v - cdf_min) / (mn - cdf_min)) * (255))
    hv.append(h_v)
eqdict={}
for i in range(len(unique_pixels)):
    eqdict[unique_pixels[i]]=hv[i]
eq=[]
for j in range(len(im1)):
    row=[]
    for k in range(len(im1[j])):
        row.append(eqdict[im1[j][k]])
    eq.append(row)
'''print("\nUnique pixel values:")
print(unique_pixels)
print("\nPixel counts:")
print(count)
print("\nCumulative sum (CDF):")
print(cs)
print("\nEqualized values h(v):")
print(hv)
print(eqdict)'''
print("\n Equilized Image")
eq_np = np.array(eq, dtype=np.uint8)
print(eq_np)
cv2.imshow("equalized image",eq_np)
cv2.waitKey(0)
cv2.destroyAllWindows()