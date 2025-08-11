import cv2
import numpy as np
img=cv2.imread("lena5.jpg")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
rows,cols=img_gray.shape
bit_plane=[]
for i in range(8):
    plane=np.zeros((rows,cols),dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            pixel=int(img_gray[r,c])
            bit=(pixel//(2**i))%2
            plane[r,c]=255 if bit==1 else 0
    bit_plane.append(plane)
while True:
    cv2.imshow("colouredimage",img)
    cv2.imshow("grayscaleimage",img_gray)
    for j in range(8):
        cv2.imshow(f"bit plane {j}",bit_plane[j])
    if cv2.waitKey(1) & 0xFF==ord("x"):
        break
cv2.destroyAllWindows()