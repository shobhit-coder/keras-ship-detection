import cv2
import numpy as np

img1=cv2.imread('sce1.jpg')
img2=cv2.imread('google.jpg')

# img3 = img1+img2

# img=cv2.add(img1,img2)

rows,col,channels=img2.shape
roi=img1[0:rows,0:col]

img2toGray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2toGray,220,255,cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

img1_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
img2_fg = cv2.bitwise_and(img2,img2,mask=mask)

cv2.imshow('img',img1_bg)
cv2.imshow('img3',img2_fg)

final_roi = cv2.add(img1_bg,img2_fg)

img1[0:rows,0:col]=final_roi

cv2.imshow('imgFinal',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()