import numpy as np
import cv2

img= cv2.imread('165.jpg',cv2.IMREAD_COLOR)

img[100:150,100:150]=[0,0,0]


box=img[200:250,200:250]

img[0:50,0:50]=box

cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()