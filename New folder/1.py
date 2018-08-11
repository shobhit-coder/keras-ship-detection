import cv2
import numpy as np
img = cv2.imread('165.jpg',cv2.IMREAD_COLOR)
indices = np.where(img == [255])
for x in indices:
    print(x)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()