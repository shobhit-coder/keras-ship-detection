import cv2
import xml.etree.ElementTree as ET
import numpy as np

imarray=np.zeros((1,100,100))
im = cv2.imread('shiptest.jpg')
im=im[:,:,0]
imarray[0]=im

np.save('test.npy',imarray)