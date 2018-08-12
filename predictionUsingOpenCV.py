import cv2
import numpy as np
from cv2 import resize 


cap = cv2.VideoCapture('sample8.mpg')


while(1):

    # Take each frame
    _, frame = cap.read()
    frame = resize(frame, (200,200))
    frame = cv2.medianBlur(frame, 3)#ksize[, dst])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame = cv2.fastNlMeansDenoisingColored(frame,None,10,10,7,21)
    lower_red = np.array([30,150,50])
    upper_red = np.array([255,255,180])
    
    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    laplacian = cv2.Laplacian(frame,cv2.CV_64F)
    sobelx = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
    canny = cv2.Canny(frame,100,200)

    # canny = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)
    _,canny = cv2.threshold(canny,200,255,cv2.THRESH_BINARY)



    indices = np.where(canny == [255])
    # print(indices)
    
    coordinates = zip(indices[0], indices[1])
    # print(coordinates)
    sumx=0
    sumy=0
    ctr=0
    for x in coordinates:
        if x[1]==1:
            continue
        ctr+=1
        sumx+=x[0]
        sumy+=x[1]
    if ctr!=0:
        sumx= int(sumx/ctr)
        sumy= int(sumy/ctr)
        print(str(sumx)+' '+str(sumy))
        # cv2.circle(canny,(sumx,sumy), 100, (0,0,255), -1)
        cv2.circle(frame,(sumy,sumx), 5, (0,0,255), -1)

    
    cv2.imshow('Original',frame)
    # cv2.imshow('Mask',mask)
    # cv2.imshow('laplacian',laplacian)
    # # cv2.imshow('sobelx',sobelx)
    # cv2.imshow('sobely',sobely)
    cv2.imshow('canny',canny)
    # cv2.imshow('dst',dst)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()