# This file is used to detect the ship in a video.
import numpy as np
import cv2

ship_cascade = cv2.CascadeClassifier('cascade.xml')


cap = cv2.VideoCapture('sample8.mpg')
xa=[]
ya=[]
ctr=0
medx=0
medy=0
numx=0
numy=0
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ships = ship_cascade.detectMultiScale(gray,20,20)
    
    # print('average:'+str(medx)+' '+str(medy))
    # ctr+=1
    # if ctr == 60:
    #     xa.sort()
    #     ya.sort()
    #     numx=len(xa)
    #     numy=len(ya)
    #     if numx%2==0:
    #         medx=(xa[int(numx/2)] + xa[int(numx/2 +1)] )/2
    #     else:
    #         medx=  xa[int((numx+1)/2)]
    #     if numy%2==0:
    #         medy=( ya[int(numy/2)] + ya[int(numx/2 +1)] )/2 
    #     else:
    #         medy=ya[int((numy+1)/2)]
    #     xa[:]=[]
    #     ya[:]=[]
    
    
    for (x,y,w,h) in ships:
        
        xa.append(x)
        ya.append(y) 
        
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        

        print('x:'+str(x)+'y:'+str(y))
        
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        
    # cv2.rectangle(img,(0,0),(20,20),(0,0,0),2)
    # cv2.rectangle(img,(int(medx),int(medy)),(int(medx+20),int(medy+20)),(0,0,0),2)
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()