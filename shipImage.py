import cv2
import numpy as np

ship_cascade = cv2.CascadeClassifier('cascade.xml')
img = cv2.imread('0.jpg',cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ships = ship_cascade.detectMultiScale(gray,20,20)
print(ships)  
for (x,y,w,h) in ships:

    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    print('x:'+str(x)+'y:'+str(y))
       
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
        

cv2.imshow('img',img)
cv2.waitKey(0)

cv2.destroyAllWindows()