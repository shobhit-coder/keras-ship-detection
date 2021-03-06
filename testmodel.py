import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras
import numpy as np
import cv2

model=keras.models.load_model('23septtrainedmodel1.h5')

# x=np.load('test.npy')
# x=x.reshape(1, 100, 100, 1)
# x=x/255

# print(model.predict(x))

cap = cv2.VideoCapture('sample8.mpg')
while(1):

    # Take each frame
    _, frame = cap.read()

    # print(type(frame))
    # frame = cv2.resize(frame, (400,400))
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame = cv2.medianBlur(frame, 3)#ksize[, dst])
    # frame=frame.reshape(1, 400, 400, 1)
    # frame=np.true_divide(frame,255)
    # print(model.predict(frame))   
    #take output of above statement and make rectangle

    print(type(frame))
    
    frame = cv2.resize(frame, (400,400))

    fr1 = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame, 3)#ksize[, dst])
    # #cv2.imshow('Original',frame)
    cv2.imshow('Original',frame)
    frame = frame.reshape(1, 400, 400, 1)
    frame=frame/255
    coord=model.predict(frame)
    print(coord)
    topleftx=int(coord[0][0])
    toplefty=int(coord[0][1])
    bottomrightx=int(coord[0][2])
    bottomrighty=int(coord[0][3])
    
    cv2.rectangle(fr1,(topleftx,toplefty) ,(bottomrightx,bottomrighty),100,5)

    cv2.imshow('frame',fr1)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()

