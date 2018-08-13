from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras
import numpy as np
import cv2

model=keras.models.load_model('12augtrainedmodel.h5')

x=np.load('test.npy')
x=x.reshape(1, 100, 100, 1)
x=x/255

# print(model.predict(x))

cap = cv2.VideoCapture('stuff/sample8.mpg')
while(1):

    # Take each frame
    _, frame = cap.read()
    print(type(frame))
    frame = cv2.resize(frame, (100,100))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.medianBlur(frame, 3)#ksize[, dst])
    frame=frame.reshape(1, 100, 100, 1)
    frame/=255
    print(model.predict(frame))   
    #take output of above statement and make rectangle