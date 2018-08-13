from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
import tensorflow as tf

from collections import namedtuple
# import numpy as np
# import cv2


def bb_intersection_over_union(boxA1, boxB1):
    with tf.Session() as sess:
        print("hello\n\n---------\n\n\n")
        print(type(sess.run(boxA1)))
    # x1=boxA1[0]
    # y1=boxA1[1]
    # w1=boxA1[2]
    # h1=boxA1[3]

    # x2=boxB1[0]
    # y2=boxB1[1]
    # w2=boxB1[2]
    # h2=boxB1[3]


    # boxA=[0,0,0,0]
    # boxA[0]=x1-w1/2
    # boxA[1]=y1-h1/2
    # boxA[2]=x1+w1/2
    # boxA[3]=y1+h1/2

    # boxB=[0,0,0,0]
    # boxB[0]=x2-w2/2
    # boxB[1]=y2-h2/2
    # boxB[2]=x2+w2/2
    # boxB[3]=y2+h2/2

    # print(type(boxA[0]))
    # print(type(boxB[0]))
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
    return iou




img_rows=400
img_cols=400
num_images=730
x=np.load('new_innpy1fixed.npy')
# print(x.shape)
# x=x.reshape(num_images, img_rows, img_cols, 1)
x=x/255
y=np.load('new_outnpy1fixed.npy')

# Your Code Here
ship_model = Sequential()
ship_model.add(Conv2D(20,kernel_size=(2,2),activation='relu',input_shape=((img_rows, img_cols, 1))))
ship_model.add(MaxPooling2D(pool_size = (2,2)))
ship_model.add(Conv2D(20,kernel_size=(2,2),activation='relu'))
ship_model.add(MaxPooling2D(pool_size = (2,2)))
# ship_model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
# ship_model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
# ship_model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
# ship_model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
# ship_model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
# ship_model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
# ship_model.add(Conv2D(100,kernel_size=(3,3),activation='relu'))
ship_model.add(Flatten())
# ship_model.add(Dense(100,activation='relu'))
ship_model.add(Dense(50,activation='relu'))
# ship_model.add(Dense(60,activation='relu'))
# ship_model.add(Dense(50,activation='relu'))
ship_model.add(Dense(4,activation='relu'))

import keras

ship_model.compile(loss=bb_intersection_over_union,
              optimizer='adam',
              metrics=['accuracy'])


ship_model.fit(x, y,
          batch_size=16,
          epochs=20,
          validation_split = 0.2)

ship_model.save('13augtrainedmodel.h5')
