from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np
img_rows=100
img_cols=100
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

ship_model.compile(loss=keras.losses.mean_squared_error,
              optimizer='adam',
              metrics=['accuracy'])


ship_model.fit(x, y,
          batch_size=16,
          epochs=20,
          validation_split = 0.2)

ship_model.save('12augtrainedmodel.h5')