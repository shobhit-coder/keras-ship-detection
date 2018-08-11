from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import keras
import numpy as np

model=keras.models.load_model('trainedmodel3pooled.h5')

x=np.load('test.npy')
x=x.reshape(1, 100, 100, 1)
x=x/255

print(model.predict(x))