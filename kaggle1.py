# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.python import keras
import os
import cv2
import numpy as np

img_rows, img_cols = 28, 28
num_classes = 10

def prep_data(raw, train_size, val_size):
    y = raw[:, 0]
    out_y = keras.utils.to_categorical(y, num_classes)
    
    x = raw[:,1:]
    num_images = raw.shape[0]
    out_x = x.reshape(num_images, img_rows, img_cols, 1)
    out_x = out_x / 255
    return out_x, out_y

# fashion_file = "../input/fashionmnist/fashion-mnist_train.csv"
# fashion_data = np.loadtxt(fashion_file, skiprows=1, delimiter=',')
# x, y = prep_data(fashion_data, train_size=50000, val_size=5000)


def prepare():
	for filename in os.listdir('allimages'):
		if filename.endswith(".JPEG"):
			if not os.path.exists('allimages/'+filename.split('.')[0]+str('.xml')):
				os.remove('allimages/'+filename)
				print('deleted'+filename) 

def converttonumpy():
	ctr=0
	imarray=np.zeros((239,size,size))
	for filename in os.listdir('allimages'):
		if filename.endswith(".jpg"):
			
			im = cv2.imread('allimages/'+filename)
			im=im[:,:,0]
			imarray[ctr]=im
			ctr+=1
	return imarray
arr=converttonumpy()
print(arr)

			# im.append()
			# print(im.shape)
			# print(im[3][3])
			# while(1):
			# 	continue
def makedata():
	for filename in os.listdir('allimages'):
		if filename.endswith(".JPEG"):
			im = cv2.imread('allimages/'+filename)
			gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
			cv2.imwrite('allimages/'+filename.split('.')[0]+str('.jpg'),gray_image)



def makedata1():
	for filename in os.listdir('allimages'):
		if filename.endswith(".JPEG"):
			os.remove('allimages/'+filename)

def resize():
	for filename in os.listdir('allimages'):
		if filename.endswith(".jpg"):
