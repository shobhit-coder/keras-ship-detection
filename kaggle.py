# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.python import keras
import os

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
import cv2
import xml.etree.ElementTree as ET
import numpy as np
def converttonumpy():
	ctr=0
	imarray=np.zeros((796,100,100))
	outputarray=np.zeros((796,4))
	for filename in os.listdir('resizedall'):
		if filename.endswith(".JPEG"):
			
			im = cv2.imread('resizedall/'+filename)
			im=im[:,:,0]
			imarray[ctr]=im
			# print(filename)
			tree = ET.parse('resizedall/'+str(filename.split('.')[0]+'.xml'))
			root = tree.getroot()
			if root[2].tag=='source':
				img_x_size=int(root[3][0].text)
				img_y_size=int(root[3][1].text)
				outputarray[ctr][0]=str(int(int(root[5][4][0].text)/img_x_size*100))
				outputarray[ctr][1]=str(int(int(root[5][4][1].text)/img_y_size*100))
				outputarray[ctr][2]=str(int(int(root[5][4][2].text)/img_x_size*100))
				outputarray[ctr][3]=str(int(int(root[5][4][3].text)/img_y_size*100))
			else:
				img_x_size=int(root[4][0].text)
				img_y_size=int(root[4][1].text)
				outputarray[ctr][0]=str(int(int(root[6][4][0].text)/img_x_size*100))
				outputarray[ctr][1]=str(int(int(root[6][4][1].text)/img_y_size*100))
				outputarray[ctr][2]=str(int(int(root[6][4][2].text)/img_x_size*100))
				outputarray[ctr][3]=str(int(int(root[6][4][3].text)/img_y_size*100))
				# outputarray[ctr][0]=root[6][4][0].text#=str(int(int(root[5][4][0].text)/img_x_size*100))
				# outputarray[ctr][1]=root[6][4][1].text#=str(int(int(root[5][4][1].text)/img_y_size*100))
				# outputarray[ctr][2]=root[6][4][2].text#=str(int(int(root[5][4][2].text)/img_x_size*100))
				# outputarray[ctr][3]=root[6][4][3].text#=str(int(int(root[5][4][3].text)/img_y_size*100))
			ctr+=1
			print(filename)



	return outputarray,imarray
outarr,inarr=converttonumpy()
# print(outarr)
np.save('newdata_innpy.npy',inarr)
np.save('newdata_outnpy.npy', outarr)

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
	# image=cv2.imread('shipcheck.jpg')
	# print(str(image.shape[:2])+' '+str(filename))
	for filename in os.listdir('resizedall'):
		if filename.endswith(".JPEG"):
			image=cv2.imread('resizedall/'+filename)
			# print(str(image.shape[:2])+' '+str(filename))
# resize()
#pic=pic[:][:][0]
#print(pic.shape)
# filename='n04530566_182.jpg'#'3D-Matplotlib.png'#
# im = cv2.imread('allimages/'+filename)
# cv2.imshow('frame',im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# from tensorflow.python import keras
# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, Flatten, Conv2D

# # Your Code Here
# fashion_model = Sequential()
# fashion_model.add(Conv2D(12,kernel_size=(3,3),activation='relu',input_shape=((img_rows, img_cols, 1))))
# fashion_model.add(Conv2D(2,kernel_size=(3,3),activation='relu'))
# fashion_model.add(Flatten())
# fashion_model.add(Dense(100,activation='relu'))
# fashion_model.add(Dense(num_classes,activation='softmax'))

# fashion_model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer='adam',
#               metrics=['accuracy'])

# fashion_model.fit(x, y,
#           batch_size=100,
#           epochs=4,
#           validation_split = 0.2)