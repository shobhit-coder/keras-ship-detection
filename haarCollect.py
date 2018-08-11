# This file is used to extract images from image-net.org ,resize images and generate description files.
# http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04530566
# http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09376198
import os
import cv2
import numpy as np 
import urllib.request


def store_raw_image():
    images_link = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n04530566'
    images_link = urllib.request.urlopen(images_link).read().decode()

    if not os.path.exists('ship2'):
        os.makedirs('ship2')

    pic_num=1

    for i in images_link.split('\n'):
        try:
            print(pic_num)
            pic_num+=1
            if not os.path.isfile('n04530566/n04530566_'+str(pic_num)+'.xml'):
                continue
            print(i)
            urllib.request.urlretrieve(i,"ship2/n04530566_"+str(pic_num)+'.jpg')

            img = cv2.imread("ship2/n04530566_"+str(pic_num)+".jpg")#,cv2.IMREAD_GRAYSCALE)
            resized_image = img#cv2.resize(img,(100,100))

            cv2.imwrite("ship2/n04530566_"+str(pic_num)+".jpg",resized_image)
            

        except Exception as e:
            print(str(e))

def myfunc():

    for filename in os.listdir('temp'): #For every element of this list (containing 'test' only atm)
        print(filename)
        try: #Try to
            if os.path.exists('temp/'+str(filename)): #If the image pic_num (= 1 at first) exists
                print(filename) #Prints 'test' (because it's only element of the list)
                #Initialize var img with image content (opened with lib cv2)
                img=cv2.imread('temp/'+str(filename)) 
                img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                #We resize the image to dimension 100x100 and store result in var resized_image
                resized_image=cv2.resize(img,(24,24)) 
                #Save the result on disk in the "small" folder
                cv2.imwrite("temp/m"+filename,resized_image)
            # pic_num+=1 #Increment variable pic_num
        except Exception as e: #If there was a problem during the operation
            print(str(e)) #Prints the exception
  



def find_uglies():
    match = False
    for file_type in ['all']:
        for img in os.listdir(file_type):
            for ugly in os.listdir('uglies'):
                try:
                    current_image_path = str(file_type)+'/'+str(img)
                    ugly = cv2.imread('uglies/'+str(ugly))
                    question = cv2.imread(current_image_path)
                    if ugly.shape == question.shape and not(np.bitwise_xor(ugly,question).any()):
                        print('That is one ugly pic! Deleting!')
                        print(current_image_path)
                        os.remove(current_image_path)
                except Exception as e:
                    print(str(e))


def create_pos_n_neg():
    for file_type in ['small24']:
        
        for img in os.listdir(file_type):

            if file_type == 'small24':
                print('testing24')
                line = file_type+'/'+img+' 1 0 0 24 24\n'
                with open('info24.dat','a') as f:
                    f.write(line)
            elif file_type == 'neg':
                print('test')
                line = file_type+'/'+img+'\n'
                with open('bg.txt','a') as f:
                    f.write(line)

# myfunc()
# create_pos_n_neg()
find_uglies()
# store_raw_image()
