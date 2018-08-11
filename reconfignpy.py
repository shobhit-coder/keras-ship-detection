import numpy as np
img_rows=100
img_cols=100
num_images=239
y=np.load('outnpy1.npy')
# print(y) 
ctr=0
import os
for filename in os.listdir('allimages'):
    if filename.endswith('.jpg'):
        print(filename+str(y[ctr]))
        ctr+=1

# for num in range(239):
#     xmin=y[num][0]
#     ymin=y[num][1]
#     xmax=y[num][2]
#     ymax=y[num][3]
#     heightx=xmax-xmin
#     heighty=ymax-ymin
#     centerx=(xmax+xmin)/2
#     centery=(ymax+ymin)/2
#     y[num][0]=centerx
#     y[num][1]=centery
#     y[num][2]=heightx
#     y[num][3]=heighty

# np.save('outnpy1.npy',y)
# x=x.reshape(num_images, img_rows, img_cols, 1)
# print(x)