import numpy as np
img_rows=100
img_cols=100
num_images=796
y=np.load('new_outnpy1fixed.npy')
# print(y) 
ctr=0
import os
for filename in os.listdir('resizedall'):
    if filename.endswith('.JPEG'):
        print(filename+str(y[ctr]))
        ctr+=1
        # if(ctr==1):
        #     break

# # for num in range(796):
# #     xmin=y[num][0]
# #     ymin=y[num][1]
# #     xmax=y[num][2]
# #     ymax=y[num][3]
# #     heightx=xmax-xmin
# #     heighty=ymax-ymin
# #     centerx=(xmax+xmin)/2
# #     centery=(ymax+ymin)/2
# #     y[num][0]=centerx
# #     y[num][1]=centery
# #     y[num][2]=heightx
# #     y[num][3]=heighty

# np.save('new_outnpy1fixed.npy',y)
# x=x.reshape(num_images, img_rows, img_cols, 1)
# np.save('new_innpy1fixed.npy',x)
# print(x)