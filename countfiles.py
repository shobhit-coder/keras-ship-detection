import os
ctr=0
for filename in os.listdir('resizedall'):
    if filename.endswith(".JPEG"):
        ctr+=1

# for filename in os.listdir('resizedall'):
#     if filename.endswith(".JPEG"):
#         if not os.path.exists('resizedall/'+filename.split('.')[0]+str('.xml')):
#             os.remove('resizedall/'+filename)
#             print('deleted'+filename)

print(ctr)