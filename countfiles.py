import os
ctr=0
for filename in os.listdir('resizedall'):
    if filename.endswith(".JPEG"):
        ctr+=1

# for filename in os.listdir('400x400'):
#     if filename.endswith(".JPEG"):
#         if not os.path.exists('400x400/'+filename.split('.')[0]+str('.xml')):
#             os.remove('400x400/'+filename)
#             print('deleted'+filename)

print(ctr)