import os
ctr=0
for filename in os.listdir('all'):
    if filename.endswith(".xml"):
        ctr+=1

print(ctr)