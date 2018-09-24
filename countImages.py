import os
num=0
for filename in os.listdir('imagescopy1'):
	if filename.endswith(".JPEG") or filename.endswith(".png"):
            print(filename)
            num+=1

print(num)
