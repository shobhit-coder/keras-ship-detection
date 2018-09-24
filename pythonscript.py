import os 

def prepare():
	for filename in os.listdir('imagescopy'):
		if filename.endswith(".JPEG"):
			if os.path.exists('imagescopy/'+filename.split('.')[0]+str('.xml')):
				os.remove('imagescopy/'+filename)
				os.remove('imagescopy/'+filename.split('.')[0]+str('.xml'))
				print('deleted'+filename)

prepare()
