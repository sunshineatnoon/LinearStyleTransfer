from PIL import Image
import os

for root,dirs,files in os.walk('./'):
    for file in files:
	if(file.endswith('.jpg')):
            img = Image.open(file)
            img.save(file.split('.')[0]+'.png')

