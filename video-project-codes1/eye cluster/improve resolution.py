from PIL import Image
import glob, os
import numpy as np
from scipy import misc

# improve resolution and quality
for i in xrange(20):
	im = Image.open('/Users/wuqiong/Desktop/eye_data/'+str(i+1)+'.bmp')
	im = im.resize((512,256),Image.ANTIALIAS)
	quality_val = 1000
	im.save('/Users/wuqiong/Desktop/new/'+str(i+1)+'.jpg', 'JPEG', quality=quality_val)
 
'''
im1 = Image.open('/Users/wuqiong/Desktop/set/7.png')
im1 = im1.convert('L') 
im1 = im1.resize((512,256),Image.ANTIALIAS)
quality_val = 5000
im1.save('/Users/wuqiong/Desktop/new/x.jpg', 'JPEG', quality=quality_val)

'''
