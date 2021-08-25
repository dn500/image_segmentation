# unsupervised image segmentation
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb , mark_boundaries
from skimage.util import img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour

sc = 2000
sg = 0.66
ms = 8000

#sc = 2100
#sg = 0.66
#ms = 5000

#sc = 2000
#sg = 0.66
#ms = 4800

name = 'F100.png' 

img = plt.imread(name)
segments_fz = felzenszwalb(img, scale=sc, sigma=sg, min_size=ms)
print(f"Felzenszwalb number of segments F100: {len(np.unique(segments_fz))-1}")


ver_len,hor_len,nmask = img.shape


top_crop = 0
for i in range(ver_len//2):
	if np.mean(img[i,:])==1:
		top_crop = top_crop + 1 

bottom_crop = ver_len//2		
for i in range(ver_len//2,ver_len):
	if np.mean(img[i,:])<1:
		bottom_crop = bottom_crop + 1 

cropped_img = img[top_crop:bottom_crop,:,:]

ver_len,hor_len,nmask = cropped_img.shape




sc = 2000
sg = 0.65
ms = 16000

for nchunks in [2 ,4 ,8]:
	print(nchunks)
	for i in range(nchunks):
		ini = int(ver_len/nchunks)*i
		fin = ini+int(ver_len/nchunks)-1
		subimg = cropped_img[ini:fin,:]
		if np.mean(subimg)<1:
			subsegments_fz = felzenszwalb(subimg, scale=sc, sigma=sg, min_size=ms//nchunks)
			print(f"Felzenszwalb number of segments subimage F100: {len(np.unique(subsegments_fz))-1}")
			#plt.imshow(subsegments_fz)
			#plt.show()



