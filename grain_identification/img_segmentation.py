# unsupervised image segmentation
#
#
# 
# pip3 install numpy skimage scipy cv2 matplotlib
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb , mark_boundaries, find_boundaries
from skimage.util import img_as_float
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from scipy import ndimage
import cv2


##############################################

sc = 2000
sg = 0.66
ms = 8000

##############################################
def aspect_ratio(grain,com_row,com_col):
	#com_row,com_col = ndimage.measurements.center_of_mass(grain)
	nr, nc = grain.shape # same as whole image

	ar = []
	for theta in np.linspace(0,90,2):
		rotation_matrix = cv2.getRotationMatrix2D((com_row,com_col),theta,1)
		test = cv2.warpAffine(grain*img[:,:,0],rotation_matrix,(nc,nr))

		bottom_projection = []
		for i in range(nc):
			bottom_projection.append(sum(test[:,i]))

		rigth_projection = []
		for i in range(nr):
			rigth_projection.append(sum(test[i,:]))


		lv = np.max(bottom_projection) # vertical length as the max of the projection
		lh = np.max(rigth_projection)
		r = np.max([lh,lv])/np.min([lh,lv])

		ar.append(r)

	return np.max(ar)

##############################################


#sc = 2100
#sg = 0.66
#ms = 5000

#sc = 2000
#sg = 0.66
#ms = 4800

inputImage = 'F100-exp.jpeg'
# img = plt.imread('F100.png')
img = plt.imread(inputImage)

ver_len,hor_len,nmask = img.shape



sc = 1000  
sg = 0.6
ms = 400
segments_fz = felzenszwalb(img, scale=sc, sigma=sg, min_size=ms)
boundaries = find_boundaries(segments_fz,connectivity=1, mode='thick', background=0)
number_of_grains = len(np.unique(segments_fz))
#plt.imshow(img)
#plt.imshow(boundaries,alpha=0.4) , plt.axis('equal') , plt.show()
# Substracted 3: background + flat bottom layer + boundary 
# print(f"Felzenszwalb segmentation - number of grains F100: {(number_of_grains) - 3}")

#plt.imshow(boundaries) , plt.axis('equal') , plt.show()
plt.imshow(segments_fz) , plt.axis('equal') , plt.show()

## Calculate the white space at the top and at the bottom of the image
top_white_space = 0
for i in range(ver_len//2):
	if np.mean(img[i,:])==1:
		top_white_space = top_white_space + 1 

bottom_white_space = ver_len//2	
for i in range(ver_len//2,ver_len):
	if np.mean(img[i,:])<1:
		bottom_white_space = bottom_white_space + 1 

left_white_space = 0
for i in range(hor_len//2):
	if np.mean(img[:,i])==1:
		left_white_space = left_white_space + 1 

right_white_space = hor_len//2
for i in range(hor_len//2,hor_len):
	if np.mean(img[:,i])<1:
		right_white_space = right_white_space + 1 

## cube size
lx = right_white_space - left_white_space
ly = bottom_white_space - top_white_space


grains_description = [] 
x = [] 
y = []

for gi in range(number_of_grains):
	grain = (segments_fz==gi)
	# Calculate center of mass of the grain, relative to the bottom left corner of the cube
	com_row,com_col = ndimage.measurements.center_of_mass(grain)
	comx = com_col - left_white_space
	comy = com_row # ver_len - com_row - (ver_len - bottom_white_space)
	cx = comx/lx
	cy = comy/ly
	# Calculate the grain size and which percentage it represent with in the cube
	grain_area = np.count_nonzero(grain) 
	# Calculate the aspect ratio
	ar = 1# aspect_ratio(grain,com_row,com_col)
	grains_description.append([comx,comy,cx,cy,grain_area,ar])
	#print([comx,comy,cx,cy,grain_area,ar])
	#x.append(comy)
	#y.append(grain_area)


data = np.asmatrix(grains_description)
#background_area = np.max(grains_description[:][4]) # assuming the backgound has index 0
#cube_area = hor_len*hor_len-background_area


np.savetxt(inputImage+'.dat', grains_description, delimiter='\t')

x = data[:,1]
y = data[:,4]
#z = data[:,5]
#plt.plot(x,y,'o') , plt.show()
#img = plt.imread('E100.png')
#segments_fz = felzenszwalb(img, scale=sc, sigma=sg, min_size=ms)
#print(f"Felzenszwalb number of segments E100: {len(np.unique(segments_fz))-1}")
##plt.imshow(segments_fz)
##plt.show()
#
#img = plt.imread('F2011.png')
#segments_fz = felzenszwalb(img, scale=sc, sigma=sg, min_size=ms)
#print(f"Felzenszwalb number of segments F2011: {len(np.unique(segments_fz))-1}")
##plt.imshow(segments_fz)
##plt.show()
#
#img = plt.imread('F210.png')
#segments_fz = felzenszwalb(img, scale=sc, sigma=sg, min_size=ms)
#print(f"Felzenszwalb number of segments F210: {len(np.unique(segments_fz))-1}")
##plt.imshow(segments_fz)
##plt.show()
#plt.imshow(segments_fz==4) , plt.axis('equal') , plt.show()
#
#img = plt.imread('F765.png')
#segments_fz = felzenszwalb(img, scale=sc, sigma=sg, min_size=ms)
#print(f"Felzenszwalb number of segments F765: {len(np.unique(segments_fz))-1}")
##plt.imshow(segments_fz)
##plt.show()#