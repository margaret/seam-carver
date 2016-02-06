import sys
import numpy as np
from PIL import Image

def get_img_arr(filename):
	"""
	:string filename
		path to png or jpg file to use

	returns:
		the image as an np.array (of uint8) shape (height, width, 3)
		the pixel at arr[x][y] is the array [r,g,b]"""
	return np.array(Image.open(filename))


def simple_energy(x0, x1, y0, y1):
	"""e(I) = |deltax I| + |deltay I| The first energy function introduced in
	https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj4-seamcarving/imret.pdf
	:params
		The east/west/north/south neighbors of the pixel whose energy to calculate.
		Each is an len-3 array [r,g,b]
	:returns
		float
	"""
	return sum(abs(x0-x1) + abs(y0-y1))


def dual_gradient_energy(x0, x1, y0, y1):
	"""Suggested from
	http://www.cs.princeton.edu/courses/archive/spring14/cos226/assignments/seamCarving.html

	:params
		The east/west/north/south neighbors of the pixel whose energy to calculate.
		Each is an len-3 array [r,g,b]
	:returns
		float
	"""
	return sum(pow((x0-x1), 2) + pow((y0-y1), 2))


def neighbors(img, row, col):
	"""
	:img
		the np array representing the image
	:row, col
		int coordinates for the pixel to calculate energy for
	
	:returns 
		tuple of 3 numpy arrays [r,g,b]
		   y0
		x0 -- x1
		   y1
	"""
	height, width = img.shape[:2]

	if row == 0:
		y0 = img[height-1][col]
		y1 = img[row+1][col]
	elif row == height - 1:
		y0 = img[row-1][col]
		y1 = img[0][col]
	else:
		y0 = img[row-1][col]
		y1 = img[row+1][col]

	if col == 0:
		x0 = img[row][width-1]
		x1 = img[row][col+1]
	elif col == width - 1:
		x0 = img[row][col-1]
		x1 = img[row][0]
	else:
		x0 = img[row][col-1]
		x1 = img[row][col+1]

	return x0, x1, y0, y1


def energy_map(img, fn):
	"""
	:img
		numpy array representing the image of interest
		shape is (height,width,3)
	:fn
		The energy function to use. Should take in 4 pixels
		and return a float.

	:returns
		numpy array with the same height and width as img, but each 
		energy[x][y] is an int specifying the energy of that pixel

	Not sure if we should be recasting into uint8.
	Fix this later to use the numpy loop optimization thing, which I think is a thing.
	"""
	energy = np.zeros(img.shape[:2])
	for i,row in enumerate(img):
		for j,pixel in enumerate(row):
			energy[i][j] = fn(*neighbors(img, i,j))
	return energy


def display_energy_map(img_map):
	"""
	:img
		array representing energy map, shaped like (height, width)
	"""

	normed = img_map / float(img_map.max())
	scaled = normed * 255
	energy = Image.fromarray(scaled).show()


if __name__ == "__main__":
	# Display the simple energy and dual gradient energy maps for input file
	filename = sys.argv[1]
	img = get_img_arr("imgs/" + filename)
	Image.fromarray(img).show()

	dual_gradient_energy_map = energy_map(img, dual_gradient_energy)
	display_energy_map(dual_gradient_energy_map)

	simple_energy_map = energy_map(img, simple_energy)
	display_energy_map(simple_energy_map)
