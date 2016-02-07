#!/usr/local/bin/python
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
		tuple of 3 1-D numpy arrays [r,g,b]
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
		2-D numpy array with the same height and width as img, but each 
		energy[x][y] is an int specifying the energy of that pixel

	Fix this later to use the numpy loop optimization things, which I think is a thing.
	"""
	energy = np.zeros(img.shape[:2])
	for i,row in enumerate(img):
		for j,pixel in enumerate(row):
			energy[i][j] = fn(*neighbors(img, i,j))
	return energy


def cumulative_energy_map(energy):
	"""
	https://en.wikipedia.org/wiki/Seam_carving#Dynamic_Programming
	
	:energy
		numpy array produced by energy_map

	:returns
		3-D array with shape (height, width, 2) where each element of the matrix is
		(cumulative energy, index of previous link)
	"""
	pass


def find_seam(total_energy):
	"""
	:total_energy
		output of cumulative_energy_map
	
	:returns
		1-dimensional array the height of the image where each element is the
		x-coordinate of the pixel to be removed at that y-coordinate.
		e.g. [4,4,3,2] means "remove pixels (0,4), (1,4), (2,3), and (3,2)"
	"""
	pass


def remove_seam(img, seam):
	"""
	:img
		3-D numpy array representing the RGB image you want to resize
	:seam
		1-D numpy array of the seam to remove. Output of seam function
	
	:returns
		3-D numpy array of the image that is 1 pixel shorter in width than
		input img
	"""
	pass


def display_energy_map(img_map):
	"""
	:img
		2-D array representing energy map, shaped like (height, width)
	"""
	scaled = img_map * 255 / float(img_map.max())
	energy = Image.fromarray(scaled).show()


def compress_image(full_img, cropped_pixels, energy_fn):
	"""
	Yo momma so fat, this bleeding-edge optimized seam-carving algorithm still
	tryna compress her picture to normal size and it's been running all day.

	:full_img
		3-D numpy array of the image you want to crop.

	:cropped_pixels
		int - number of pixels you want to shave off the width. Aka how many
		vertical seams to remove.

	:energy_fn
		energy function for energy_map to use. Should have the same interface
		as dual_gradient_energy and simple_energy

	:returns
		3-D numpy array of your now cropped_pixels-slimmer image. 

	(Also jk jk this is totally the naive implementation. Doesn't change how
	fat yo momma is though.)
	"""
	# we practice a non-destructive philosophy around these parts
	img = full_img.copy()

	for i in range(cropped_pixels):
		e_map = energy_map(img, energy_fn)
		e_paths = cumulative_energy_map(e_map)
		seam = find_seam(e_paths)
		img = remove_seam(img, seam)

	return img


if __name__ == "__main__":
	# Display the simple energy and dual gradient energy maps for input file
	filename = sys.argv[1]
	img = get_img_arr("imgs/" + filename)
	Image.fromarray(img).show()

	dual_gradient_energy_map = energy_map(img, dual_gradient_energy)
	display_energy_map(dual_gradient_energy_map)

	simple_energy_map = energy_map(img, simple_energy)
	display_energy_map(simple_energy_map)
