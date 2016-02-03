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


def simple_energy(n1, n2, n3, n4, n5, n6, n7, n8):
	"""
	:params n1, n2, n3, n4, n5, n6, n7, n8
		each is length 3 array [r, g, b] that represents the
		neighbor of a pixel
	
		n1 n2 n3
		n4 ## n5
		n6 n7 n8

	:returns energy
		int representing the energy of the pixel with those neighbors

	"""
	pass


def neighbors(img, pix_coords):
	"""Return a list of the neighboring pixels of the pixel at pix_coords

	:pix_coords
		tuple (x, y) of the location of the target pixel within img
	:img
		np array representing the image of interest

	:returns
		list of neighboring pixels [[r, g, b], [r, g, b], ...] ordered for
		use in the energy functions

	"""
	pass


def energy_map(img):
	"""
	:img
		numpy array representing the image of interest

	:returns
		numpy array with the same height and width as img, but each 
		energy[x][y] is an int specifying the energy of that pixel

	Not sure if we should be recasting into uint8.
	"""
	pass


def display_energy_map(img):
	pass


if __name__ == "__main__":
	print("Hi, we are closed for rennovation.\nPlease check back later.")