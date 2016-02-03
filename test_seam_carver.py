import numpy as np
from PIL import Image
import unittest
from seam_carver import (
	get_img_arr
)

class UtilsTestCase(unittest.TestCase):

	def setUp(self):
		# from the princeton site's project instructions
		self.test_img = np.array([
			[[255, 101, 51], [255, 101, 153], [255, 101, 255]],
			[[255, 153, 51], [255, 153, 153], [255, 153, 255]],
			[[255, 203, 51], [255, 204, 153], [255, 205, 255]],
			[[255, 255, 51], [255, 255, 153], [255, 255, 255]]
		])

	def test_import_square(self):
		square_img = get_img_arr("imgs/mountain_icon.jpg")
		self.assertEqual(square_img.shape, (64, 64, 3))

	def test_import_rect(self):
		rectangle_img = get_img_arr("imgs/HJoceanSmall.png")
		self.assertEqual(rectangle_img.shape, (285, 507, 3))

	def test_simple_energy(self):
		pass



if __name__ == "__main__":
	unittest.main()
