#!/usr/bin/env python
from __future__ import (
  division,
  print_function
)
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from PIL import Image

from seam_carver import (
    energy_map,
    neighbors,
    cumulative_energy,
    find_seam,
    remove_seam
)
from energy_functions import (
    dual_gradient_energy
)
from utils import (
    get_img_arr,
    new_shape_for_ratio
)
class UtilsTestCase(unittest.TestCase):

    def setUp(self):
        # numbers from the princeton site's project instructions
        # http://www.cs.princeton.edu/courses/archive/spring14/cos226/assignments/seamCarving.html
        self.img_1 = np.array([
            [[255, 101, 51], [255, 101, 153], [255, 101, 255]],
            [[255, 153, 51], [255, 153, 153], [255, 153, 255]],
            [[255, 203, 51], [255, 204, 153], [255, 205, 255]],
            [[255, 255, 51], [255, 255, 153], [255, 255, 255]]
        ])

        self.img_2 = np.array([
            [[ 78,209, 79], [ 63,118,247], [ 92,175, 95], [243, 73,183], [210,109,104], [252,101,119]],
            [[224,191,182], [108, 89, 82], [ 80,196,230], [112,156,180], [176,178,120], [142,151,142]],
            [[117,189,149], [171,231,153], [149,164,168], [107,119, 71], [120,105,138], [163,174,196]],
            [[163,222,132], [187,117,183], [ 92,145, 69], [158,143, 79], [220, 75,222], [189, 73,214]],
            [[211,120,173], [188,218,244], [214,103, 68], [163,166,246], [ 79,125,246], [211,201, 98]]
        ])

        self.img_2_dual_gradient_energy_map = [
            [57685,   50893,    91370,    25418,    33055,    37246],
            [15421,   56334,    22808,    54796,    11641,    25496],
            [12344,   19236,    52030,    17708,    44735,    20663],
            [17074,   23678,    30279,    80663,    37831,    45595],
            [32337,   30796,    4909,     73334,    40613,    36556]
        ]

        self.img_3 = np.array([
            [1, 4, 3, 5, 2],
            [3, 2, 5, 2, 3],
            [5, 2, 4, 2, 1]
        ])

        self.img_3_paths = np.array([
            np.arange(5) * np.nan,
            [0, -1, 0, 1, 0],
            [1, 0, -1, 0, -1]
        ], dtype=np.int64)

    def test_import_square(self):
        square_img = get_img_arr("imgs/mountain_icon.jpg")
        self.assertEqual(square_img.shape, (64, 64, 3))

    def test_import_rect(self):
        rectangle_img = get_img_arr("imgs/HJoceanSmall.png")
        self.assertEqual(rectangle_img.shape, (285, 507, 3))

    def test_neighbors_on_row_edge(self):
        pixels_right = neighbors(self.img_1, 1, 2)
        answer_right = ([255, 153, 153], [255, 153, 51], [255, 101, 255], [255, 205, 255])
        # raises AssertionError if wrong, otherwise returns None
        assert_array_equal(pixels_right, answer_right)
        pixels_left = neighbors(self.img_1, 2, 0)
        answer_left = ([255, 205, 255], [255, 204, 153], [255, 153, 51], [255, 255, 51])
        assert_array_equal(pixels_left, answer_left)

    def test_neighbors_on_col_edge(self):
        pixels = neighbors(self.img_1, 3, 1)
        answer = ([255, 255, 51], [255, 255, 255], [255, 204, 153], [255, 101, 153])
        assert_array_equal(pixels, answer)

    def test_neighbors_on_corner(self):
        pixels = neighbors(self.img_1, 0,0)
        answer = ([255, 101, 255], [255, 101, 153], [255, 255, 51], [255, 153, 51])
        assert_array_equal(pixels, answer)

    def test_dual_gradient(self):
        img1_21 = dual_gradient_energy(*neighbors(self.img_1, 2, 1))
        self.assertEqual(img1_21, 52024)
        img1_11 = dual_gradient_energy(*neighbors(self.img_1, 1, 1))
        self.assertEqual(img1_11, 52225)

    def test_energy_map(self):
        dge_map = energy_map(self.img_2, dual_gradient_energy)
        assert_array_equal(dge_map, self.img_2_dual_gradient_energy_map)

    def test_cumulative_energy(self):
        # numbers from wikipedia article
        cumulative_energy_key = np.array([
            [1, 4, 3, 5, 2],
            [4, 3, 8, 4, 5],
            [8, 5, 7, 6, 5]
        ])
        results = cumulative_energy(self.img_3)
        assert_array_equal(results[0], self.img_3_paths, err_msg='paths not equal')
        assert_array_equal(results[1], cumulative_energy_key, err_msg='path_energies not equal')

    def test_find_seam(self):
        start = 1
        seam_result = find_seam(self.img_3_paths, start)
        seam_key = [0, 1, 1]
        assert_array_equal(seam_result, seam_key)

    def test_remove_seam(self):
        seam = [0, 1, 1]
        cropped_result = remove_seam(self.img_3, seam)
        cropped_key = np.array([
            [4, 3, 5, 2],
            [3, 5, 2, 3],
            [5, 4, 2, 1]
        ])
        assert_array_equal(cropped_result, cropped_key)

    def test_scaling(self):
        im = get_img_arr('imgs/castle_small.jpg')

        h, w = new_shape_for_ratio(im, 315, 851, scale_x=False)
        assert(abs(h/w - 315/851) < 0.001)

        h, w = new_shape_for_ratio(im, 215, 851)
        assert(abs(h/w - 215/851) < 0.001)


if __name__ == "__main__":
    unittest.main()
