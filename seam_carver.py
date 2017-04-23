#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import sys

import numba
import numpy as np
from PIL import Image
from tqdm import trange

from energy_functions import (
    simple_energy,
    dual_gradient_energy,
)
from utils import (
    display_energy_map,
    every_n,
    get_img_arr,
    highlight_seam,
    pad_img
)

def neighbors(img, row, col):
    """
    Parameters
    ==========
    img: 3-D numpy.array
        the image
    row: int
    col: int
        coordinates for the pixel to calculate energy for

    Returns
    =======
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
    Parameters
    ==========

    img: numpy.array with shape (height, width, 3)
    fn: function
        The energy function to use. Should take in 4 pixels and return a float.

    :returns 2-D numpy array with the same height and width as img
        Each energy[x][y] is an int specifying the energy of that pixel
    """
    x0 = np.roll(img, -1, axis=1).T
    x1 = np.roll(img, 1, axis=1).T
    y0 = np.roll(img, -1, axis=0).T
    y1 = np.roll(img, 1, axis=0).T

    # we do a lot of transposing before and after here because sums in the
    # energy function happen along the first dimension by default when we
    # want them to be happening along the last (summing the colors)
    return fn(x0, x1, y0, y1).T

@numba.jit()
def cumulative_energy(energy):
    """
    https://en.wikipedia.org/wiki/Seam_carving#Dynamic_programming

    Parameters
    ==========
    energy: 2-D numpy.array(uint8)
        Produced by energy_map

    Returns
    =======
        tuple of 2 2-D numpy.array(int64) with shape (height, width).
        paths has the x-offset of the previous seam element for each pixel.
        path_energies has the cumulative energy at each pixel.
    """
    height, width = energy.shape
    paths = np.zeros((height, width), dtype=np.int64)
    path_energies = np.zeros((height, width), dtype=np.int64)
    path_energies[0] = energy[0]
    paths[0] = np.arange(width) * np.nan

    for i in range(1, height):
        for j in range(width):
            # Note that indexing past the right edge of a row, as will happen if j == width-1, will
            # simply return the part of the slice that exists
            prev_energies = path_energies[i-1, max(j-1, 0):j+2]
            least_energy = prev_energies.min()
            path_energies[i][j] = energy[i][j] + least_energy
            paths[i][j] = np.where(prev_energies == least_energy)[0][0] - (1*(j != 0))

    return paths, path_energies


def seam_end(energy_totals):
    """
    Parameters
    ==========
    energy_totals: 2-D numpy.array(int64)
        Cumulative energy of each pixel in the image

    Returns
    =======
        numpy.int64
        the x-coordinate of the bottom of the seam for the image with these
        cumulative energies
    """
    return list(energy_totals[-1]).index(min(energy_totals[-1]))


def find_seam(paths, end_x):
    """
    Parameters
    ==========
    paths: 2-D numpy.array(int64)
        Output of cumulative_energy_map. Each element of the matrix is the offset of the index to
        the previous pixel in the seam
    end_x: int
        The x-coordinate of the end of the seam

    Returns
    =======
        1-D numpy.array(int64) with length == height of the image
        Each element is the x-coordinate of the pixel to be removed at that y-coordinate. e.g.
        [4,4,3,2] means "remove pixels (0,4), (1,4), (2,3), and (3,2)"
    """
    height, width = paths.shape[:2]
    seam = [end_x]
    for i in range(height-1, 0, -1):
        cur_x = seam[-1]
        offset_of_prev_x = paths[i][cur_x]
        seam.append(cur_x + offset_of_prev_x)
    seam.reverse()
    return seam


def remove_seam(img, seam):
    """
    Parameters
    ==========
    img: 3-D numpy.array
        RGB image you want to resize
    seam: 1-D numpy.array
        seam to remove. Output of seam function

    Returns
    =======
        3-D numpy array of the image that is 1 pixel shorter in width than the input img
    """
    height, width = img.shape[:2]
    return np.array([np.delete(img[row], seam[row], axis=0) for row in range(height)])


def resize_image(full_img, cropped_pixels, energy_fn, pad=False, savepoints=None, save_name=None,
                 rotated=False, highlight=False):
    """
    Parameters
    ==========
    full_img: 3-D numpy.array
        Image you want to crop.
    cropped_pixels: int
        Number of pixels you want to shave off the width. Aka how many vertical seams to remove.
    energy_fn: function
        Energy function for energy_map to use. Should have the same interface as
        dual_gradient_energy and simple_energy
    pad: bool
        Whether or not to pad the saved image with a black border
    savepoints: list(int)
        Iterations on which to save the image
    save_name: str
        Required if savepoints is present. Base name for saved images.
        Must include file extension. E.g. if savename is 'castle_small_dge.jpg' and savepoints is a
        list of mod 20, then 'castle_small_dge_20.jpg', 'castle_small_dge_20.jpg', etc. will be
        stored in the directory 'castle_small_dge/'
    rotated: bool
        Whether the image has been transposed (and needs to be transposed back before saving)
    highlight: bool
        Whether to draw the seam to be removed on the image

    Returns
    =======
        3-D numpy array of your now cropped_pixels-slimmer image.
    """
    if savepoints is None:
        savepoints = []
    img = full_img.copy()
    if savepoints:
        os.mkdir(save_name.split('.')[0])
    for i in trange(cropped_pixels, desc='cropping image by {0} pixels'.format(cropped_pixels)):
        e_map = energy_map(img, energy_fn)
        e_paths, e_totals = cumulative_energy(e_map)
        seam = find_seam(e_paths, seam_end(e_totals))
        if i in savepoints:
            save_image_with_options(img, highlight, pad, seam, rotated, save_name,
                                    full_img.shape[0], full_img.shape[1], i, savepoints)
        img = remove_seam(img, seam)
    return img


def save_image_with_options(img, highlight, pad, seam, rotated, savename, original_height,
                            original_width, point, savepoints):
    if highlight:
        img = highlight_seam(img, seam)
    if pad:
        img = np.array(pad_img(img, original_height, original_width))
    if rotated:
        img = Image.fromarray(np.transpose(img, axes=(1, 0, 2)))
    else:
        img = Image.fromarray(img)
    base, ext = savename.split('.')
    img.save(base+'/'+base.split('/')[-1]+'_'+str(point).zfill(len(str(savepoints[-1])))+'.'+ext)


def main():
    parser = argparse.ArgumentParser(description="Intelligently crop an image along one axis")
    parser.add_argument('input_file')
    parser.add_argument('-a', '--axis', required=True,
                        help="What axis to shrink the image on.", choices=['x', 'y'])
    parser.add_argument('-p', '--pixels', type=int, required=True,
                        help="How many pixels to shrink the image by.")

    parser.add_argument('-o', '--output',
                        help="What to name the new cropped image.")
    parser.add_argument('-i', '--interval', type=int,
                        help="Save every i intermediate images.")
    parser.add_argument('-b', '--border', type=bool,
                        help="Whether or not to pad the cropped images to the size of the original")
    parser.add_argument('-s', '--show_seam', type=bool,
                        help="Whether to highlight the removed seam on the intermediate images.")

    args = vars(parser.parse_args())
    print(args)

    img = get_img_arr(args['input_file'])

    if args['axis'] == 'y':
        img = np.transpose(img, axes=(1, 0, 2))

    if args['output'] is None:
        name = args['input_file'].split('.')
        args['output'] = name[0] + '_crop.' + name[1]

    savepoints = every_n(args['interval'], img.shape[1]) if args['interval'] else None

    cropped_img = resize_image(img, args['pixels'], dual_gradient_energy,
                               save_name=args['output'], savepoints=savepoints,
                               rotated=args['axis'] == 'y', pad=args['border'],
                               highlight=args['show_seam'])

    if args['axis'] == 'y':
        cropped_img = np.transpose(cropped_img, axes=(1, 0, 2))

    if args['border']:
        h, w = img.shape[:2]
        if args['axis'] == 'y':
            h, w = w, h
        cropped_img = pad_img(cropped_img, h, w)
        cropped_img.save(args['output'])
    else:
        Image.fromarray(cropped_img).save(args['output'])

    print("\nImage {0} cropped by {1} pixels along the {2}-axis and saved as {3}\n".format(
        args['input_file'], args['pixels'], args['axis'], args['output']))


if __name__ == "__main__":
    main()

    ### Display the simple energy and dual gradient energy maps for input file ###
    # dual_gradient_energy_map = energy_map(img, dual_gradient_energy)
    # display_energy_map(dual_gradient_energy_map)

    # simple_energy_map = energy_map(img, simple_energy)
    # display_energy_map(simple_energy_map)
