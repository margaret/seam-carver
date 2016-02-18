#!/usr/local/bin/python
import sys
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import trange
from energy_functions import (
    simple_energy,
    dual_gradient_energy,
)
from utils import (
    pad_img,
    every_n,
    get_img_arr,
)


def neighbors(img, row, col):
    """
    :img
        the 3-D np array representing the image
    :row, col
        int coordinates for the pixel to calculate energy for

    :returns tuple of 3 1-D numpy arrays [r,g,b]
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

    :returns 2-D numpy array with the same height and width as img
        Each energy[x][y] is an int specifying the energy of that pixel
    """
    energy = np.zeros(img.shape[:2])
    for i,row in enumerate(img):
        for j,pixel in enumerate(row):
            energy[i][j] = fn(*neighbors(img, i,j))
    return energy


def cumulative_energy(energy):
    """
    https://en.wikipedia.org/wiki/Seam_carving#Dynamic_Programming
    
    :energy
        2-D numpy array produced by energy_map

    :returns tuple of 2 2-D array with shape (height, width).
        paths has the x-offset of the previous seam element for each pixel.
        path_energies has the cumulative energy at each pixel.
    """
    height, width = energy.shape
    paths = np.zeros((height,width))
    path_energies = np.zeros((height,width))
    
    for i in xrange(height):
        for j in xrange(width):
            target_energy = energy[i][j]
            if i == 0:
                path_energies[i][j] = target_energy
                paths[i][j] = float('nan')
            else:
                if j == 0:
                    prev_energies = list(path_energies[i-1, j:j+2])
                    least_energy = min(prev_energies)
                    path_energies[i][j] =  target_energy + least_energy
                    paths[i][j] = prev_energies.index(least_energy)
                else:
                    # Note that indexing past the right edge of a row, as will happen
                    # if j == width-1, will simply return the part of the slice that exists
                    prev_energies = list(path_energies[i-1, j-1:j+2])
                    least_energy = min(prev_energies)
                    path_energies[i][j] =  target_energy + least_energy
                    paths[i][j] = prev_energies.index(least_energy) - 1

    return paths, path_energies


def seam_end(energy_totals):
    """
    :energy_totals
        2-D numpy array with the cumulative energy of each pixel in the image

    :returns float
        the x-coordinate of the bottom of the seam for the image with these
        cumulative energies
    """
    return list(energy_totals[-1]).index(min(energy_totals[-1]))


def find_seam(paths, end_x):
    """
    :paths
        output of cumulative_energy_map
        2-D array where each element of the matrix is the offset of the index
        to the previous pixel in the seam
    :end_x
        int or float, the x-coordinate of the end of the seam
        list(energies[-1]).index(min(energies[-1]))
        
    :returns 1-D array with length == height of the image
        each element is the x-coordinate of the pixel to be removed at that
        y-coordinate. e.g. [4,4,3,2] means "remove pixels (0,4), (1,4), (2,3),
        and (3,2)"
    """
    height,width = paths.shape[:2]
    seam = [end_x]
    for i in xrange(height-1,0,-1):
        cur_x = seam[-1]
        offset_of_prev_x = paths[i][cur_x]
        seam.append(cur_x + offset_of_prev_x)
    seam.reverse()
    return seam


def remove_seam(img, seam):
    """
    :img
        3-D numpy array representing the RGB image you want to resize
    :seam
        1-D numpy array of the seam to remove. Output of seam function
    
    :returns 3-D numpy array of the image that is 1 pixel shorter in width than
        the input img
    """
    height,width = img.shape[:2]
    return np.array([np.delete(img[row], seam[row], axis=0) for row in xrange(height)])


def display_energy_map(img_map):
    """
    :img
        2-D array representing energy map, shaped like (height, width)
    """
    scaled = img_map * 255 / float(img_map.max())
    energy = Image.fromarray(scaled).show()


def display_seam(img, seam):
    """
    :img
        3-D numpy array representing the image
    :seam
        1-D numpy array with length == height of img representing the
        x-coordinates of the pixel to remove from each row.
    """
    highlight = img.copy()
    height,width = img.shape[:2]
    for i in xrange(height):
        j = seam[i]
        highlight[i][j] = np.array([255, 0, 0])
    Image.fromarray(highlight).show()


def resize_image(full_img, cropped_pixels, energy_fn, display=False, pad=False, savepoints=None, save_name=None, rotated=False):
    """
    :full_img
        3-D numpy array of the image you want to crop.
    :cropped_pixels
        int - number of pixels you want to shave off the width. Aka how many
        vertical seams to remove.
    :energy_fn
        energy function for energy_map to use. Should have the same interface
        as dual_gradient_energy and simple_energy
    :savepoints
        list of ints indicating iterations on which to save the image
    :save_name
        str - required if savepoints is present. Base name for saved images.
        Must include file extension. E.g. if savename is 'castle_small_dge.jpg'
        and savepoints is a list of mod 20, then 'castle_small_dge_20.jpg',
        'castle_small_dge_20.jpg', etc. will be stored in the directory 
        'castle_small_dge/'
    :pad
        bool - whether or not to pad the saved image with a black border
        (not implemented)
    :display
        bool - whether or not to display intermediate images

    :returns 3-D numpy array of your now cropped_pixels-slimmer image. 
    """
    if savepoints == None:
        savepoints = []
    # we practice a non-destructive philosophy around these parts
    img = full_img.copy()
    base,ext = save_name.split('.')
    if savepoints:
        os.mkdir(base)
    for i in trange(cropped_pixels, desc='cropping image by {0} pixels'.format(cropped_pixels)):
        e_map = energy_map(img, energy_fn)
        e_paths, e_totals = cumulative_energy(e_map)
        seam = find_seam(e_paths, seam_end(e_totals))
        img = remove_seam(img, seam)
        temp_img = img.copy()
        if i in savepoints:
            if pad:
                temp_img = np.array(pad_img(temp_img, full_img.shape[0], full_img.shape[1]))
            if rotated:
                temp_img = Image.fromarray(np.transpose(temp_img, axes=(1,0,2)))
            else:
                temp_img = Image.fromarray(temp_img)
            temp_img.save(base+'/'+base.split('/')[-1]+'_'+str(i).zfill(len(str(savepoints[-1])))+'.'+ext)
            if display:
                temp_img.show()
    return img


def main():
    parser = argparse.ArgumentParser(description="Intelligently crop an image along one axis")
    parser.add_argument('input_file')
    parser.add_argument('-a', '--axis', required=True, help="What axis to shrink the image on.", choices=['x', 'y'])
    parser.add_argument('-p', '--pixels', type=int, required=True, help="How many pixels to shrink the image by.")

    parser.add_argument('-o', '--output', help="What to name the new cropped image.")
    parser.add_argument('-i', '--interval', type=int, help="Save every i intermediate images.")
    parser.add_argument('-b', '--border', type=bool, help="Whether or not to pad the cropped images to the size of the original")

    args = vars(parser.parse_args())
    print args

    img = get_img_arr(args['input_file'])

    if args['axis'] == 'y':
        img = np.transpose(img, axes=(1,0,2))

    if args['output'] is None:
        name = args['input_file'].split('.')
        args['output'] = name[0] + '_crop.' + name[1]

    savepoints = every_n(args['interval'], img.shape[1]) if args['interval'] else None

    cropped_img = resize_image(img, args['pixels'], dual_gradient_energy, save_name=args['output'], savepoints=savepoints, rotated=args['axis']=='y', pad=args['border'])

    if args['axis']=='y':
        cropped_img = np.transpose(cropped_img, axes=(1,0,2))

    if args['border']:
        h, w = img.shape[:2]
        if args['axis']=='y':
            h, w = w, h
        cropped_img = pad_img(cropped_img, h, w)
        cropped_img.save(args['output'])
    else:
        Image.fromarray(cropped_img).save(args['output'])

    print "\nImage {0} cropped by {1} pixels along the {2}-axis and saved as {3}\n".format(args['input_file'], args['pixels'], args['axis'], args['output'])

if __name__ == "__main__":
    main()

    ### Display the simple energy and dual gradient energy maps for input file ###
    # dual_gradient_energy_map = energy_map(img, dual_gradient_energy)
    # display_energy_map(dual_gradient_energy_map)

    # simple_energy_map = energy_map(img, simple_energy)
    # display_energy_map(simple_energy_map)
