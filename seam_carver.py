#!/usr/local/bin/python
import sys
import os
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


def resize_image(full_img, cropped_pixels, energy_fn, display=False, pad=False, savepoints=None, save_name=None):
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
    os.mkdir(base)
    for i in trange(cropped_pixels, desc='cropping image by {0} pixels'.format(cropped_pixels)):
        e_map = energy_map(img, energy_fn)
        e_paths, e_totals = cumulative_energy(e_map)
        seam = find_seam(e_paths, seam_end(e_totals))
        img = remove_seam(img, seam)
        if i in savepoints:
            temp_img = Image.fromarray(img)
            temp_img.save(base+'/'+base+'_'+str(i).zfill(len(str(savepoints[-1])))+'.'+ext)
            if display:
                temp_img.show()
    return img


if __name__ == "__main__":
    filename = sys.argv[1]
    img = get_img_arr("imgs/" + filename)
    savefilename = sys.argv[2]
    crop = int(sys.argv[3])
    if len(sys.argv) == 5:
        print "Saving intermediate images in folder {0}".format(savefilename.split('.')[0])
        save_spacing = int(sys.argv[4])
        savepoints = every_n(save_spacing, img.shape[1])
        cropped = resize_image(img, crop, dual_gradient_energy, savepoints=savepoints, save_name=savefilename)
    else:
        cropped = resize_image(img, crop, dual_gradient_energy, save_name=savefilename)

    print "Image cropped from {0} to {1}".format(img.shape[:2], cropped.shape[:2])
    Image.fromarray(cropped).save(savefilename)

    ### Display the simple energy and dual gradient energy maps for input file ###
    # dual_gradient_energy_map = energy_map(img, dual_gradient_energy)
    # display_energy_map(dual_gradient_energy_map)

    # simple_energy_map = energy_map(img, simple_energy)
    # display_energy_map(simple_energy_map)
