import os
import numpy as np
from PIL import Image

def get_img_arr(filename):
    """
    :string filename
        path to png or jpg file to use

    :returns 3-D np.array (of uint8) shape (height, width, 3)
        where the pixel at arr[x][y] is the array [r,g,b]"""
    return np.array(Image.open(filename))


def pad_img(img, target_height, target_width, center=False):
    """
    Pad img to be target_height by target_width, with empty areas filled in
    black.
    http://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python

    :img
        3-D numpy array representing an RGB image
    :target_height
        int
    :target_width
        int
    :center
        bool -- whether to center the orignal image. Otherwise, it's pasted into the
        upper left corner.

    :returns
        the padded PIL Image
    """
    old_size = (img.shape[1], img.shape[0])
    new_size = (target_width, target_height)
    paste_coords = ((new_size[0]-old_size[0])/2, (new_size[1]-old_size[1])/2) if center else (0,0)

    old_img = Image.fromarray(img)
    new_im = Image.new("RGB", new_size) # initialized to black padding
    
    new_im.paste(old_img, box=paste_coords)
    
    return new_im


def bulk_pad(unpadded, padded, height, width):
    """
    Script to pad all images in a directory to the same shape.

    :unpadded
        str - name of directory containing images to pad. 
    :padded
        str - name of directory (should already exist) to place padded images.
    :height
        int - height in pixels to pad images to
    :width
        int - width in pixels to pad images to
    """
    for f in os.listdir(unpadded):
        im = get_img_arr(unpadded + '/' + f)
        padded = pad_img(im, 279, 411)
        padded.save(padded + '/' + f)


def new_shape_for_ratio(img, h, w, shrink=False):
    """
    Calculate the height and width of an image scaled to the ratio h:w

    :img
        3-D numpy array shape=(height,width,3) representing an image
    :h
        int - height of desired ratio
    :w
        int - width of desired ratio
    :shrink
        bool - whether to return the largest smaller crop of the image

    :returns
        tuple (int, int) of the new height and width in pixels
    """
    raise NotImplementedError


def every_n(n, height):
    """
    n and height are int or float

    returns a list of every nth nonzero int up to and not including height
    """
    return [i for i in xrange(1,height) if i%n==0]

                