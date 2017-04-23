from __future__ import division
import os
import numpy as np
from PIL import Image

def get_img_arr(filename):
    """
    Parameters
    ==========
    filename: str
        path to png or jpg file to use

    Returns
    =======
        3-D np.array(uint8) with shape (height, width, 3) where the pixel at
        arr[x][y] is the array [r,g,b]"""
    return np.array(Image.open(filename))


def display_energy_map(img_map):
    """
    Parameters
    ==========
    img: 2-D numpy.array with shape (height, width)
        The energy map of the image
    """
    scaled = img_map * 255 / img_map.max()
    energy = Image.fromarray(scaled).show()


def highlight_seam(img, seam):
    """
    Parameters
    ==========
    img: 3-D numpy.array
        The image
    seam: 1-D numpy..array with length == height of img
        The x-coordinates of the pixel to remove from each row.

    Returns
    =======
        3-D numpy array representing the image, with the seam highlighted in red
    """
    if len(seam) != img.shape[0]:
        err_msg = "Seam height {0} does not match image height {1}"
        raise ValueError(err_msg.format(img.shape[0], len(seam)))
    highlight = img.copy()
    height, width = img.shape[:2]
    for i in range(height):
        j = seam[i]
        highlight[i][j] = np.array([255, 0, 0])
    return highlight


def pad_img(img, target_height, target_width, center=False):
    """
    Pad img to be target_height by target_width, with empty areas filled in black.
    http://stackoverflow.com/questions/11142851/adding-borders-to-an-image-using-python

    img: 3-D numpy array
        RGB image
    target_height: int
        Height to crop img to
    target_width: int
        Width to crop img to
    center: bool
        Whether to center the orignal image. Otherwise, it's pasted into the upper left corner.

    Returns
    =======
        Padded PIL Image
    """
    old_size = (img.shape[1], img.shape[0])
    new_size = (target_width, target_height)
    paste_coords = ((new_size[0]-old_size[0])/2, (new_size[1]-old_size[1])/2) if center else (0, 0)

    old_img = Image.fromarray(img)
    new_img = Image.new("RGB", new_size) # initialized to black padding

    new_img.paste(old_img, box=paste_coords)

    return new_img


def bulk_pad(unpadded, padded, height, width):
    """
    Script to pad all images in a directory to the same shape.

    Parameters
    ==========
    unpadded: str
        Name of directory containing images to pad.
    padded: str
        Name of directory (should already exist) to place padded images.
    height: int
        Height in pixels to pad images to
    width: int
        Width in pixels to pad images to
    """
    for f in os.listdir(unpadded):
        if not f.startswith('.'):
            im = get_img_arr(unpadded + '/' + f)
            padded_img = pad_img(im, height, width)
            padded_img.save(padded + '/' + f)


def new_shape_for_ratio(img, h, w, scale_x=True):
    """
    Calculate the height and width of an image scaled to the ratio h:w

    Parameters
    img: 3-D numpy.array with shape (height, width, 3)
        An image
    h: int
        Height of desired ratio
    w: int
        Width of desired ratio
    scale_x: bool
        Whether to shrink the image horizontally

    Returns
    =======
        tuple (int, int) of the new height and width in pixels
    """
    old_height, old_width = img.shape[:2]
    if scale_x:
        new_height = old_height
        scale = w / h
        new_width = int(old_height * scale)
    else:
        new_width = old_width
        scale = h / w
        new_height = int(old_width * scale)

    return new_height, new_width


def every_n(n, height):
    """
    Parameters
    n: int
    height: int

    Returns
    =======
        List of every nth nonzero int up to and not including height
    """
    return [i for i in range(1, height) if i%n == 0]
