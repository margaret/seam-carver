# seam-carver

Basic implementation of content-aware image resizing. Still in progress. 

Based on assignments from:

[UC Berkeley](https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj4-seamcarving/index.html), [Princeton](http://www.cs.princeton.edu/courses/archive/spring14/cos226/assignments/seamCarving.html), and [Brown's](http://cs.brown.edu/courses/cs129/results/proj3/taox/) computational image processing courses.

And the always-dependable [wikipedia page](https://en.wikipedia.org/wiki/Seam_carving).

[Original paper](https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj4-seamcarving/imret.pdf) by Shai Avidan and Ariel Shamir.


### To do / In progress

* write seam-finder (lol)
* Make the thing actually usable (start out with a command line program, maybe eventually get an interactive version up). Put in a couple of reasonable try-catch blocks.
* try other energy functions (foward energy, etc)
* optimizations - only recompute changed energies on each iteration
* add face-detection or thing for specifying a preservation mask

### Outline

Things that we need to be able to do in order to do a simple content-aware image resize in a single direction. The other direction can be done by rotating the image and shoving it through the exact same steps. 

1. Read in an image
2. Calculate the energy function for the whole image
	1. Function to calculate the energy of a single pixel, given the values of its neighboring pixels
	2. Function to calculate energy map for entire image
	3. (extra) Display energy map of image as heat map
3. Find the seam of lowest energy
4. Remove seam of lowest energy
5. Repeat 2 through 4 until image is as small as specified
6. Save resized image.

### Notes

* Uses the notation where img[x][y] means img[row][col], which is consistent with numpy arrays, but which is supposedly the opposite of the convention in image processing.
* I'm working in Python 2 out of habit (all academic exercises are in Python 2, right??).
* seam_carver.py cannot run as script as-is in a virtual env because of something simple probably that I just haven't looked up yet.


### Dependencies

Numpy (installed as part of the SciPy pack) and Pillow (the active fork of PIL). At the moment PIL is just used for file i/o as all of the actual image manipulation is in numpy, but might use it later for other stuff.

Standard `pip install -r requirements.txt`. I recommend installing this in a virtual environment because installing any of the Python scientific libraries is a pain, and while Anaconda is convenient, it can also potentially make installing other libraries/SDKs even more complicated. 
