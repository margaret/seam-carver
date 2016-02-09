# seam-carver

Basic implementation of content-aware image resizing. Still in progress. 

Based on assignments from:

[UC Berkeley](https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj4-seamcarving/index.html), [Princeton](http://www.cs.princeton.edu/courses/archive/spring14/cos226/assignments/seamCarving.html), and [Brown's](http://cs.brown.edu/courses/cs129/results/proj3/taox/) computational image processing courses.

And the always-dependable [wikipedia page](https://en.wikipedia.org/wiki/Seam_carving).

[Original paper](https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj4-seamcarving/imret.pdf) by Shai Avidan and Ariel Shamir.

In this version, after installing the requirements, you can run `python seam_carver.py castle_small.jpg` in the top level of the directory and you can see two example energy maps for that image.

### To do / In progress

* Clean up code
	* Write functions for common running different steps (e.g. just energy map or just display a seam) from an input file
	* Write function to generate saved file name?
* Make command line usage (probably with argparse)
	* Basic usage: `python seam_carver.py FILEPATH`
	* Options
		* energy function (default dual gradient)
		* saved file name
		* display initial energy map
		* display / save initial seam
		* display / save every n seams
		* remove n seams per iterataion
		* horizontal or vertical (default vertical seam removal, e.g. reduce image width)
* Try other energy functions (foward energy, etc)
* Add option for horizontal seam removal
* Optimizations
	* Only recompute changed energies (triangle surrounding seam) on each iteration
	* Option to remove multiple seams per pass
	* Option to save every intermediate image or every n intermediate image (save as what? Numpy array? Bitmap? JPG/PNG?)
* [Progress bar](https://github.com/tqdm/tqdm) instead of print statements
* Add face-detection or thing for specifying a preservation mask

### Algorithm Outline

Things that we need to be able to do in order to do a simple content-aware image resize in a single direction. The other direction can be done by rotating the image and shoving it through the exact same steps. 

1. Read in an image
2. Calculate the energy function for the whole image
	1. Calculate the energy of a single pixel, given the values of its neighboring pixels
4. Calculate cumulative energy map and seam paths for image
5. Find the seam of lowest energy
6. Remove seam of lowest energy
7. Repeat 2 through 6 until image is as small as specified
8. Save resized image.

### Notes

* Uses the notation where img[x][y] means img[row][col], which is consistent with numpy arrays, but which is supposedly the opposite of the convention in image processing.
* I'm working in Python 2 out of habit (all academic exercises are in Python 2, right??).
* seam_carver.py cannot run as script as-is in a virtual env because of something simple probably that I just haven't looked up yet.


### Dependencies

Numpy (installed as part of the SciPy pack) and Pillow (the active fork of PIL). At the moment PIL is just used for file i/o as all of the actual image manipulation is in numpy, but might use it later for other stuff.

Standard `pip install -r requirements.txt`. I recommend installing this in a virtual environment because installing any of the Python scientific libraries is a pain, and while Anaconda is convenient, it can also potentially make installing other libraries/SDKs even more complicated. 
