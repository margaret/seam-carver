# seam-carver

Basic implementation of content-aware image resizing. Still in progress. 

Based on assignments from:

http://www.cs.princeton.edu/courses/archive/spring14/cos226/assignments/seamCarving.html

http://cs.brown.edu/courses/cs129/results/proj3/taox/

### Update 2016-02-02

Aight, I'm gonna just redo this, it was becoming kind of a mess, anyway. 

### To do / In progress

* Set up
	* requirements.txt for virtualenv
* Write pseudocode for naively (no optimizations) resizing a single image
* Figure out which functions from the old version can be reused
* Write stubs and unit tests for functions
* Make the thing actually usable (start out with a command line program, maybe eventually get an interactive version up)
* try other energy functions (foward energy, etc)
* use the numpy-optimized arithmetic functions
* add face-detection

### Outline

Things that we need to be able to do in order to do a simple content-aware image resize in a single direction. The other direction can be done by rotating the image and shoving it through the exact same steps. 

1. Read in an image
2. Calculate the energy function for the whole image
	a. Function to calculate the energy of a single pixel, given the values of its neighboring pixels
	b. Function to calculate energy map for entire image
	c. (extra) Display energy map of image as heat map
3. Find the seam of lowest energy
4. Remove seam of lowest energy
5. Repeat 2 through 4 until image is as small as specified

### Notes

* Uses the notation where img[x][y] means img[row][col], which is consistent with numpy arrays, but which is supposedly the opposite of the convention in image processing.
* Keeping `seam_carver_old.py` around as reference file.
* I'm working in Python 2 out of habit (all academic exercises are in Python 2, right??).


### Dependencies

Numpy (installed as part of the SciPy pack) and Pillow (the active fork of PIL). At the moment PIL is just used for file i/o as all of the actual image manipulation is in numpy, but might use it later for other stuff.

Standard `pip install -r requirements.txt`. I recommend installing this in a virtual environment because installing any of the Python scientific libraries is a pain, and while Anaconda is convenient, it can also potentially make installing other libraries/SDKs even more complicated. 
