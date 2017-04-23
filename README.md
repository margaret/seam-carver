# seam-carver

![](https://travis-ci.org/margaret/seam-carver.svg?branch=master)

Basic implementation of content-aware image resizing. Still in progress! This is mostly for fun, as Photoshop has an implementation of this called [Content Aware Scaling](https://helpx.adobe.com/photoshop/using/content-aware-scaling.html).

![castle_demo](imgs/castle_small_300_seams.gif)

Based on assignments from [UC Berkeley](https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj4-seamcarving/index.html), [Princeton](http://www.cs.princeton.edu/courses/archive/spring14/cos226/assignments/seamCarving.html), and [Brown's](http://cs.brown.edu/courses/cs129/results/proj3/taox/) computational image processing courses, and the always-dependable [Wikipedia page](https://en.wikipedia.org/wiki/Seam_carving).

[Original paper](https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj4-seamcarving/imret.pdf) by Shai Avidan and Ariel Shamir.


## Usage

### Dependencies

* Numpy and Pillow (the active fork of PIL). At the moment Pillow is just used for file i/o as all of the actual image manipulation is in numpy, but might use it later for other stuff.
* tqdm for progress bar
* numba is used to speed up the part of the algorithm that uses dynamic programming and thus can't be optimized easily in pure numpy. On a Macbook Pro running OS X Sierra, this cuts the time to crop 100 pixels off `imgs/castle_small.jpg` from ~60s to ~6s. If you are having a hard time getting numba to install properly on your machine, you can simply comment out the `import numba` and the `@numba.jit()` decorator above the `cumulative energy` function and it should all still run, just a lot slower.

I strongly recommend using `conda` to set this up, because otherwise it can potentially be a huge pain to install numba. Otherwise you should probably at least use a virtualenv because a lot of the numpy stack / image processing stack tends to be finicky about dependencies.

Travis builds on Python 2.7, 3.5, and 3.6 (Linux) and it also works (manually installed via conda) on OS X Sierra.

#### Conda

[Install Anaconda](https://conda.io/docs/get-started.html) and then run `conda create env -f environment.yaml`, then enter the environment with `source activate seamcarver`. Exit the environment with `source deactivate`.

#### Other

`pip install -r requirements.txt` should work locally (assuming you've installed pip), in a virtualenv, or even inside conda.


### Arguments
Required args

One positional arg: filename of image to crop

```
    -a --axis       What axis to shrink the image on (x or y)
    -p --pixels     How many pixels to crop off the image
```

Optional args
```
    -o --output     What to name the cropped image.
    -i --interval   Save every i intermediate image.
    -b --border     Whether or not to pad the cropped images to the size of
                        the original.
    -s --show_seam  Whether or not to  draw the seam on the image before saving it.
```

Example: Crop 100 pixels off the height of `imgs/castle_small.jpg` and save every 10th iteration, with padding would be `python seam_carver.py imgs/castle_small.jpg -p 100 -a y -b True -i 10`



## Algorithm

Steps for reizing an image a single direction. The other direction can be done by rotating the image and shoving it through the exact same steps. 

1. Read in an image
2. Calculate the energy function for the whole image
	1. Calculate the energy of a single pixel, given the values of its neighboring pixels
4. Calculate cumulative energy map and seam paths for image
5. Find the seam of lowest energy
6. Remove seam of lowest energy
7. Repeat 2 through 6 until image is as small as specified
8. Save resized image.

## Notes

* Uses the notation where img[x][y] means img[row][col], which is consistent with numpy arrays, but which is supposedly the opposite of the convention in image processing. 

## Future

![castle_vertical](imgs/castle_small_vertical.gif)

* Currently only uses dual energy gradient energy function. As you can see in the vertical resizing example, it slowly decapitates the human figure and ends up scoring the grass as important (probably) due to the many small changes in color across the grass. Different energy functions work well with different types of images, for example, using a forward-energy algorithm would be better at preserving edges.
* Options to rescale by ratio. Instead of putting in the exact number of pixels, can rescale image to 5:3, 3:6, etc ratio.
