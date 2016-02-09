### To do / In progress

* Clean up code
	* separate To do into a TODO.md file
	* add license
	* Write functions for common running different steps (e.g. just energy map or just display a seam) from an input file
	* Write function to generate saved file name?
* Make command line usage (use argparse)
	* Basic usage: `python seam_carver.py FILEPATH`
	* Options
		* energy function (default dual gradient)
		* saved file name
		* display initial energy map
		* display / save initial seam
		* display / save every n seams
		* remove n seams per iterataion
		* horizontal or vertical (default vertical seam removal, e.g. reduce image width)
* Use Gooey to make GUI from argparse.
* Try other energy functions (foward energy, etc)
* Add option for horizontal seam removal
* Optimizations
	* Only recompute changed energies (triangle surrounding seam) on each iteration
	* Option to remove multiple seams per pass
	* Option to save every intermediate image or every n intermediate image (save as what? Numpy array? Bitmap? JPG/PNG?)
* Add face-detection or thing for specifying a preservation mask