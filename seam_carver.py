import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc


# basic seam carver implementation
# http://www.cs.princeton.edu/courses/archive/spring13/cos226/assignments/seamCarving.html
# http://cs.brown.edu/courses/cs129/results/proj3/taox/

class SeamCarver:
    def __init__(self, img_file, fn='simple_energy'):
        """
        indexing is
        (0,0) (0,1)
        (1,0) (1,1)
        """
        self.scary = 0
        self.img = misc.imread(img_file).astype(dtype=np.uint8, copy=True)
        # self.img = misc.imread(img_file)
        self.width = self.img.shape[1]
        self.height = self.img.shape[0]
        self.cost_cache = np.ones([self.height, self.width], dtype=float) * -1
        self.energy_funcs = {'simple_energy': self.simple_energy}

    def init_cache(self, fn='simple_energy'):
        # initialize cost_cache
        for i in xrange(self.width):
            self.cost_cache[0][i] = self.energy(0, i, self.energy_funcs[fn])
        # populate cost_cache
        print self.cost_cache[0]
        for i in xrange(1, self.img.shape[0]):
            for j in xrange(self.img.shape[1]):
                self.cost_cache[i][j] = self.cumulative_energy(i, j, fn)
        print self.scary

    def reset_cache(self):
        print "RESETTING PIXEL ENERGY CALCULATIONS"
        self.cost_cache = np.ones([self.img.shape[0], self.img.shape[1]], dtype=float) * -1
        self.init_cache()

    def display(self):
        plt.imshow(self.img)
        plt.show()

    def energy(self, x, y, fn):
        """
        calculate energy of pixel at x,y using energy function fn
        returns float"""
        return fn(x, y)

    def simple_energy(self, x, y):
        """
        absolute value instead of square
        returns int
        """
        if x == 0:  # in top border, rollover pixel above (x,y)
            # up = self.height - 1
            up = self.img.shape[0] - 1
        else:
            up = x - 1  # row above
        # if x == self.height - 1:  # in lower border, rollover pixel below
        if x == self.img.shape[0] - 1:
            down = 0
        else:
            down = x + 1
        if y == 0:  # in left border, rollover pixel to the left
            # left = self.width - 1
            left = self.img.shape[1] - 1
        else:
            left = y - 1
        # if y == self.width - 1:  # in right border, rollover pixel to the right
        if y == self.img.shape[1] - 1:
            right = 0
        else:
            right = y + 1

        # print "thing ", self.img[x_left][y][0] - self.img[x_right][y][0]
        try:
            r_x = np.abs(float(self.img[x][left][0]) - float(self.img[x][right][0]))
            r_y = np.abs(float(self.img[up][y][0]) - float(self.img[down][y][0]))
            g_x = np.abs(float(self.img[x][left][1]) - float(self.img[x][right][1]))
            g_y = np.abs(float(self.img[up][y][1]) - float(self.img[down][y][1]))
            b_x = np.abs(float(self.img[x][left][2]) - float(self.img[x][right][2]))
            b_y = np.abs(float(self.img[up][y][2]) - float(self.img[down][y][2]))
        except IndexError as e:
            raise Exception(e, "width:{0} height:{1} x:{2} y:{3} left:{4} right:{5} up:{6} down:{7}".format(self.width,
                                                                                                            self.height,
                                                                                                            x, y, left,
                                                                                                            right, up,
                                                                                                            down))

        return np.sum([r_x, r_y, g_x, g_y, b_x, b_y])

    def cumulative_energy(self, x, y, fn):
        """
        vertical cumulative energy at (x,y)
        (energy at x,y) + min(cumulative energy from adjacent pixels above)
        returns int
        """
        # print "called on {0}, {1}".format(x, y)
        self.scary += 1
        if x == 0 or (self.cost_cache[x][y] > -1):
            return self.cost_cache[x][y]
        else:
            energies = [self.cumulative_energy(x - 1, y, fn)]
            if y > 0:  # not in leftmost column
                energies.insert(0, self.cumulative_energy(x - 1, y - 1, fn))
            elif y < self.width:  # not in rightmost column
                energies.append(self.cumulative_energy(x - 1, y + 1, fn))
            return self.energy_funcs[fn](x, y) + min(energies)

    def find_vertical_seam(self):
        """
        returns list of indices for vertical seam (top --> bottom)
        """
        seam = np.zeros(self.height)  # array of x-coordinates
        # find index of min cumulative energy in bottom row
        tmp = np.where(self.cost_cache[self.height - 1] == self.cost_cache[self.height - 1].min())[0]
        if len(tmp) > 1:
            tmp = tmp[0]
        print "index of min cumulative energy: ", tmp
        seam[0] = tmp
        for i in xrange(2, self.height):
            x_prev = seam[i - 1]
            # get relevant slice of previous row
            if x_prev == 0: # lowest energy in leftmost column
                neighbors = self.cost_cache[self.height - i][x_prev:x_prev + 1]
            elif x_prev == self.width - 1: # lowest energy in rightmost column
                neighbors = self.cost_cache[self.height - i][x_prev - 1:x_prev]
            else: # somewhere in the middle
                neighbors = self.cost_cache[self.height - i][x_prev - 1:x_prev + 1]
            # print "x_prev ", x_prev
            # print "new offset ", np.where(neighbors == neighbors.min())[0][0] - 1
            print "x_prev ", x_prev
            print "neighoring pixels ", neighbors
            # print "row from cache ", self.cost_cache[self.height - i]
            print "should be a thing here ", self.cost_cache[self.height - i][x_prev:x_prev + 1]
            seam[i] = x_prev + np.where(neighbors == neighbors.min())[0][0] - 1  # shift neighbors
        return seam[::-1]

    def __pixel_shift__(self, x):
        """
        remove pixel from row and shift everything left
        """
        try:
            parts = np.vsplit(self.img[x], np.array([x, x + 1]))
            self.img[x] = np.vstack((parts[0], parts[2], parts[1]))
        except IndexError as e:
            print "x ", x
            print "img.shape ", self.img.shape
            raise e

    def __rm_right_col__(self):
        self.img = np.hsplit(self.img, np.array([self.width - 1]))[0]
        self.width -= 1

    def remove_vertical_seam(self, seam):
        """
        remove vertical seam from image
        """
        print "removing seam"
        for y in xrange(self.height - 1): # get dimensions from self.img, so it works with horizontal seam rmval
            self.__pixel_shift__(seam[y])
        self.__rm_right_col__()
        self.reset_cache()


    def scale_horizontal(self, delta):
        """
        remove delta vertical seams from image
        """
        print "HORIZONTAL"
        for i in xrange(delta):
            print "removing seam ", i
            seam = self.find_vertical_seam()
            self.remove_vertical_seam(seam)

    def scale_vertical(self, delta):
        """
        FIX THIS
        remove delta horizontal seams from image
        """
        print "VERTICAL"
        print "height {0}, width {1}".format(self.height, self.width)
        self.img = self.img.transpose()
        tmp = self.width
        self.width = self.height
        self.height = tmp
        print "height {0}, width {1}".format(self.height, self.width)
        self.scale_horizontal(delta)
        self.img = self.img.transpose()
        self.height = self.width
        self.width = tmp

    def scale(self, x, y):
        """
        ??? Does it matter if horizontal or vertical is called first?
        ? do both, then remove the one with lower energy
        new_dim is tuple of new (width, height)
        """
        if x > self.height  or y > self.width:
            raise Exception("Seam insertion is not implemented yet.")
        else:
            self.scale_horizontal(self.width - y)
            self.scale_vertical(self.height - x)

