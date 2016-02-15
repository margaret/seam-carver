
def simple_energy(x0, x1, y0, y1):
    """e(I) = |deltax I| + |deltay I| The first energy function introduced in
    https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj4-seamcarving/imret.pdf
    :params
        The east/west/north/south neighbors of the pixel whose energy to calculate.
        Each is an len-3 array [r,g,b]
    :returns float
        simple energy of pixel with those neighbors
    """
    return sum(abs(x0-x1) + abs(y0-y1))


def dual_gradient_energy(x0, x1, y0, y1):
    """Suggested from
    http://www.cs.princeton.edu/courses/archive/spring14/cos226/assignments/seamCarving.html

    :params
        The east/west/north/south neighbors of the pixel whose energy to calculate.
        Each is an len-3 array [r,g,b]
    :returns float
        dual gradient energy at the pixel with those neighbors
    """
    return sum(pow((x0-x1), 2) + pow((y0-y1), 2))
