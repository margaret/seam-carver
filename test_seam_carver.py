from seam_carver import *

# sun.jpg is the realistic looking sun
sc = SeamCarver('imgs/HJOceanSmall.png')
# sc = SeamCarver('imgs/sun.jpg')
print "height: ", sc.height
print "width: ", sc.width
# sc.display()
sc.init_cache('simple_energy')
sc.scale(285,400)
# sc.scale(30, 20)
sc.display()
# To Fix
# int overflow
