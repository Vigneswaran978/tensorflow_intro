import cv2

import numpy as np
from scipy import misc

i = misc.ascent()

# Next we can use the pyplot libraty to draw the image so we know what it looks like

import matplotlib.pyplot as plt

plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

