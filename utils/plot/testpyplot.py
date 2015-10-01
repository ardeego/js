import plotutils as pltut
import numpy as np
import matplotlib.pyplot as plt

#from http://www.astrobetter.com/making-rgb-images-from-fits-files-with-pythonmatplotlib/

img = np.zeros((480,640, 3), dtype=float)
img[5:100,50:80,1] = 1.0
img[5:100,50:180,2] = 1.0
img[5:160,50:70,0] = 1.0

#img[:,:,0] = pltut.linear(img[:,:,0], scale_min=0, scale_max=1)
#img[:,:,1] = pltut.linear(img[:,:,1], scale_min=0, scale_max=1)
#img[:,:,2] = pltut.linear(img[:,:,2], scale_min=0, scale_max=1)

img = pltut.linear(img, scale_min=0, scale_max=1)

plt.figure()
plt.clf()
plt.imshow(img, aspect='equal')
plt.show()
