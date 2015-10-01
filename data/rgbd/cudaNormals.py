import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
import scipy.io

import os
import re

import mayavi.mlab as mlab
from rgbdframe import RgbdFrame

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as comp

def depth2normalsCUDA(d):
  w,h = d.shape[1],d.shape[0]
  block = (16,16,1)
  grid = (int(np.ceil(float(w)/16.0)),int(np.ceil(float(h)/16.0)),1)
  
  with open('normals_kernel.cu') as kernelFile:
    kernel = kernelFile.read()
  mod = comp.SourceModule(kernel)
  depth2normals = mod.get_function('depth2normals')

  n = np.zeros((d.shape[0],d.shape[1],3),dtype=np.float32)
  depth2normals(drv.In(d),drv.InOut(n),np.int32(w),np.int32(h),
      grid =grid,block=block)

  return n


if __name__ == "__main__":

  path = '/home/jstraub/workspace/research/vpCluster/data/'
  algo = 'guy'#'guy'

  name = 'table_1'

  print 'processing '+path+name
  rgbd = RgbdFrame(525.0)
  rgbd.load(path+name)
  rgbd.showRgbd()
  rgbd.getPc()

  n=depth2normalsCUDA(rgbd.d)

  print n.shape
  print n.strides

  fig = plt.figure()
  plt.imshow(n)
  fig.show()

  rgbd.getNormals(algo=algo)

  raw_input()
