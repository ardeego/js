
import mayavi.mlab as mlab
import numpy as np
from scipy.linalg import solve, inv

from js.utils.plot.colors import colorScheme
from vpCluster.manifold.sphere import *

def plotGaussianTpS2(R):
  """ plot the tangent space """
  X,Y = np.meshgrid(np.linspace(-0.5,0.5,100),np.linspace(-.5,.5,100))
  # color according to covariance S
  S = np.eye(2)*0.03
  pts = np.c_[X.ravel(),Y.ravel()]
  C = -0.5*(pts.T*np.dot(inv(S),pts.T)).sum(axis=0) 
  C = np.exp(np.reshape(C,X.shape))
  Z = C*0.5 + 1.

  p3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()].T
  print p3.shape
  p3 = R.dot(p3)
  X = np.reshape(p3[0,:],X.shape)
  Y = np.reshape(p3[1,:],Y.shape)
  Z = np.reshape(p3[2,:],Z.shape)

  return X,Y,Z,C

def plotTangentSpace(R):
  """ plot the tangent space """
  X,Y = np.meshgrid(np.linspace(-1.,1.0,100),np.linspace(-1.,1.0,100))
  Z = np.ones(X.shape)
  p3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()].T
  p3 = R.dot(p3)
  X = np.reshape(p3[0,:],X.shape)
  Y = np.reshape(p3[1,:],Y.shape)
  Z = np.reshape(p3[2,:],Z.shape)
  return X,Y,Z

def plotPlaneXY(R,t,color,op=1.,mesh=True):
  X,Y = np.meshgrid(np.linspace(-1.,1.0,10),np.linspace(-1.,1.0,10))
  Z = np.zeros(X.shape)
  p3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()].T
  p3 = R.dot(p3) + t
  X = np.reshape(p3[0,:],X.shape)
  Y = np.reshape(p3[1,:],Y.shape)
  Z = np.reshape(p3[2,:],Z.shape)
  if mesh:
    mlab.mesh(X,Y,Z,color=color, opacity=op)
  else:
    mlab.points3d(X.ravel(),Y.ravel(),Z.ravel(),mode='sphere',
        color=color,scale_factor=0.1)
  return X,Y,Z

def plotPlaneXZ(R,t,color,op=1.,mesh=True):
  X,Z = np.meshgrid(np.linspace(-1.,1.0,10),np.linspace(-1.,1.0,10))
  Y = np.zeros(X.shape)
  p3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()].T
  p3 = R.dot(p3) + t
  X = np.reshape(p3[0,:],X.shape)
  Y = np.reshape(p3[1,:],Y.shape)
  Z = np.reshape(p3[2,:],Z.shape)
  if mesh:
    mlab.mesh(X,Y,Z,color=color, opacity=op)
  else:
    mlab.points3d(X.ravel(),Y.ravel(),Z.ravel(),mode='sphere',
        color=color,scale_factor=0.1)
  return X,Y,Z

def plotPlaneYZ(R,t,color,op=1.,mesh=True):
  Y,Z = np.meshgrid(np.linspace(-1.,1.0,10),np.linspace(-1.,1.0,10))
  X = np.zeros(Y.shape)
  p3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()].T
  p3 = R.dot(p3) + t
  X = np.reshape(p3[0,:],X.shape)
  Y = np.reshape(p3[1,:],Y.shape)
  Z = np.reshape(p3[2,:],Z.shape)
  if mesh:
    mlab.mesh(X,Y,Z,color=color, opacity=op)
  else:
    mlab.points3d(X.ravel(),Y.ravel(),Z.ravel(),mode='sphere',
        color=color,scale_factor=0.1)
  return X,Y,Z

def plotBox(R,t,colors):
#  colors = [(1.0,0.0,0.0),(1.0,0.5,0.5),(0.0,1.0,0.0),(0.5,1.0,0.5),(0.0,0.0,1.0),(0.5,0.5,1.0)]
  tp = R.dot(np.array([[0],[0],[-1]])) +t
  plotPlaneXY(R,tp,colors[0])

  tp = R.dot(np.array([[0],[0],[1]])) +t
  plotPlaneXY(R,tp,colors[1])

  tp = R.dot(np.array([[0],[-1],[0]])) +t
  plotPlaneXZ(R,tp,colors[2])

  tp = R.dot(np.array([[0],[1],[0]])) +t
  plotPlaneXZ(R,tp,colors[3])

  tp = R.dot(np.array([[-1],[0],[0]])) +t
  plotPlaneYZ(R,tp,colors[4])

  tp = R.dot(np.array([[1],[0],[0]])) +t
  plotPlaneYZ(R,tp,colors[5])

#
#R0 = np.eye(3)
#theta = np.pi/2.
#Rx = np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
#theta = np.pi/2.
#Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
#Rs = [R0,Rx,Ry]
#
#
#R = R0
#colors = colorScheme('label')
#print colors
#
#figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
#tp = R.dot(np.array([[0],[0],[ 1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
#tp = R.dot(np.array([[0],[ 1.0],[0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['green'])
#tp = R.dot(np.array([[ 1.0],[0],[0]]))
#plotPlaneYZ(R,tp,colorScheme('labelMap')['red'])
#
## old plane segmentation
##figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
##tp = R.dot(np.array([[0],[0],[-1.3]])) 
##plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
##tp = R.dot(np.array([[0],[-1.3],[0]]))
##plotPlaneXZ(R,tp,colorScheme('labelMap')['orange'])
##tp = R.dot(np.array([[-1.3],[0],[0]]))
##plotPlaneYZ(R,tp,(0.6,0.6,0.6))
#
## directional segmentation
#figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
#tp = R.dot(np.array([[0],[0],[1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
#tp = R.dot(np.array([[0],[1.0],[0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['orange'])
#tp = R.dot(np.array([[1.0],[0],[0]]))
#plotPlaneYZ(R,tp,(0.6,0.6,0.6))
#tp = R.dot(np.array([[2.0],[0],[-1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
#tp = R.dot(np.array([[0],[2.0],[-1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
#tp = R.dot(np.array([[2.0],[2.0],[-1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
#tp = R.dot(np.array([[0.],[-1.0],[2.0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['orange'])
#tp = R.dot(np.array([[-1.0],[0],[2.0]]))
#plotPlaneYZ(R,tp,(0.6,0.6,0.6))
#tp = R.dot(np.array([[2.0],[-1.0],[-0.0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['orange'])
#tp = R.dot(np.array([[-1.0],[2.0],[-0.0]]))
#plotPlaneYZ(R,tp,(0.6,0.6,0.6))
#tp = R.dot(np.array([[2.0],[-1.0],[ 2.0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['orange'])
#tp = R.dot(np.array([[-1.0],[2.0],[ 2.0]]))
#plotPlaneYZ(R,tp,(0.6,0.6,0.6))
#
## directional segmentation
#figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
#tp = R.dot(np.array([[0],[0],[1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['blue'])
#tp = R.dot(np.array([[0],[1.0],[0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['red'])
#tp = R.dot(np.array([[1.0],[0],[0]]))
#plotPlaneYZ(R,tp,(0.3,0.3,0.3))
#tp = R.dot(np.array([[2.0],[0],[-1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
#tp = R.dot(np.array([[0],[2.0],[-1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
#tp = R.dot(np.array([[2.0],[2.0],[-1.0]])) 
#plotPlaneXY(R,tp,colorScheme('labelMap')['turquoise'])
#tp = R.dot(np.array([[0.],[-1.0],[2.0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['orange'])
#tp = R.dot(np.array([[-1.0],[0],[2.0]]))
#plotPlaneYZ(R,tp,(0.6,0.6,0.6))
#tp = R.dot(np.array([[2.0],[-1.0],[-0.0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['orange'])
#tp = R.dot(np.array([[-1.0],[2.0],[-0.0]]))
#plotPlaneYZ(R,tp,(0.6,0.6,0.6))
#tp = R.dot(np.array([[2.0],[-1.0],[ 2.0]]))
#plotPlaneXZ(R,tp,colorScheme('labelMap')['orange'])
#tp = R.dot(np.array([[-1.0],[2.0],[ 2.0]]))
#plotPlaneYZ(R,tp,(0.6,0.6,0.6))
#
#mlab.show(stop=True)
#
#colors = [(0.6,0.6,0.6) for i in range(6)]
#figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
#tp = R.dot(np.array([[0],[0],[-1.3]])) 
#plotPlaneXY(R,tp,colors[0],mesh=False)
#tp = R.dot(np.array([[0],[-1.3],[0]]))
#plotPlaneXZ(R,tp,colors[2],mesh=False)
#tp = R.dot(np.array([[-1.3],[0],[0]]))
#plotPlaneYZ(R,tp,colors[4],mesh=False)
#
#color=(232/255.0,65/255.0,32/255.0)
#colorAxis = [(1.0,0.0,0.0),(1.0,0.0,0.0),(0.0,1.0,0.0),(0.0,1.0,0.0),(0.0,0.0,1.0),(0.0,0.0,1.0)]
#
#colorGray = [(0.6,0.6,0.6) for i in range(6)]
#colorOrange = [colorScheme('labelMap')['orange'] for i in range(6)]
#colorTurq = [colorScheme('labelMap')['turquoise'] for i in range(6)]
#
## red
#R = R0
#figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
#plotBox(R,np.array([[0],[0],[0]]),colorGray)
#plotBox(R,np.array([[2.5],[0],[0]]),colorGray)
##plotBox(R,np.array([[-2.5],[0],[0]]),colorAxis)
#
#plotBox(R,np.array([[0],[2.5],[0]]),colorGray)
#plotBox(R,np.array([[2.5],[2.5],[0]]),colorGray)
##plotBox(R,np.array([[-2.5],[2.5],[0]]),colorAxis)
#
##plotBox(R,np.array([[0],[-2.5],[0]]),colorAxis)
##plotBox(R,np.array([[2.5],[-2.5],[0]]),colorAxis)
##plotBox(R,np.array([[-2.5],[-2.5],[0]]),colorAxis)
#
#theta = np.pi/4.
#Rz = np.array([[np.cos(theta),np.sin(theta),0],
#               [-np.sin(theta),np.cos(theta),0],
#               [0,0,1]])
#
#figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
#plotBox(R,np.array([[0],[0],[0]]),colorGray)
#plotBox(R,np.array([[3.],[0],[0]]),colorGray)
##plotBox(R,np.array([[-3.],[0],[0]]),colorAxis)
#
#plotBox(Rz,np.array([[0],[3.],[0]]),colorOrange)
#plotBox(Rz,np.array([[3.],[3.],[0]]),colorOrange)
##plotBox(Rz,np.array([[-3.],[3.],[0]]),colorAxis)
#
##plotBox(Rz,np.array([[0],[-3.],[0]]),colorAxis)
##plotBox(Rz,np.array([[3.],[-3.],[0]]),colorAxis)
##plotBox(Rz,np.array([[-3.],[-3.],[0]]),colorAxis)
#
#theta = np.pi/4.
#Rz = np.array([[np.cos(theta),np.sin(theta),0],
#               [-np.sin(theta),np.cos(theta),0],
#               [0,0,1]])
#Ry = np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
#
#figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
#plotBox(R,np.array([[0],[0],[0]]),colorGray)
#plotBox(Ry,np.array([[3.],[0],[0]]),colorTurq)
##plotBox(R,np.array([[-3.],[0],[0]]),colorAxis)
#
#plotBox(Rz,np.array([[0],[3.],[0]]),colorOrange)
#plotBox(Ry,np.array([[3.],[3.],[0]]),colorTurq)
##plotBox(Rz,np.array([[-3.],[3.],[0]]),colorAxis)
#
##plotBox(Ry,np.array([[0],[-3.],[0]]),colorAxis)
##plotBox(Ry,np.array([[3.],[-3.],[0]]),colorAxis)
##plotBox(Ry,np.array([[-3.],[-3.],[0]]),colorAxis)
#
#
#
#mlab.show()
#
##  mlab.savefig('./MF_sphere_Gaussian_rot{}.png'.format(j),figure=figm,size=(1200,1200))
