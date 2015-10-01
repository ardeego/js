import numpy as np 
import mayavi.mlab as mlab
from vpCluster.manifold.sphere import *
from js.utils.plot.colors import colorScheme

def plotvMF3D(mu,tau):
  T,P = np.meshgrid(np.linspace(-np.pi,np.pi,300), \
      np.linspace(0.,np.pi,150))
  X = np.cos(T)*np.sin(P)
  Y = np.sin(T)*np.sin(P)
  Z = np.cos(P)
  x = np.c_[X.ravel(),Y.ravel(),Z.ravel()]
  x = (x.T / np.sqrt((x**2).sum(axis=1)))

  pdf = np.exp(tau*mu.T.dot(x))
  pdf *= 1./pdf.sum()
  print mu.T.dot(x)
  C = np.reshape(pdf,X.shape)
#  X = X*(C+1.01)
#  Y = Y*(C+1.01)
#  Z = Z*(C+1.01)
  X = X*(1.001)
  Y = Y*(1.001)
  Z = Z*(1.001)
  return X,Y,Z,C

#M = Sphere(2)
#R0 = np.eye(3)
#
#mu = np.array([0.,0.,1.])
#
#R0 [:,2] *= -1
#R0n = np.copy(R0); R0n[:,2] *= -1
#Rs = [R0,R0n]
#
#for tau in [1.0,10.,100.]:
#    figm = mlab.figure(bgcolor=(1.0,1.0,1.0))
#    M.plot(figm,1.0,color=(1.,1.,1.))
#    i=1
#    X,Y,Z,C = plotvMF3D(mu,tau)
#    p3 = np.c_[X.ravel(), Y.ravel(), Z.ravel()].T
#    p3 = Rs[i].dot(p3)
#    X = np.reshape(p3[0,:],X.shape)
#    Y = np.reshape(p3[1,:],Y.shape)
#    Z = np.reshape(p3[2,:],Z.shape)
#    mlab.mesh(X,Y,Z,scalars=C, colormap='hot', opacity=0.5, figure=figm)
#      #mlab.points3d(X.ravel(),Y.ravel(),Z.ravel(),opacity=0.6, figure=figm)
#    #mlab.points3d([0],[0],[1],color=(0.,0.,1.),opacity=1., scale_factor=0.05,
#    #        figure=figm)
#mlab.show(stop=True)
