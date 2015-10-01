
import numpy as np
import mayavi.mlab as mlab
import ipdb

from vpCluster.manifold.sphere import Sphere
from js.utils.plot.colors import colorScheme

S2 = Sphere(3)
def plotvMF(mu,tau,pi,figm,color=(1.,0.0,0.0)):
  X,Y,Z = S2.mesh(1.)
  q = np.c_[X.ravel(),Y.ravel(),Z.ravel()].T
#  ipdb.set_trace()
  pdf = pi*np.exp(mu.T.dot(q)*tau)*tau/(4*np.pi*np.sinh(tau)) +1.
  X*=np.reshape(pdf,X.shape)
  Y*=np.reshape(pdf,X.shape)
  Z*=np.reshape(pdf,X.shape)

  mlab.mesh(X,Y,Z,color=color,opacity=0.3,figure=figm,mode='2dtriangle')
  mlab.plot3d([0,mu[0]],[0,mu[1]],[0,mu[2]],color=color,opacity=0.9,figure=figm)
#  mlab.mesh(X,Y,Z,scalars=np.reshape(pdf,X.shape),opacity=0.5,figure=figm)

def samplevMF(mu,tau,N):
  ''' 
  rejection sampling based vMF sampling; propose from uniform distribution on
  sphere 
  '''
  x= np.zeros((3,0))
  while x.shape[1] <N:
    #propose from a uniform distribution over the sphere
    q = np.random.randn(3,N*10)
    #/tau + np.resize(mu,(N,3)).T;
    q /= np.sqrt((q**2).sum(axis=0))
    # rejection sampling
    u = np.random.rand(N*10);
    pdf = np.exp(mu.T.dot(q)*tau)*tau/(4*np.pi*np.sinh(tau))
    M = np.exp(tau)*tau/(4*np.pi*np.sinh(tau))
    if x.shape[1] == 0:
      x = q[:, u < pdf/( M *4.*np.pi)]
    else:
      x = np.c_[x, q[:, u < pdf/( M *4.*np.pi)]]
#  pdf = np.exp(mu.T.dot(q)*tau)*tau/(4*np.pi*np.sinh(tau)) 
#  pdf = np.exp(mu.T.dot(q)*tau)*tau/(4*np.pi*np.sinh(tau)) 
  return x[:,:N]

def normalize(x):
  return x/np.sqrt((x**2).sum(axis=0))
def normed(x):
  return x/np.sqrt((x**2).sum(axis=0))
  
def plotSamplevMF(mu,tau,N,figm,color):
  q = samplevMF(mu,tau,N);
  mlab.points3d(q[0,:],q[1,:],q[2,:],figure=figm,
      scale_factor=0.03,mode='sphere',color=color)
  mlab.plot3d([0,mu[0]],[0,mu[1]],[0,mu[2]],
      color=color,opacity=0.9,figure=figm)

def plotvMF3D(mu,tau,colorOnly=True,maxAngle=np.pi):
  T,P = np.meshgrid(np.linspace(-np.pi,np.pi,300), \
      np.linspace(0.,maxAngle,150))
  X = np.cos(T)*np.sin(P)
  Y = np.sin(T)*np.sin(P)
  Z = np.cos(P)
  x = np.c_[X.ravel(),Y.ravel(),Z.ravel()]
  x = (x.T / np.sqrt((x**2).sum(axis=1)))

  pdf = np.exp(tau*mu.T.dot(x))
  pdf *= 1./pdf.sum()
  print mu.T.dot(x)
  C = np.reshape(pdf,X.shape)
  if not colorOnly:
    X = X*(C+1.01)
    Y = Y*(C+1.01)
    Z = Z*(C+1.01)
  else:
    X = X*(1.001)
    Y = Y*(1.001)
    Z = Z*(1.001)
  return X,Y,Z,C

#figm = mlab.figure(bgcolor=(1,1,1))
#S2.plotFanzy(figm,1)
#mu1 = np.array([1.,0,0,])
#mu2 = np.array([.0,1.0,0])
#mu3 = np.array([.0,0.0,1])
#plotvMF(mu1,10.0,.5,figm,color=(1.,0,0))
#plotvMF(mu2,4.0,.5,figm,color=(0.,1.0,0))
#plotvMF(mu3,40.0,.5,figm,color=(0.,.0,1))

#color1 = colorScheme('labelMap')['orange']
#color2 = colorScheme('labelMap')['green']
#color3 = colorScheme('labelMap')['blue']
#color4 = colorScheme('labelMap')['red']
#
#figm = mlab.figure(bgcolor=(1,1,1))
#S2.plotFanzy(figm,1)
#plotSamplevMF(normalize(np.array([1.,0.0,.0])),100.0,1000,figm,color1)
#plotSamplevMF(normalize(np.array([.0,1.0,0.0])),20.0,1000,figm,color2)
#plotSamplevMF(normalize(np.array([.0,.0,1.0])),40.0,1000,figm,color3)
#mlab.savefig('../results/DDPvMF_pictorial_0.png',figure=figm,size=(1200,1200))
#
#figm = mlab.figure(bgcolor=(1,1,1))
#S2.plotFanzy(figm,1)
#plotSamplevMF(normalize(np.array([1.,0.0,.0])),60.0,100,figm,color1)
#plotSamplevMF(normalize(np.array([.2,1.0,0.2])),20.0,1000,figm,color2)
#plotSamplevMF(normalize(np.array([.0,.4,1.0])),40.0,1000,figm,color3)
#mlab.savefig('../results/DDPvMF_pictorial_1.png',figure=figm,size=(1200,1200))
#
#figm = mlab.figure(bgcolor=(1,1,1))
#S2.plotFanzy(figm,1)
##plotSamplevMF(normalize(np.array([1.,0.0,.0])),100.0,1000,figm,color1)
#plotSamplevMF(normalize(np.array([.3,1.0,.2])),20.0,1000,figm,color2)
#plotSamplevMF(normalize(np.array([.0,.8,1.0])),40.0,1000,figm,color3)
#mlab.savefig('../results/DDPvMF_pictorial_2.png',figure=figm,size=(1200,1200))
#
#figm = mlab.figure(bgcolor=(1,1,1))
#S2.plotFanzy(figm,1)
#plotSamplevMF(normalize(np.array([1.,0.0,.2])),100.0,300,figm,color1)
#plotSamplevMF(normalize(np.array([.7,1.0,.2])),20.0,1000,figm,color2)
#plotSamplevMF(normalize(np.array([.0,1.,1.0])),40.0,1000,figm,color3)
#plotSamplevMF(normalize(np.array([.5,-0.4,1.0])),40.0,1000,figm,color4)
#mlab.savefig('../results/DDPvMF_pictorial_3.png',figure=figm,size=(1200,1200))
#
#
##figm = mlab.figure(bgcolor=(1,1,1))
##S2.plotFanzy(figm,1)
##plotSamplevMF(mu1,10.0,1000,figm,color1)
##plotSamplevMF(mu2,4.0,1000,figm,color2)
##plotSamplevMF(mu3,40.0,1000,figm,color3)
#
#mlab.show(stop=True)
#
#
#
#
