# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.
import numpy as np

def normed(x):
  # return unit L2 length vectors
  return x / np.sqrt((x**2).sum(axis=0)) 

def FitMlDPGMM(xs,lamb,it=20):
  dpMeans = DPmeans(lamb)
  dpMeans.Compute(xs,it)
  mu = dpMeans.mus.copy()
  K = dpMeans.K
  Ss = dpMeans.GetSigmas(xs)
  pi = np.bincount(dpMeans.zs,minlength=K).astype(np.float)
  pi /= float(pi.sum())
  return mu, Ss, pi

class DPmeans(object):
  def __init__(self, lamb):
    self.lamb = lamb
  def RemoveCluster(self,k):
    self.mus = np.concatenate((self.mus[:,:k],self.mus[:,k+1::]),axis=1)
    self.N_ = np.concatenate((self.N_[:k],self.N_[k+1::]),axis=1)
    self.zs[self.zs>k] -= 1
    self.K -= 1

  def LabelAssign(self,i,x_i):
    z_i = np.argmin(np.r_[np.sqrt(((x_i - self.mus)**2).sum(axis=0)),np.array([self.lamb])])
    # check if this was the last datapoint in a cluster if so do not
    # assign to it again
    if self.N_[self.zs[i]] == 0 and z_i == self.zs[i]:
      self.RemoveCluster(self.zs[i])
      z_i = np.argmin(np.r_[np.sqrt(((x_i - self.mus)**2).sum(axis=0)),np.array([self.lamb])])
    # creata a new cluster if required
    if z_i == self.K:
      self.mus = np.concatenate((self.mus,x_i),axis=1)
      self.N_ = np.concatenate((self.N_,np.array([0])),axis=0)
      self.K += 1
    return z_i

  def Compute(self,x,Tmax=100):
    # init stuff
    print x.shape
    N = x.shape[1]
    self.K = 1
    self.zs = np.zeros(N,dtype=np.int) # labels for each data point
    self.mus = x[:,0][:,np.newaxis] # the first data point always creates a cluster
    self.N_ = np.bincount(self.zs,minlength=self.K) # counts per cluster
    self.C = np.zeros(Tmax) # cost function value
    self.C[0] = 1e6
    for t in range(1,Tmax):
      # label assignment
      for i in range(N):
        self.N_[self.zs[i]] -= 1 
        self.zs[i] = self.LabelAssign(i,x[:,i][:,np.newaxis])
        self.N_[self.zs[i]] += 1
      # centroid update
      for k in range(self.K-1,-1,-1):
        if self.N_[k] > 0:
          self.mus[:,k] = (x[:,self.zs==k].sum(axis=1))/self.N_[k]
        else:
          self.RemoveCluster(k)
      # eval cost function
      self.C[t] = np.array(\
          [np.sqrt(((x[:,i][:,np.newaxis]-self.mus[:,z_i][:,np.newaxis])**2).sum(axis=0)) for i,z_i in enumerate(self.zs)]).sum() \
          + self.K*self.lamb
      print 'iteration {}:\tcost={};\tcounts={}'.format(t,self.C[t], self.N_)
      if self.C[t] >= self.C[t-1]:
        break;
  def GetSigmas(self,xs):
    '''
    Compute the ML estimate of the covariances in each cluster
    '''
    Ss = []
    for k in range(self.K):
      # regularize
      S = np.identity(3)*0.01
      for x in xs[:,self.zs==k].T:
        S += np.outer(x,x)
      N = (self.zs==k).sum()
      if N > 1:
        Ss.append(S/(N-1.)-(N/(N-1.))*np.outer(self.mus[:,k],self.mus[:,k]))
      else:
        Ss.append(S)
    return Ss

if __name__=="__main__":
  # generate two noisy clusters
  N = 10
  x = np.concatenate(\
      ((np.random.randn(3,N/2).T*0.1+np.array([1,0,0])).T,\
       (np.random.randn(3,N/2).T*0.1+np.array([0,1,0])).T),axis=1)
  print x.shape
  # instantiate DP-means algorithm object
  dpmeans = DPmeans(lamb = 0.4)
  # compute clustering (maximum of 30 iterations)
  dpmeans.Compute(x,Tmax=30)
