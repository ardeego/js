# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu> Licensed
# under the MIT license. See the license file LICENSE.
import numpy as np

def normed(x):
  # return unit L2 length vectors
  return x / np.sqrt((x**2).sum(axis=0)) 

def FitMlDPGMM(xs,lamb,it=20, ws=None):
  dpMeans = DPmeans(lamb)
  dpMeans.Compute(xs,it)
  Ns, mus, Ss = dpMeans.GetClusterParameters(xs,ws)
  pi = Ns/float(Ns.sum())
  return mus, Ss, pi

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
      print 'iteration {}:\tcost={:2.5f};\tK={}\tavg counts={:2.2f} range={} {}'.format(t,
          self.C[t],self.K,np.mean(self.N_),np.min(self.N_),np.max(self.N_))
      if self.C[t] >= self.C[t-1]:
        break;
  def GetClusterParameters(self,xs, ws=None):
    '''
    Compute the sufficient statistics in each cluster.
    Optionally weights for each data point can be passed in.
    '''
    N = xs.shape[1]
    if ws is None:
      # no weights passed in so use 1s
      ws = np.ones(N)
      Ns = self.N_.copy()
      mus = self.mus.copy()
    else:
      mus = np.zeros((3,self.K))
      Ns = np.zeros(self.K)
      for k in range(self.K):
        ids = self.zs == k
        Ns[k] = np.sum(ws[ids])
        mus[:,k] = np.sum(ws[ids]*xs[:,ids], axis=1)/Ns[k]
    Ss = []
    for k in range(self.K):
      ids = self.zs==k
#      if ids.sum() <= 0.1*N:
#        continue # sort out very small clusters because they tend to
#                 # make numeric difficulties 
      # regularize
      S = np.zeros((3,3))
      Stilde = np.zeros((3,3))
      for i,x in enumerate(xs[:,ids].T):
        S += ws[ids][i]*np.outer(x,x)
        Stilde += np.outer(x,x)
#        Ss.append(np.identity(3)*1e-12+S/(Ns[k]-1.)-(Ns[k]/(Ns[k]-1.))*np.outer(mus[:,k],mus[:,k]))
#        Ss.append(S/(Ns[k]-1.)-(Ns[k]/(Ns[k]-1.))*np.outer(mus[:,k],mus[:,k]))
      if self.N_[k] > 1:
        Ss.append((S-Ns[k]*np.outer(mus[:,k],mus[:,k]))/(Ns[k]-((ws[ids]**2).sum()/Ns[k])))
      else:
        Ss.append(np.identity(3)*1e6)
      if False:
        Nsk = ids.sum()
        Stilde = Stilde/(Nsk-1.)-(Nsk/(Nsk-1.))*np.outer(self.mus[:,k],self.mus[:,k])
        print '---', Ss[-1]
        print Stilde
        print Ss[-1]-Stilde
        
#      else:
#        print "low number of data points in cluster ",k, ": ",Ns[k] 
#        Ss.append(np.identity(3)*1e6)
#      print Ns[k], mus[:,k]
#      print Ss[k]

    # delete small clusters
    idKeep = self.N_ > max(10,0.001*N)
    print "threshold for cluster removal {}".format(max(10,0.001*N))
    print "removing {} of {} clusters".format(self.K - idKeep.sum(),self.K)
    K = idKeep.sum()
    mus = mus[:,idKeep]
    Ns = Ns[idKeep]
    Ss = [S for i,S in enumerate(Ss) if idKeep[i]]
    return Ns, mus, Ss

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
