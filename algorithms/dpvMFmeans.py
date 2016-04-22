import numpy as np

def norm(x):
  return np.sqrt((x**2).sum())
def normed(x):
  return x/norm(x)

def MLestTau(mu, xSum, count):
  tau = 1.0;                                                           
  prevTau = 0.;                                                        
  eps = 1e-8;                                                          
  R = norm(xSum)/count;                                               
  while np.abs(tau - prevTau) > eps:
    inv_tanh_tau = 1./np.tanh(tau)
    inv_tau = 1./tau
    f = -inv_tau + inv_tanh_tau - R
    df = inv_tau*inv_tau - inv_tanh_tau*inv_tanh_tau + 1.
    prevTau = tau
    tau -= f/df
  return tau

def FitMlDPvMFMM(ns, lamb, it=20, mu=None):
  dpMeans = DPvMFmeans(lamb, mu)
  for i,n in enumerate(ns):
    dpMeans.AddObservation(n)
  f, fPrev = 0,0
  for i in range(it):
    dpMeans.UpdateMeans()
    fPrev = f
    f = dpMeans.UpdateLabels()
    K = dpMeans.mus.shape[0]
    Ns = np.bincount(dpMeans.zs,minlength=K).astype(np.float)
    print "@{}: cost={:4f}\tK={}\tavg Ns: {:.2f}, range Ns: {} {}".format(i, 
        f, K, np.mean(Ns), np.min(Ns), np.max(Ns))
    if f-fPrev <= 0.:
      break
  mu = dpMeans.mus.copy()
  K = mu.shape[0]
  tau = dpMeans.GetTaus()
  pi = np.bincount(dpMeans.zs,minlength=K).astype(np.float)
  pi /= float(pi.sum())
  return mu, tau, pi

def logSumExp(a,pmFactors=None):
  aMax = np.max(a)
  if pmFactors is None:
    return np.log((np.exp(a-aMax)).sum())+aMax
  else:
    return np.log((pmFactors*np.exp(a-aMax)).sum())+aMax

def log2SinhOverZ(z):
  if z > 10:
    logF =-np.log(z)+z
  else:
    logF =np.log(2.+z**2/3.+z**4/50.+z**6/2520.)
  return logF

def logPdfDPvMFMM(mu,tau,pi,x):
  K = tau.size
  logPdfs = np.zeros(K)
  for k in range(K):
    logPdfs[k] = np.log(pi[k]) + mu[k,:].dot(x)*tau[k]\
      -np.log(4.*np.pi) -log2SinhOverZ(tau[k])
  return logSumExp(logPdfs)

class DPvMFmeans(object):
  def __init__(self, lamb, mu=None):
    self.lamb = lamb
    if mu is None:
      self.mus = np.zeros((0,3))
    else:
      self.mus = mu
    self.qs = np.zeros((0,3))
    self.zs = np.zeros(0, dtype=np.int)
  def AddObservation(self, q):
    ''' 
    Add an observation q to the set of observations. Assigns a label to
    that new observation and returns the label.
    '''
#    print self.qs.shape, q.shape, q[np.newaxis,:].shape
#    print self.zs.shape, z.shape, z[np.newaxis,:].shape
    self.qs = np.r_[self.qs, q[np.newaxis,:]]
    self.zs = np.r_[self.zs, np.array([self.ComputeLabel(q)[0]])]
    return self.zs[-1]
  def ComputeLabel(self, q):
    ''' Computes the label for a given observation '''
    if self.mus.shape[0] == 0:
      self.mus = np.copy(q)[np.newaxis,:]
      return 0, self.lamb
    dots = self.mus.dot(q)
    if dots.max() > self.lamb:
      return np.argmax(dots), dots.max() 
    else:
      self.mus = np.r_[self.mus, q[np.newaxis,:]]
      return dots.size, self.lamb
  def UpdateMeans(self):
    ''' Update means of current observation set '''
    for k in range(self.mus.shape[0]):
      self.mus[k,:] = normed(self.qs[self.zs==k,:].sum(axis=0))
    #TODO: remove single data point clusters
  def UpdateLabels(self):
    ''' Recompute labels and cost function '''
    f = 0.
    for i in range(self.qs.shape[0]):
      self.zs[i], df = self.ComputeLabel(self.qs[i,:])
      f += df
    return f
  def GetTaus(self):
    K = self.mus.shape[0]
    tau = np.zeros(K)
    for k in range(K):
      Nk = (self.zs==k).sum()
      if Nk == 1:
        tau[k] = 1e-12
      else:
        tau[k] = MLestTau(self.mus[k,:],self.qs[self.zs==k,:].sum(axis=0), Nk)
    return tau
