
import math
import numpy as np
from scipy.special import gammaln, multigammaln
from scipy.linalg import inv, sqrtm

from sampling import sample_invwishart

class Dir(object):
  def __init__(s,alpha):
    s.alpha = alpha

  def logPdf(s,pi):
    return gammaln(np.sum(s.alpha)) - np.sum(gammaln(s.alpha)) + np.sum((s.alpha-1)*np.log(pi))

  def sample(s,n_k=None):
    if n_k is None:
      print 'n_k was None'
      print s.alpha
      pi = np.random.dirichlet(s.alpha.astype(np.float64))
    else:
      pi = np.random.dirichlet((s.alpha+n_k).astype(np.float64))
    return pi

  def posteriorFromCounts(s,n_k):
    return Dir(s.alpha+n_k)

  def posteriorFromIndicators(s,z):
    if z.size >0:
      n_k = np.bincount(z,minlength=s.alpha.size) 
      return s.posteriorFromCounts(n_k)
    else:
      return s.posteriorFromCounts(np.zeros(s.alpha.size))

  def logMarginalLikelihood(s,z):
    '''
    dirichlet multinomial
    '''
    if z.size >0:
      n_k = np.bincount(z,minlength=s.alpha.size) 
    else:
      n_k = np.zeros(s.alpha.size)
    #print n_k
    #print s.alpha
    logP = gammaln(np.sum(s.alpha)) - gammaln(np.sum(s.alpha+n_k))
    logP += np.sum(gammaln(s.alpha+n_k) - gammaln(s.alpha))
    return logP

  def logMarginalPosterior(s,z):
    '''
    probability of z=k | Z, alpha
    where Z is a set of observations
    '''
    n_k = np.bincount(z,minlength=s.alpha.size)
    logP_z = np.log(s.alpha+n_k)- np.log((s.alpha+n_k).sum()-1)
    return logP_z
  def marginalBeta(self,j):
    '''
    returns the marginal distribution when integrating over all but the jth pi 
    element
    '''
    if j >= self.alpha.size:
      raise ValueError
    return Beta(self.alpha[j],self.alpha.sum()-self.alpha[j])

class Cat(object):
  '''
  categorical distribution
  '''
  def __init__(self,pmf):
    self.pmf = pmf
  def sample(self):
    return int(np.where(np.random.multinomial(1, self.pmf))[0][0])


class Beta(object):
  def __init__(self,alpha,beta):
    self.alpha = alpha
    self.beta = beta
  
  def logPdf(self,pi):
    logP = (self.alpha-1.)*np.log(pi) + (self.beta-1.)*np.log(1.0-pi)
    logP += gammaln(self.alpha + self.beta)
    logP -= gammaln(self.alpha) + gammaln(self.beta)
    return logP
  
class IW(object):
  '''
  inverse wishart distribution
  '''
  def __init__(s,delta,nu):
    s.delta = delta
    s.nu = nu
    if s.nu < s.delta.shape[0]:
      raise ValueError(s.nu,s.delta.shape[0])

  def logPdf(s,S):
    d = s.delta.shape[0]
    logPdf = -.5*np.trace(s.delta.dot(inv(S)))
    (sign,logdetDelta) = np.linalg.slogdet(s.delta)
    (sign,logdetS) = np.linalg.slogdet(S)
    logPdf += logdetS*(-.5*(s.nu+1.+d))
    logPdf += logdetDelta*(.5*s.nu)
    logPdf -= np.log(2.0)*0.5*s.nu*d
    logPdf -= multigammaln(0.5*s.nu,d) 
    return logPdf

  def posterior(s, x):
    if x.size==0:
      return IW(s.delta,s.nu)
    else:
      n=x.shape[1]
      scatter = np.dot(x,x.T) 
      return IW(s.delta+scatter,s.nu+n)

  def sample(s):
    return sample_invwishart(s.delta,s.nu)

  def logMarginalLikelihood(s,x):
    d = s.delta.shape[0]
    n=x.shape[1]
    if not x.shape[0] == d:
      raise ValueError
    scatter = np.dot(x,x.T) 
    (sign,logdetD) = np.linalg.slogdet(s.delta)
    (sign,logdetDS) = np.linalg.slogdet(s.delta+scatter)

    logP = multigammaln((s.nu+n)*.5,d) - multigammaln(s.nu*.5,d)
    logP -= 0.5*(n*d)*np.log(np.pi)
    logP += s.nu*0.5 * logdetD
    logP -= (s.nu+n)*0.5 * logdetDS
     
    return logP

class NIW(object):
  '''
  normal inverse wishart
  '''
  def __init__(self, delta, nu, vartheta, lamb):
    self.iw = IW(delta, nu)
    self.vartheta = vartheta
    self.lamb = lamb
  def sample(self):
    Sigma = self.iw.sample()
    self.normal = Gaussian(self.vartheta, Sigma/self.lamb)
    mu = self.normal.sample()
    return Gaussian(mu,Sigma)

class Gaussian(object):
  def __init__(self, mu, Sigma):
    self.mu = mu
    self.Sigma = Sigma
    self.d = Sigma.shape[0]
    self.sqrtSigma = sqrtm(self.Sigma)
  def sample(self):
    return self.sqrtSigma.real.dot(np.random.normal(size=(self.d))) + self.mu
  def logPdf(self,x):
    if  not self.d == x.shape[0]:
      raise ValueError
    logP = -0.5*(np.linalg.solve(self.Sigma, x) * x).sum(axis=0)
    logP += -0.5*(self.d*math.log(2.*math.pi)+np.linalg.slogdet(self.Sigma)[1])
    return logP


if __name__ == "__main__":

  d = 3
  nu = 100
  Delta = np.eye(d)*nu
  print Delta
  iw = IW(Delta,nu) 
  print iw.sample()

  
