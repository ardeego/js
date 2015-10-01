
import numpy as np
import numpy.random as rnd
from numpy import sin,cos
from scipy.linalg import  norm, det, qr, expm,inv, solve
import scipy.linalg
import scipy.stats as stats
import math


cimport numpy as np
cimport cython # so later we can turn off bounds-checking 

ctypedef np.float64_t my_dtype 

def choice(np.ndarray[my_dtype, ndim=1] p, int n):
  ''' 
  draw n samples from a discrete pdf described by p
  '''
  cdef np.ndarray[my_dtype, ndim=1] P = np.zeros(p.size+1)
  P[1::] = np.cumsum(p)
  cdef np.ndarray[np.int32_t, ndim=1] ind = np.arange(P.size,dtype=np.int32)
  cdef np.ndarray[np.int32_t, ndim=1] c = np.zeros(n,dtype=np.int32)
  cdef np.ndarray[my_dtype, ndim=1] x = rnd.rand(n)
  for i in xrange(n):
    c[i] = np.max(ind[P<=x[i]]) 
  return c

def choiceMult(np.ndarray[my_dtype, ndim=2] p):
  ''' 
  draw 1 samples from each of the discrete pdfs described by
  the rows of p
  '''
  cdef np.ndarray[my_dtype, ndim=2] P = np.zeros((p.shape[0],p.shape[1]+1))
  P[:,1::] = np.cumsum(p,axis=1)
  cdef np.ndarray[np.int32_t, ndim=1] ind = np.arange(P.shape[1],dtype=np.int32)
  cdef np.ndarray[np.int32_t, ndim=1] c = np.zeros(P.shape[0],dtype=np.int32)
  cdef np.ndarray[my_dtype, ndim=1] x = rnd.rand(P.shape[0])
  for i in xrange(P.shape[0]):
    c[i] = np.max(ind[P[i,:]<=x[i]]) 
  return c


def sample_niw(mu_0,lmbda_0,kappa_0,nu_0):
    '''
    Returns a sample from the normal/inverse-wishart distribution, conjugate
    prior for (simultaneously) unknown mean and unknown covariance in a
    Gaussian likelihood model. Returns covariance.  '''
    # this is completely copied from Matlab's implementation, ignoring
    # the copyright. I'm sorry.
    # reference: p. 87 in Gelman's Bayesian Data Analysis

    # first sample Sigma ~ IW(lmbda_0^-1,nu_0)
    lmbda = sample_invwishart(lmbda_0,nu_0) # lmbda = np.linalg.inv(sample_wishart(np.linalg.inv(lmbda_0),nu_0))
    # then sample mu | Lambda ~ N(mu_0, Lambda/kappa_0)
    mu = np.random.multivariate_normal(mu_0,lmbda / kappa_0)

    return mu, lmbda

def sample_invwishart(lmbda,dof):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    n = lmbda.shape[0]
    chol = np.linalg.cholesky(lmbda)

    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(dof,n)
    else:
        x = np.diag(np.sqrt(stats.chi2.rvs(dof-(np.arange(n)))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T).T
    return np.dot(T,T.T)

def sample_wishart(sigma, dof):
    '''
    Returns a sample from the Wishart distn, conjugate prior for precision matrices.
    '''

    n = sigma.shape[0]
    chol = np.linalg.cholesky(sigma)

    # use matlab's heuristic for choosing between the two different sampling schemes
    if (dof <= 81+n) and (dof == round(dof)):
        # direct
        X = np.dot(chol,np.random.normal(size=(n,dof)))
    else:
        A = np.diag(np.sqrt(np.random.chisquare(dof - np.arange(0,n),size=n)))
        A[np.tri(n,k=-1,dtype=bool)] = np.random.normal(size=(n*(n-1)/2.))
        X = np.dot(chol,A)

    return np.dot(X,X.T)




