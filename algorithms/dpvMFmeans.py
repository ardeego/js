import numpy as np

def norm(x):
  return np.sqrt((x**2).sum())
def normed(x):
  return x/norm(x)

class DPvMF(object):
  def __init__(self, lamb):
    self.lamb = lamb
    self.mus = np.zeros((0,3))
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
    self.zs = np.r_[self.zs, np.array([self.ComputeLabel(q)])]
    return self.zs[-1]
  def ComputeLabel(self, q):
    ''' Computes the label for a given observation '''
    if self.mus.shape[0] == 0:
      self.mus = np.copy(q)[np.newaxis,:]
      return 0
    dots = self.mus.dot(q)
    if dots.max() > self.lamb:
      return np.argmax(dots)
    else:
      self.mus = np.r_[self.mus, q[np.newaxis,:]]
      return dots.size
  def UpdateMeans(self):
    ''' Update means of current observation set '''
    for k in range(self.mus.shape[0]):
      self.mus[k,:] = normed(self.qs[self.zs==k,:].sum(axis=0))
  def UpdateLabels(self):
    ''' Recompute labels '''
    for i in range(self.qs.shape[0]):
      self.zs[i] = self.ComputeLabel(self.qs[i,:])
