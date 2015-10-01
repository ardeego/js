
import time


class StopWatch(object):
  '''
  class for generall purpose timing of functions
  usage:
  sw = StopWatch(True)
  ... computation
  sw.toctic()
  ... more computation
  sw.toc()
 
  '''
  def __init__(s,verbose=True):
    s.verbose = verbose
    s.t0 = time.time()
  def tic(s):
    s.t0 = time.time()
  def toc(s,description='StopWatch'):
    dt = time.time()-s.t0
    if s.verbose:
      print '{}: dt = {}ms'.format(description,dt*1000.0)
    return dt
  def toctic(s,description='StopWatch'):
    dt = s.toc(description)
    s.tic()
    return dt
