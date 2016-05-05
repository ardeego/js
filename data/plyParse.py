import numpy as np
import re

import mayavi.mlab as mlab

class PlyParse:
  def __init__(self):
    self.pts = None
    self.n = None 
    self.rgb = None
    self.face = None
  def parse(self,path):
    f = open(path)
    line = ''
    elem = ''
    self.elements = dict()
    # parse the header
    while not line == 'end_header':
      line = f.readline()[:-1]
      lineSplit = line.split(' ')
    
      if not re.search('element ',line) is None:
        elem = lineSplit[1]
        nElem = int(lineSplit[2])
        self.elements[elem] = {'N':nElem, 'props':[]}
    
      if not re.search('property ',line) is None:
        if not elem == '':
          typ = lineSplit[1]
          name = ' '.join(lineSplit[2::])
          self.elements[elem]['props'].append([typ,name])
#          print elem+': type='+typ+' name='+name
        else:
          print 'not sure which element this property belongs to'
          raise NotImplementedError
    for elem in ['vertex','face']: #self.elements.keys():
      if elem == 'vertex' and elem in self.elements.keys():
        data = np.zeros((self.elements[elem]['N'],len(self.elements[elem]['props'])))
        for i in range(self.elements[elem]['N']):
          line = f.readline()[:-1]
          data[i,:] = np.fromstring(line,sep=' ') 
        self.pts = np.zeros((3,self.elements[elem]['N']))
        self.n = np.zeros((3,self.elements[elem]['N']))
        self.rgb = np.zeros((3,self.elements[elem]['N']))
#        print self.n.shape
#        print self.n.strides
        for j,typeName in enumerate(self.elements[elem]['props']):
          if typeName[1] == 'x':
            self.pts[0,:] = data[:,j]
          elif typeName[1] == 'y':
            self.pts[1,:] = data[:,j]
          elif typeName[1] == 'z':
            self.pts[2,:] = data[:,j]
          elif typeName[1] == 'nx':
            self.n[0,:] = data[:,j]
          elif typeName[1] == 'ny':
            self.n[1,:] = data[:,j]
          elif typeName[1] == 'nz':
            self.n[2,:] = data[:,j]
          elif typeName[1] == 'red':
            self.rgb[0,:] = data[:,j].astype(np.uint8)
          elif typeName[1] == 'blue':
            self.rgb[1,:] = data[:,j].astype(np.uint8)
          elif typeName[1] == 'green':
            self.rgb[2,:] = data[:,j].astype(np.uint8)
        self.pts = self.pts.T
        self.n = self.n.T
        self.rgb = self.rgb.T
      elif elem == 'face' and elem in self.elements.keys():
        # will only read triangle indices and ignore texture
        self.face = np.zeros((self.elements[elem]['N'],3))
        for i in range(self.elements[elem]['N']):
          line = f.readline()[:-1].split(' ')
          self.face[i,0] = float(line[1])
          self.face[i,1] = float(line[2])
          self.face[i,2] = float(line[3])
  def write(self,path):
    self.elements=dict()
    self.elements['vertex'] = {'N':self.pts.shape[0],'props':
        [['float','x'],['float','y'],['float','z'],
         ['float','nx'],['float','ny'],['float','nz'],
         ['uchar','red'],['uchar','green'],['uchar','blue']]}
    if not self.face is None:
      self.elements['face']= {'N':self.face.shape[0],'props':
        [['list','uchar int vertex_indices']]}
    f = open(path,'w')
    f.write('ply\nformat ascii 1.0\ncomment PlyPrase generated\n')
    for elem in self.elements.keys():
      f.write('element {} {}\n'.format(elem,self.elements[elem]['N']))
      for typeName in self.elements[elem]['props']:
        f.write('property {} {}\n'.format(typeName[0],typeName[1]))
    f.write('end_header\n')
    # actual data
    for elem in ['vertex','face']:
      if elem == 'vertex':
        data = np.c_[self.pts,self.n,self.rgb]
        print data.shape
        np.savetxt(f, data, fmt='%.5f %.5f %.5f %.5f %.5f %.5f %u %u %u')
      elif elem == 'face' and not self.face is None:
        data = np.c_[np.ones((self.face.shape[0],1))*3,self.face]
        np.savetxt(f, data, fmt='%u %d %d %d')
  def getPc(self):
    ''' return point cloud from ply file ''' 
    return self.pts
  def plotPc(self):
    pc = self.getPc()
    mlab.figure(bgcolor=(1,1,1))
    mlab.points3d(pc[:,0],pc[:,1],pc[:,2],mode='point',color=(0.3,0.3,0.3))
  def getNormals(self):
    ''' return normals from ply file ''' 
    for i in range(self.n.shape[0]):
      if np.abs(np.sqrt((self.n[i,:]**2).sum())-1.0) > 1e-6 or np.isnan( np.sqrt((self.n[i,:]**2).sum())):
#        print np.sqrt((self.n[i,:]**2).sum()), self.n[i,:]
        self.n[i,:] = [1.,0.,0.]
      else:
        self.n[i,:] /= np.sqrt((self.n[i,:]**2).sum())
    return self.n
  def plotNormals(self):
    n = self.getNormals()
    mlab.figure(bgcolor=(1,1,1))
    mlab.points3d(n[:,0],n[:,1],n[:,2],mode='point',color=(0.3,0.3,0.3))
  def getRgb(s):
    ''' return coloring from ply file ''' 
    return self.rgb

if __name__ == '__main__':
  ply = PlyParse()
  ply.parse('/data/vision/fisher/data1/pointClouds/bosch/7c613881b02540bc9c6560fad2c8eed5/7c613881b02540bc9c6560fad2c8eed5.ply')
  ply.write('./test.ply')
  
  pc = ply.getPc()
  n = ply.getNormals()
      
  mlab.figure(bgcolor=(1,1,1))
  mlab.points3d(pc[:,0],pc[:,1],pc[:,2],mode='point',color=(0.3,0.3,0.3))
  
  mlab.figure(bgcolor=(1,1,1))
  mlab.points3d(n[:,0],n[:,1],n[:,2],mode='point',color=(0.3,0.3,0.3))
  mlab.show(stop=True)
