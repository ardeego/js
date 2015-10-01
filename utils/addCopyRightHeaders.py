#!/usr/bin/env python

# Copyright (c) 2013, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the GPLv3 license. See the license file LICENSE.
#
# If this code is used, the following should be cited:
# [1] Fast Relocalization For Visual Odometry Using Binary Features (J.
# Straub, S. Hilsenbeck, G. Schroth, R. Huitl, A. Moeller, E. Steinbach), In
# IEEE International Conference on Image Processing (ICIP), 2013


import os, re
import copy
import subprocess as subp

pathToHeaderSnipped = {'.hpp':'./copyrightSnipped_GPLv3_cpp.txt',
    '.h':'./copyrightSnipped_GPLv3_cpp.txt',
    '.c':'./copyrightSnipped_GPLv3_cpp.txt',
    '.cpp':'./copyrightSnipped_GPLv3_cpp.txt',
    '.py':'./copyrightSnipped_GPLv3_py.txt'}

pathToHeaderSnipped = {'.hpp':'./copyrightSnipped_cpp.txt',
    '.h':'./copyrightSnipped_cpp.txt',
    '.c':'./copyrightSnipped_cpp.txt',
    '.cpp':'./copyrightSnipped_cpp.txt',
    '.cu':'./copyrightSnipped_cpp.txt',
    '.py':'./copyrightSnipped_py.txt'}

fileNames=[]
for subdir, dirs, files in os.walk('./'):          
  for f in files:                                                     
    if os.path.splitext(f)[1] in ['.hpp','.cpp','.h','.c','.py','.cu']: 
      print os.path.join(subdir,f)                                    
      fileNames.append(os.path.join(subdir,f))  

print fileNames
for fileName in fileNames:
  _,fileType = os.path.splitext(fileName)
  with open(fileName,'r') as f:
    txt = f.readlines()
    pos=[]
    if fileType in ['.hpp','.cpp','.h','.c','.cu']:
      for i,line in enumerate(txt):
        if re.search('\*\/',line):
          #pos.append(i)
          txt[i] =re.sub('\*\/','',txt[i])
          break
        elif re.search(' \*',line) or re.search('\/\*',line):
          pos.append(i)
          if len(pos)>1:
            if pos[-1]-pos[-2] > 1:
              pos = pos[:-1]
              break
    elif fileType in ['.py']:
      for i,line in enumerate(txt):
        if re.search('#',line) and not re.search('#!',line):
          pos.append(i)
          if len(pos)>1:
            if pos[-1]-pos[-2] > 1:
              pos = pos[:-1]
              break

    if len(pos)>=2:
#      print txt[max(0,min(pos)-1)][:-1]
      for i in pos:
        print '\x1b[33m{}\x1b[0m'.format(txt[i][:-1])
#      print txt[max(pos)+1][:-1]
    
    with open(pathToHeaderSnipped[fileType], 'r') as fSnip:
      newHeader = fSnip.readlines()
    # check if the header is already the propper header
    same = True
    for j,i in enumerate(pos):
      same = same and (txt[i][:-1] == newHeader[j][:-1])
    if same:
      print '{} already propper header'.format(fileName)
      continue

    for i, line in enumerate(txt):
      if not i in pos:
        newHeader.append(line)
  
  print 'in {}:'.format(fileName)
  doIt=raw_input('replace header? (y/n/c)') 
  if doIt == 'y':
    with open(fileName,'w') as f:
      f.writelines(newHeader)
    print 'replaced headers'
  elif doIt == 'c':
    print 'oppening in vim'
    subp.call(['vim',fileName])
  else:
    print 'not replacing headers'

