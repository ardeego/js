#!/usr/bin/env python

import os, time, re
from subprocess import Popen
import socket               # Import socket module
import threading, time
import multiprocessing
import argparse

class WorkThread(threading.Thread):
  def __init__(self,name,cmd): 
    threading.Thread.__init__(self)
    self.cmd = cmd
    self.name = name
  def run(self):
    print '\x1b[32m------- #{}: {} \x1b[0m'.format(self.name,command)
    proc = Popen(self.cmd,shell=True)                                 
    proc.wait()                                                             
#    print '-------'
    
def appendCmd(threads,cmd):
  appended = False
  while not appended:
    for i in range(len(threads)):
      if threads[i] is None or not threads[i].isAlive():
        threads[i] = WorkThread(str(i),cmd)
        threads[i].start()
        appended = True
        break
    time.sleep(1) 

parser = argparse.ArgumentParser(description = 'server for command execution')      
parser.add_argument('-p','--port', type=int, default=1235, help='port number under which communications are accepted')     
parser.add_argument('-c','--cores', type=int, default=1, help='number of cores per cmd')
args = parser.parse_args()     

s = socket.socket()         # Create a socket object
host = socket.gethostname() # Get local machine name
port = args.port            # Reserve a port for your service.
s.bind((host, port))        # Bind to the port

nThreads = multiprocessing.cpu_count()/args.cores
print '# cores: {} -> # threads: {}'.format(multiprocessing.cpu_count(),nThreads)

threads = [None for i in range(nThreads)]

s.listen(5)                 # Now wait for client connection.
while True:
  print 'Waiting for a connection'
  c, addr = s.accept()     # Establish connection with client.
#  print 'Got connection from', addr
  command = ""
  try:
    # Receive the data in small chunks and retransmit it
    while True:
      command += c.recv(1024)
      if not re.search('<EOM>',command) is None:
        command = re.sub('<EOM>','',command)
        print 'Done receiving from {}'.format(addr)
        appendCmd(threads,command) 
#        print 'Executed - sending ACK'
        c.sendall('ACK')
        break
  finally:
    # Clean up the connection
    c.close()

