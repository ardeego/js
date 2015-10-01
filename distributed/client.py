# Copyright (c) 2015, Julian Straub <jstraub@csail.mit.edu>
# Licensed under the MIT license. See the license file LICENSE.
#!/usr/bin/env python

from socket import error as socket_error
import threading, time
import argparse

def shift(seq, n):
  n = n % len(seq)
  return seq[n:] + seq[:n]

class SendThread(threading.Thread):
  def __init__(self,host,port,cmd): 
    threading.Thread.__init__(self)
    self.host = host
    self.port = port
    self.cmd = cmd
  def connect(self):
    self.socket = socket.socket()         # Create a socket object
    try:
      self.socket.connect((self.host, self.port))
    except socket_error as serr:
      print '{}:{} {}'.format(self.host,self.port,serr)
      return False
      self.socket = None
    return True
  def run(self):  
    if self.socket is None:
      return
    try:
      print '{}:{} sending "{}"'.format(self.host,self.port,self.cmd)
      # Send data
      self.socket.sendall(self.cmd+'<EOM>')
        
      reply = ''
      while not reply == 'ACK':
        reply = self.socket.recv(255)
#        print 'received "%s"' % reply
    finally:
      print '{}:{} done "{}"'.format(self.host,self.port,self.cmd)
      self.socket.close()

parser = argparse.ArgumentParser(description = 'server for command execution')      
parser.add_argument('-p','--port', type=int, default=1235, help='port number under which communications are accepted')     
args = parser.parse_args()     
port = args.port                # Reserve a port for your service.

cmdFilePath = './harderCmdList.txt'
cmdFilePath = 'cmdList.txt'
hosts = ['jones','redbean']
hosts = ['flan']
hosts = ['flan','jones','waffle']
hosts = ['flan','waffle']
hosts = ['flan','waffle','puddin','redbean','strudel','froyo','pastelito']
hosts = ['flan','waffle','puddin','redbean','strudel','froyo']
hosts = ['flan','waffle','puddin','redbean','froyo']
hosts = ['tulumba','waffle','flan']
hosts = ['flan','puddin','redbean','froyo']

#print s.recv(1024)
#s.close                     # Close the socket when done
threads = dict()
for host in hosts:
  threads[host] = None

fin = open(cmdFilePath,'rw')
while True:
  rawCmd = fin.readline()[:-1]
  if rawCmd == '':
    print 'DONE with all cmds in {} -- some may still process'.format(cmdFilePath)
    break

#  host = socket.gethostname() # Get local machine name
  sent = False
  while not sent:
    for host in hosts:
      if threads[host] is None or not threads[host].isAlive():
        threads[host] = SendThread(host,port,rawCmd)
        if threads[host].connect():
          threads[host].start()
          sent = True
          break
    time.sleep(1)
  hosts = shift(hosts,1)
  print hosts

fin.close()
