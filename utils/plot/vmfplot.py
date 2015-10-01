import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

from js.utils.plot.colors import colorScheme

def sampleVMF(mu, tau, n):
    edmax = tau/(4.0*np.pi*spec.iv(0, tau))*np.exp(tau)
    #rejection sample from uniform
    tmpsamps = -np.pi+2.0*np.pi*np.random.rand(n)
    heights = np.random.rand(n)*edmax
    hits = tau/(4.0*np.pi*spec.iv(0, tau))*np.exp(tau*np.cos(tmpsamps)) > heights
    nhits = np.sum(hits.astype(int))
    samps = tmpsamps[hits == True]
    n -= nhits
    while (n > 0):
        tmpsamps = -np.pi+2.0*np.pi*np.random.rand(n)
        heights = np.random.rand(n)*edmax
        hits = tau/(4.0*np.pi*spec.iv(0, tau))*np.exp(tau*np.cos(tmpsamps)) > heights
        nhits = np.sum(hits.astype(int))
        samps = np.append(samps, tmpsamps[hits == True])
        n -= nhits
    vecx = np.cos(samps)*mu[0] - np.sin(samps)*mu[1]
    vecy = np.sin(samps)*mu[0] + np.cos(samps)*mu[1]
    return np.vstack((vecx, vecy))

u = np.linspace(0, 2*np.pi, 1000)
x = np.sin(u)
y = np.cos(u)

mu = np.ones(2)/np.sqrt(2.0)
tau = np.array([0.1, 1.0, 10.0, 100.0])

d = mu[0]*x+mu[1]*y

ed = tau[:, np.newaxis]/(4.0*np.pi*spec.iv(0, tau[:, np.newaxis]))*np.exp(tau[:, np.newaxis]*d)
edmax = tau[:, np.newaxis]/(4.0*np.pi*spec.iv(0, tau[:, np.newaxis]))*np.exp(tau[:, np.newaxis])

for i in range(0, ed.shape[0]):
    fig = plt.figure(facecolor='white')
    ax = plt.axes(frameon=False)

    plt.plot(x, y, color='k', lw=5)
    plt.plot((1.0+ed[i, :]/edmax[i])*x, (1.0+ed[i, :]/edmax[i])*y, color=colorScheme('labelMap')['turquoise'], lw=5)
    for j in range(0, x.shape[0], 10):
        plt.plot(np.array([x[j], (1.0+ed[i, j]/edmax[i])*x[j]]), np.array([y[j], (1.0+ed[i, j]/edmax[i])*y[j]]), lw=1., color=colorScheme('labelMap')['turquoise'])
#    samps = sampleVMF(mu, tau[i], 50)
#    plt.scatter(samps[0, :], samps[1, :], c=colorScheme('labelMap')['orange'], s=200)
    plt.quiver(0, 0,  1.0/np.sqrt(2), 1.0/np.sqrt(2), lw=1, color='k', angles='xy', scale_units='xy', scale=1.0)
    plt.ylim([-2.0, 2.0])
    plt.xlim([-2.0, 2.0])
    ax.axes.get_yaxis().set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.set_aspect('equal', 'datalim')
    plt.savefig('vmf' + str(i) +'.pdf')

plt.show()

#u = np.linspace(0, 2*np.pi, 100)
#v = np.linspace(0, np.pi, 100)
#
#x = np.outer(np.cos(u), np.sin(v))
#y = np.outer(np.sin(u), np.sin(v))
#z = np.outer(np.ones(np.size(u)), np.cos(v))
#
#mu = np.ones(3)/np.sqrt(3)
#tau = np.array([1.0, 5.0, 10.0])
#
#d = mu[0]*x + mu[1]*y + mu[2]*z
#
#ed0 = tau[0]/(4*np.pi*np.sinh(tau[0]))*np.exp(tau[0]*d)
#ed1 = tau[1]/(4*np.pi*np.sinh(tau[1]))*np.exp(tau[1]*d)
#ed2 = tau[2]/(4*np.pi*np.sinh(tau[2]))*np.exp(tau[2]*d)
#
#
#colors = np.zeros((x.shape[0], x.shape[1], 3))
#for i in range(x.shape[0]):
#    for j in range(x.shape[1]):
#        frac = (d[i, j]+1.0)/2.0
#        colors[i, j, :] = np.array([ frac, 0, 1.0-frac])
#
#
#fig1 = plt.figure()
#ax = fig1.add_subplot(111, projection='3d')
#ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='k')
#ax.plot_surface((1.0+ed0)*x, (1.0+ed0)*y, (1.0+ed0)*z, rstride=2, cstride=2, linewidth=.2, facecolors=colors, alpha=0.2)
#ax.view_init(elev=20, azim=-30)
#plt.savefig("vmf1.svg")
#
#fig2 = plt.figure()
#ax = fig2.add_subplot(111, projection='3d')
#ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='k')
#ax.plot_surface((1.0+ed1)*x, (1.0+ed1)*y, (1.0+ed1)*z, rstride=2, cstride=2, linewidth=.2, facecolors=colors, alpha=0.2)
#ax.view_init(elev=20, azim=-30)
#plt.savefig("vmf2.svg")
#
#
#fig3 = plt.figure()
#ax = fig3.add_subplot(111, projection='3d')
#ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='k')
#ax.plot_surface((1.0+ed2)*x, (1.0+ed2)*y, (1.0+ed2)*z, rstride=2, cstride=2, linewidth=.2, facecolors=colors, alpha=0.2)
#ax.view_init(elev=20, azim=-30)
#plt.savefig("vmf3.svg")
#
#plt.show()
