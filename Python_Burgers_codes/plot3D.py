# -*- coding: utf-8 -*-
"""

@author: Pooja
"""


import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))
    
with open ("Results_3D_80_80_80")as textFile:
        lines = [line.split() for line in textFile]
npts=len(lines)

Nx=80
Ny=80
Nz=80
Xrange=2
Yrange=2
Zrange=2
#x = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
#y = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
#z = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
u = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
v = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
w = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)



i=0
j=0
k=0
for I in  range (0,npts,1):
      # print("I",I)
      # print("i",i)
      # print("j",j)
      # print("k",k)
     # x[i][j][k] = float(lines[I][0])
     # y[i][j][k] = float(lines[I][1])
     # z[i][j][k] = float(lines[I][2])
      u[i][j][k] = float(lines[I][3])
      v[i][j][k] = float(lines[I][4])
      w[i][j][k] = float(lines[I][5])
      
      k=k+1
      
      if((I+1)%((Ny)*(Nz))==0):
          i=i+1
          j=0
          k=0
      elif((I+1)%(Nz)==0):
          j=j+1
          k=0
fig = pyplot.figure()
ax = fig.gca(projection='3d')

x = numpy.linspace(0, Xrange, Nx)
y = numpy.linspace(0, Yrange, Ny)
z = numpy.linspace(0, Zrange, Nz)
x_in_p = x[::5]
y_in_p = y[::5]
z_in_p = z[::5]
X, Y, Z = numpy.meshgrid(x_in_p, y_in_p, z_in_p)
u_in_p = u[::5, ::5, ::5]
v_in_p = v[::5, ::5, ::5]
w_in_p = w[::5, ::5, ::5]
ax.quiver(X, Y, Z, u_in_p, v_in_p, w_in_p, length=0.1, normalize=False)
ax.set_zlim(0,2)
ax.set_ylim(0,2)
ax.set_xlim(0,2)
pyplot.show();
