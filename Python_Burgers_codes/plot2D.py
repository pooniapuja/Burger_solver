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
    
with open ("Results_2D_800_800_mpi")as textFile:
        lines = [line.split() for line in textFile]
npts=len(lines)

Nx=800
Ny=800
#Finding the leading edge based on the value of leading edge point no
x = numpy.zeros((Nx,Ny),dtype=numpy.float64)
y = numpy.zeros((Nx,Ny),dtype=numpy.float64)
u = numpy.zeros((Nx,Ny),dtype=numpy.float64)
v = numpy.zeros((Nx,Ny),dtype=numpy.float64)
i=0
j=0

for I in  range (0,npts,1):
     #print(i)
     x[i][j] = float(lines[I][0])
     y[i][j] = float(lines[I][1])
     u[i][j] = float(lines[I][2])
     v[i][j] = float(lines[I][3])
     j=j+1
     if((I+1)%Nx==0):
         i=i+1
         j=0

#x = numpy.linspace(0, Xrange, Nx)
#y = numpy.linspace(0, Yrange, Ny)
    
#U1 = U[:,:,1]  # create a 1xn vector of 1's
#V1 = V[:,:,1]
fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
#X, Y = numpy.meshgrid(x, y)
ax.plot_surface(x, y,u)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$');
pyplot.show();
