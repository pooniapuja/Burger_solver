
import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import matplotlib.pyplot as pp

t1 = datetime.now()
dim=3
#Defining global variables
nu=0.01
nt = 1000
nx=80
ny=80
nz=80
Xrange=2
Yrange=2
Zrange=2
#dt=Xrange/(nx*8)



def Timeloop(u,v,w,unew,vnew,wnew,dx,dy,dz):
    # unew = u
    # vnew = v
    # wnew = w
    



    for n in range(nt): ##loop across number of time steps

        dt=dt_finder(u,v,w,dx)
        
        unew[1:-1,1:-1,1:-1] = u[1:-1, 1:-1,1:-1] - (dt / (2*dx)) * u[1:-1, 1:-1,1:-1] * (u[2:,1:-1,1:-1] - u[:-2, 1:-1,1:-1])-\
                            (dt /(2*dy)) * v[1:-1, 1:-1,1:-1] * (u[1:-1, 2:,1:-1] - u[1:-1,:-2,1:-1]) - \
                            (dim-2)*(dt /(2*dz)) * w[1:-1, 1:-1,1:-1] * (u[1:-1,1:-1, 2:] - u[1:-1,1:-1,:-2]) + \
                            (nu * dt / dx**2) * (u[2:, 1:-1,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[:-2, 1:-1,1:-1]) + \
                            (nu * dt / dy**2) * (u[1:-1, 2:,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,:-2,1:-1]) + \
                            (dim-2)*(nu * dt / dz**2) * (u[1:-1,1:-1,2:] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,1:-1,:-2])
        #print(unew)

        vnew[1:-1,1:-1,1:-1] = v[1:-1, 1:-1,1:-1] -  (dt / (2*dx)) * u[1:-1, 1:-1,1:-1] * (v[2:,1:-1,1:-1] - v[:-2, 1:-1,1:-1])-\
                            (dt /(2*dy)) * v[1:-1, 1:-1,1:-1] * (v[1:-1, 2:,1:-1] - v[1:-1,:-2,1:-1]) -\
                            (dim-2)*(dt /(2*dz)) * w[1:-1, 1:-1,1:-1] * (v[1:-1,1:-1,2:] - v[1:-1,1:-1,:-2]) + \
                            (nu * dt / dx**2) * (v[2:, 1:-1,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[:-2, 1:-1,1:-1]) + \
                            (nu * dt / dy**2) * (v[1:-1, 2:,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,:-2,1:-1]) + \
                            (dim-2)*(nu * dt / dz**2) * (v[1:-1,1:-1,2:] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,1:-1,:-2]) 

        wnew[1:-1,1:-1,1:-1] = w[1:-1, 1:-1,1:-1] -  (dt / (2*dx)) * u[1:-1, 1:-1,1:-1] * (w[2:,1:-1,1:-1] - w[:-2, 1:-1,1:-1])-\
                            (dt /(2*dy)) * v[1:-1, 1:-1,1:-1] * (w[1:-1, 2:,1:-1] - w[1:-1,:-2,1:-1]) -\
                            (dim-2)*(dt /(2*dz)) * w[1:-1, 1:-1,1:-1] * (w[1:-1,1:-1, 2:] - w[1:-1,1:-1,:-2]) + \
                            (nu * dt / dx**2) * (w[2:, 1:-1,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[:-2, 1:-1,1:-1]) + \
                            (nu * dt / dy**2) * (w[1:-1, 2:,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,:-2,1:-1]) + \
                            (dim-2)*(nu * dt / dz**2) * (w[1:-1,1:-1,2:] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,1:-1,:-2]) 

                            
    
        


        unew[0,:,:] = unew[-2,:,:]
        unew[-1,:,:] = unew[1,:,:] 

        unew[:,0,:] = unew[:,-2,:]
        unew[:,-1,:] = unew[:,1,:]

        unew[:,:,0] = unew[:,:,-2]
        unew[:,:,-1] = unew[:,:,1] 

        vnew[0,:,:] = vnew[-2,:,:]
        vnew[-1,:,:] = vnew[1,:,:] 

        vnew[:,0,:] = vnew[:,-2,:]
        vnew[:,-1,:] = vnew[:,1,:]

        vnew[:,:,0] = vnew[:,:,-2]
        vnew[:,:,-1] = vnew[:,:,1]

        wnew[0,:,:] = wnew[-2,:,:]
        wnew[-1,:,:] = wnew[1,:,:] 

        wnew[:,0,:] = wnew[:,-2,:]
        wnew[:,-1,:] = wnew[:,1,:]

        wnew[:,:,0] = wnew[:,:,-2]
        wnew[:,:,-1] = wnew[:,:,1] 

        u[:,:,:]=unew[:,:,:]
        v[:,:,:]=vnew[:,:,:]
        w[:,:,:]=wnew[:,:,:]

              
        
    return u,v,w


def OneD(Xrange):
    Nx=nx
    Ny=Nz=3
    dx=Xrange/(Nx-1)
    dy=dx
    dz=dx
    u = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    v = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    w = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    unew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    vnew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    wnew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)

    # Initial condition
    u[int(0.5/ dx):int(1/dx + 1),1,1] = 2 

    U,V,W=Timeloop(u,v,w,unew,vnew,wnew,dx,dy,dz)

    '''
    #plot
    x = numpy.linspace(0, Xrange, nx)
    u = U[:,1,1]  # create a 1xn vector of 1's
    pyplot.figure()
    pyplot.plot(x,u)
    pyplot.show()
'''

def TwoD(Xrange,Yrange):
    Nx=nx
    Ny=ny
    Nz=4
    
    dx=Xrange/(Nx-1)
    dy=Yrange/(Ny-1)
    dz=dx
    u = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    v = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    w = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    unew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    vnew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    wnew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)

    # Initial condition
    u[int(0.5/ dx):int(1/dx + 1),int(0.5/ dx):int(1/dx + 1),1] = 0.5
    v[int(0.5 / dy):int(1 / dx + 1),int(.5 / dx):int(1 / dx + 1),1] = 0.5
    #print(u[:][:][1])

    U,V,W=Timeloop(u,v,w,unew,vnew,wnew,dx,dy,dz)
    
    f = open("Results_2D_800_800", "w")
    for i in  range (0,Nx,1):
      for j in  range (0,Ny,1):
          
        f.write(str(round(dx*i,8)))
        f.write(" ")
        f.write(str(round(dy*j,8)))
        f.write(" ")
        f.write(str(round(U[i][j][1],8)))
        f.write(" ")
        f.write(str(round(V[i][j][1],8)))
        f.write("\n")
    f.close()
    
    
 

    
    #Plot
    x = numpy.linspace(0, Xrange, Nx)
    y = numpy.linspace(0, Yrange, Ny)
    
    U1 = U[:,:,1]  # create a 1xn vector of 1's
    V1 = V[:,:,1]
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    ax = fig.gca(projection='3d')
    X, Y = numpy.meshgrid(x, y)
    ax.plot_surface(X, Y, U1)
    #ax.plot_surface(X, Y, v)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$');
    pyplot.show();
    
    
    
def ThreeD(Xrange,Yrange,Zrange):
    Nx=nx
    Ny=ny
    Nz=nz
    dx=Xrange/(Nx-1)
    dy=Yrange/(Ny-1)
    dz=Zrange/(Nz-1)
    u = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    v = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    w = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    unew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    vnew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    wnew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)

    # Initial condition
    u[0:int(1 / dx), 0:int(1 / dy), 0:int(1 / dz)] = 0.5
    ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    v[0:int(1 / dx), 0:int(1 / dy), 0:int(1 / dz)] = 0.5
    w[0:int(1 / dx), 0:int(1 / dy), 0:int(1 / dz)] = 0


    U,V,W=Timeloop(u,v,w,unew,vnew,wnew,dx,dy,dz)
    f = open("Results_3D_80_80_80", "w")
    for i in  range (0,Nx,1):
      for j in  range (0,Ny,1):
        for k in  range (0,Nz,1):  
          
            f.write(str(round(dx*i,8)))
            f.write(" ")
            f.write(str(round(dy*j,8)))
            f.write(" ")
            f.write(str(round(dz*k,8)))
            f.write(" ")
            f.write(str(round(U[i][j][k],8)))
            f.write(" ")
            f.write(str(round(V[i][j][k],8)))
            f.write(" ")
            f.write(str(round(W[i][j][k],8)))
            f.write("\n")
    f.close()

    fig = pyplot.figure()
    ax = fig.gca(projection='3d')

    x = numpy.linspace(0, Xrange, nx)
    y = numpy.linspace(0, Yrange, ny)
    z = numpy.linspace(0, Zrange, nz)
    x_in_p = x[::5]
    y_in_p = y[::5]
    z_in_p = z[::5]
    X, Y, Z = numpy.meshgrid(x_in_p, y_in_p, z_in_p)
    u_in_p = U[::5, ::5, ::5]
    v_in_p = V[::5, ::5, ::5]
    w_in_p = W[::5, ::5, ::5]
    ax.quiver(X, Y, Z, u_in_p, v_in_p, w_in_p, length=0.1, normalize=False)
    ax.set_zlim(0,2)
    ax.set_ylim(0,2)
    ax.set_xlim(0,2)
    pp.savefig('3D_80_cube.png',dpi=2000)
    pyplot.show();
    

def dt_finder(u,v,w,dx):

    

    dt1=((0.8/dim)*(dx*dx)/(2*nu))

    vmax=numpy.max(u)
    
    dt2=((0.8/dim)*(dx/vmax))

    dt=min(dt1,dt2)

    return dt

#OneD(Xrange)

#TwoD(Xrange, Yrange)

ThreeD(Xrange, Yrange, Zrange)

t2 = datetime.now()
print(t2-t1)
