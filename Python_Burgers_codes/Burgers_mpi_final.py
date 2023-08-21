from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

from mpi4py import MPI
import numpy as np
comm = MPI.COMM_WORLD
nproc = comm.Get_size() # new: gives number of ranks in comm
rank = comm.Get_rank()


t1 = datetime.now()

#Defining global variables
nu=0.01
dim=3   # This is 3 for 3D Codes and 2 for 2D codes
#We use 3D codes for 2D calculation as well. This factor is used for neglecting z variations in 2D code
nt = 1000
X_division=80
Y_division=80
Z_division=80
Xrange=2
Yrange=2
Zrange=2




def Timeloop(U,V,W,dx,dy,dz,Nx,Ny,Nz):
    
    
    
    nx=int(Nx/nproc)
    dt=dt_finder(U)
#    print(dt)
    
#    data=np.zeros((Nx,Ny,Nz),dtype=float)
    if rank == 0:
       
       data_chunks = np.array_split(U,nproc,axis=0)
       data_chunks2 = np.array_split(V,nproc,axis=0)
       data_chunks3 = np.array_split(W,nproc,axis=0)
    else:
        data_chunks=None
        data_chunks2=None
        data_chunks3=None
        

    us= comm.scatter(data_chunks,root=0)
    vs= comm.scatter(data_chunks,root=0)
    ws= comm.scatter(data_chunks,root=0)
    #print("data",data)
    #print('Rank: ',rank, ', recvbuf received: ',us)
    unew= np.zeros((nx+2,Ny,Nz),dtype=float)
    vnew= np.zeros((nx+2,Ny,Nz),dtype=float)
    wnew= np.zeros((nx+2,Ny,Nz),dtype=float)
    #comm.Barrier()

    u= np.zeros((nx+2,Ny,Nz),dtype=float)
        #u1= np.zeros((nx+1,Ny,Nz),dtype=float)
    v= np.zeros((nx+2,Ny,Nz),dtype=float)
        #v1= np.zeros((nx+1,Ny,Nz),dtype=float)
    w= np.zeros((nx+2,Ny,Nz),dtype=float)
        #w1= np.zeros((nx+1,Ny,Nz),dtype=float)
    if rank==0:
                u[1:-1,:,:]=us
                v[1:-1,:,:]=vs
                w[1:-1,:,:]=ws
    elif rank==nproc-1:
                u[1:-1,:,:]=us
                u[1:-1,:,:]=vs
                u[1:-1,:,:]=ws
    else :
                u[1:-1,:,:]=us
                v[1:-1,:,:]=vs
                w[1:-1,:,:]=ws
                    


    for n in range(nt): ##loop across number of time steps
        
        
        
        recv_buf = np.empty((1,Ny,Nz),dtype=np.float64)
        
        if rank == 0:
            u_to_right=np.array(u[-2,:,:])
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)
            v_to_right=np.array(v[-2,:,:])
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)
            w_to_right=np.array(w[-2,:,:])
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)
            
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)
            u[-1,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)
            v[-1,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)
            w[-1,:,:]=recv_buf;
            #print("proc ",rank, ": \n",u1)
#            unew= np.zeros((nx+2,Ny,Nz),dtype=float)
#            vnew= np.zeros((nx+2,Ny,Nz),dtype=float)
#            wnew= np.zeros((nx+2,Ny,Nz),dtype=float)

            unew[1:-1,1:-1,1:-1] = u[1:-1, 1:-1,1:-1] - dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (u[2:,1:-1,1:-1] - u[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (u[1:-1, 2:,1:-1] - u[1:-1,:-2,1:-1]) - \
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (u[1:-1,1:-1, 2:] - u[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (u[2:, 1:-1,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (u[1:-1, 2:,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (u[1:-1,1:-1,2:] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,1:-1,:-2])

            vnew[1:-1,1:-1,1:-1] = v[1:-1, 1:-1,1:-1] -  dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (v[2:,1:-1,1:-1] - v[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (v[1:-1, 2:,1:-1] - v[1:-1,:-2,1:-1]) -\
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (v[1:-1,1:-1,2:] - v[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (v[2:, 1:-1,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (v[1:-1, 2:,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (v[1:-1,1:-1,2:] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,1:-1,:-2]) 

            wnew[1:-1,1:-1,1:-1] = w[1:-1, 1:-1,1:-1] -  dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (w[2:,1:-1,1:-1] - w[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (w[1:-1, 2:,1:-1] - w[1:-1,:-2,1:-1]) -\
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (w[1:-1,1:-1, 2:] - w[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (w[2:, 1:-1,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (w[1:-1, 2:,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (w[1:-1,1:-1,2:] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,1:-1,:-2]) 

            u_to_last=np.array(unew[2,:,:])
            comm.Send([u_to_last,MPI.DOUBLE] ,dest=nproc-1,tag=rank)
            v_to_last=np.array(vnew[2,:,:])
            comm.Send([v_to_last,MPI.DOUBLE] ,dest=nproc-1,tag=rank)
            w_to_last=np.array(wnew[2,:,:])
            comm.Send([w_to_last,MPI.DOUBLE] ,dest=nproc-1,tag=rank)

            comm.Recv([recv_buf,MPI.DOUBLE] ,source=nproc-1,tag=nproc-1)
            unew[1,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=nproc-1,tag=nproc-1)
            vnew[1,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=nproc-1,tag=nproc-1)
            wnew[1,:,:]=recv_buf;
	    
            unew[:,0,:]=unew[:,-2,:]
            vnew[:,0,:]=vnew[:,-2,:]
            wnew[:,0,:]=wnew[:,-2,:]
		
            unew[:,-1,:]=unew[:,1,:]
            vnew[:,-1,:]=vnew[:,1,:]
            wnew[:,-1,:]=wnew[:,1,:]
            
            unew[:,:,0]=unew[:,:,-2]
            vnew[:,:,0]=vnew[:,:,-2]
            wnew[:,:,0]=wnew[:,:,-2]

            unew[:,:,-1]=unew[:,:,1]
            vnew[:,:,-1]=vnew[:,:,1]
            wnew[:,:,-1]=wnew[:,:,1]
            u[:,:,:]=unew[:,:,:]
            v[:,:,:]=vnew[:,:,:]
            w[:,:,:]=wnew[:,:,:]
            Velocity=float
            Velocity=np.max(u)
 #           print(dt)	
            
        elif rank == nproc-1:                                                                                                                       
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            u[0,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            v[0,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            w[0,:,:]=recv_buf;
            
            u_to_left=np.array(u[1,:,:])                                                                                                       
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)
            v_to_left=np.array(v[1,:,:])                                                                                                       
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)
            w_to_left=np.array(w[1,:,:])                                                                                                       
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)
            #print("proc ",rank, ": \n",u1)
 #           unew= np.zeros((nx+2,Ny,Nz),dtype=float)
 #           vnew= np.zeros((nx+2,Ny,Nz),dtype=float)
 #           wnew= np.zeros((nx+2,Ny,Nz),dtype=float)

            unew[1:-1,1:-1,1:-1] = u[1:-1, 1:-1,1:-1] - dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (u[2:,1:-1,1:-1] - u[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (u[1:-1, 2:,1:-1] - u[1:-1,:-2,1:-1]) - \
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (u[1:-1,1:-1, 2:] - u[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (u[2:, 1:-1,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (u[1:-1, 2:,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (u[1:-1,1:-1,2:] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,1:-1,:-2])

            vnew[1:-1,1:-1,1:-1] = v[1:-1, 1:-1,1:-1] -  dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (v[2:,1:-1,1:-1] - v[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (v[1:-1, 2:,1:-1] - v[1:-1,:-2,1:-1]) -\
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (v[1:-1,1:-1,2:] - v[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (v[2:, 1:-1,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (v[1:-1, 2:,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (v[1:-1,1:-1,2:] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,1:-1,:-2]) 

            wnew[1:-1,1:-1,1:-1] = w[1:-1, 1:-1,1:-1] -  dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (w[2:,1:-1,1:-1] - w[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (w[1:-1, 2:,1:-1] - w[1:-1,:-2,1:-1]) -\
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (w[1:-1,1:-1, 2:] - w[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (w[2:, 1:-1,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (w[1:-1, 2:,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (w[1:-1,1:-1,2:] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,1:-1,:-2]) 

            comm.Recv([recv_buf,MPI.DOUBLE] ,source=0,tag=0)
            unew[-2,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=0,tag=0)
            vnew[-2,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=0,tag=0)
            wnew[-2,:,:]=recv_buf;

            u_to_first=np.array(unew[-3,:,:])
            comm.Send([u_to_first,MPI.DOUBLE] ,dest=0,tag=rank)
            v_to_first=np.array(vnew[-3,:,:])
            comm.Send([v_to_first,MPI.DOUBLE] ,dest=0,tag=rank)
            w_to_first=np.array(wnew[-3,:,:])
            comm.Send([w_to_first,MPI.DOUBLE] ,dest=0,tag=rank)
	    
            unew[:,0,:]=unew[:,-2,:]
            vnew[:,0,:]=vnew[:,-2,:]
            wnew[:,0,:]=wnew[:,-2,:]

            unew[:,-1,:]=unew[:,1,:]
            vnew[:,-1,:]=vnew[:,1,:]
            wnew[:,-1,:]=wnew[:,1,:]

            unew[:,:,0]=unew[:,:,-2]
            vnew[:,:,0]=vnew[:,:,-2]
            wnew[:,:,0]=wnew[:,:,-2]

            unew[:,:,-1]=unew[:,:,1]
            vnew[:,:,-1]=vnew[:,:,1]
            wnew[:,:,-1]=wnew[:,:,1]

            u[:,:,:]=unew[:,:,:]
            v[:,:,:]=vnew[:,:,:]
            w[:,:,:]=wnew[:,:,:]
            Velocity=float
            Velocity=np.max(u)
            
        elif rank%2 == 0:                                                                                                                       
            u_to_right=np.array(u[-2,:,:])                                                                                                      
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)                                                                            
            u_to_left=np.array(u[1,:,:])                                                                                                        
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)                                                         
	    
            v_to_right=np.array(v[-2,:,:])                                                                                                      
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)                                                                            
            v_to_left=np.array(v[1,:,:])                                                                                                        
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)
            w_to_right=np.array(w[-2,:,:])                                                                                                      
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)                                                                            
            w_to_left=np.array(w[1,:,:])                                                                                                        
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)
            
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)                                                                          
            u[-1,:,:]=recv_buf;                                                                                                                 
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            u[0,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)                                                                          
            v[-1,:,:]=recv_buf;                                                                                                                 
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            v[0,:,:]=recv_buf;
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)                                                                          
            w[-1,:,:]=recv_buf;                                                                                                                 
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            w[0,:,:]=recv_buf;
            #print("proc ",rank, ": \n",u)
  #          unew= np.zeros((nx+2,Ny,Nz),dtype=float)
  #          vnew= np.zeros((nx+2,Ny,Nz),dtype=float)
  #          wnew= np.zeros((nx+2,Ny,Nz),dtype=float)

            unew[1:-1,1:-1,1:-1] = u[1:-1, 1:-1,1:-1] - dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (u[2:,1:-1,1:-1] - u[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (u[1:-1, 2:,1:-1] - u[1:-1,:-2,1:-1]) - \
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (u[1:-1,1:-1, 2:] - u[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (u[2:, 1:-1,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (u[1:-1, 2:,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (u[1:-1,1:-1,2:] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,1:-1,:-2])

            vnew[1:-1,1:-1,1:-1] = v[1:-1, 1:-1,1:-1] -  dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (v[2:,1:-1,1:-1] - v[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (v[1:-1, 2:,1:-1] - v[1:-1,:-2,1:-1]) -\
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (v[1:-1,1:-1,2:] - v[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (v[2:, 1:-1,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (v[1:-1, 2:,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (v[1:-1,1:-1,2:] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,1:-1,:-2]) 

            wnew[1:-1,1:-1,1:-1] = w[1:-1, 1:-1,1:-1] -  dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (w[2:,1:-1,1:-1] - w[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (w[1:-1, 2:,1:-1] - w[1:-1,:-2,1:-1]) -\
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (w[1:-1,1:-1, 2:] - w[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (w[2:, 1:-1,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (w[1:-1, 2:,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (w[1:-1,1:-1,2:] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,1:-1,:-2]) 
            
            unew[:,0,:]=unew[:,-2,:]
            vnew[:,0,:]=vnew[:,-2,:]
            wnew[:,0,:]=wnew[:,-2,:]

            unew[:,-1,:]=unew[:,1,:]
            vnew[:,-1,:]=vnew[:,1,:]
            wnew[:,-1,:]=wnew[:,1,:]

            unew[:,:,0]=unew[:,:,-2]
            vnew[:,:,0]=vnew[:,:,-2]
            wnew[:,:,0]=wnew[:,:,-2]

            unew[:,:,-1]=unew[:,:,1]
            vnew[:,:,-1]=vnew[:,:,1]
            wnew[:,:,-1]=wnew[:,:,1]
            u[:,:,:]=unew[:,:,:]
            v[:,:,:]=vnew[:,:,:]
            w[:,:,:]=wnew[:,:,:]
            Velocity=float
            Velocity=np.max(u)

                                                                                                                                            
        elif rank%2 == 1:
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            u[0,:,:]=recv_buf;                                                                                                                  
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)                                                                          
            u[-1,:,:]=recv_buf;                                                                                                                                                                                       
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            v[0,:,:]=recv_buf;                                                                                                                  
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)                                                                          
            v[-1,:,:]=recv_buf;                                                                                                                                                                                       
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank-1,tag=rank-1)                                                                          
            w[0,:,:]=recv_buf;                                                                                                                  
            comm.Recv([recv_buf,MPI.DOUBLE] ,source=rank+1,tag=rank+1)                                                                          
            w[-1,:,:]=recv_buf;                                                                                                                                                                                       

            
            u_to_left=np.array(u[1,:,:])
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)
            u_to_right=np.array(u[-2,:,:])
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)
            v_to_left=np.array(v[1,:,:])
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)
            v_to_right=np.array(v[-2,:,:])
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)
            w_to_left=np.array(w[1,:,:])
            comm.Send([u_to_left,MPI.DOUBLE] ,dest=rank-1,tag=rank)
            w_to_right=np.array(w[-2,:,:])
            comm.Send([u_to_right,MPI.DOUBLE] ,dest=rank+1,tag=rank)
            #print("proc ",rank, ": \n",u)

   #         unew= np.zeros((nx+2,Ny,Nz),dtype=float)
   #         vnew= np.zeros((nx+2,Ny,Nz),dtype=float)
   #         wnew= np.zeros((nx+2,Ny,Nz),dtype=float)

            unew[1:-1,1:-1,1:-1] = u[1:-1, 1:-1,1:-1] - dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (u[2:,1:-1,1:-1] - u[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (u[1:-1, 2:,1:-1] - u[1:-1,:-2,1:-1]) - \
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (u[1:-1,1:-1, 2:] - u[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (u[2:, 1:-1,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (u[1:-1, 2:,1:-1] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (u[1:-1,1:-1,2:] - 2 * u[1:-1, 1:-1,1:-1] + u[1:-1,1:-1,:-2])

            vnew[1:-1,1:-1,1:-1] = v[1:-1, 1:-1,1:-1] -  dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (v[2:,1:-1,1:-1] - v[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (v[1:-1, 2:,1:-1] - v[1:-1,:-2,1:-1]) -\
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (v[1:-1,1:-1,2:] - v[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (v[2:, 1:-1,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (v[1:-1, 2:,1:-1] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (v[1:-1,1:-1,2:] - 2 * v[1:-1, 1:-1,1:-1] + v[1:-1,1:-1,:-2]) 

            wnew[1:-1,1:-1,1:-1] = w[1:-1, 1:-1,1:-1] -  dt / (2*dx) * u[1:-1, 1:-1,1:-1] * (w[2:,1:-1,1:-1] - w[:-2, 1:-1,1:-1])-\
                            dt /(2*dy) * v[1:-1, 1:-1,1:-1] * (w[1:-1, 2:,1:-1] - w[1:-1,:-2,1:-1]) -\
                            (dim-2)*dt /(2*dz) * w[1:-1, 1:-1,1:-1] * (w[1:-1,1:-1, 2:] - w[1:-1,1:-1,:-2]) + \
                            nu * dt / dx**2 * (w[2:, 1:-1,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[:-2, 1:-1,1:-1]) + \
                            nu * dt / dy**2 * (w[1:-1, 2:,1:-1] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,:-2,1:-1]) + \
                            (dim-2)*nu * dt / dz**2 * (w[1:-1,1:-1,2:] - 2 * w[1:-1, 1:-1,1:-1] + w[1:-1,1:-1,:-2]) 
            

            unew[:,0,:]=unew[:,-2,:]
            vnew[:,0,:]=vnew[:,-2,:]
            wnew[:,0,:]=wnew[:,-2,:]

            unew[:,-1,:]=unew[:,1,:]
            vnew[:,-1,:]=vnew[:,1,:]
            wnew[:,-1,:]=wnew[:,1,:]

            unew[:,:,0]=unew[:,:,-2]
            vnew[:,:,0]=vnew[:,:,-2]
            wnew[:,:,0]=wnew[:,:,-2]

            unew[:,:,-1]=unew[:,:,1]
            vnew[:,:,-1]=vnew[:,:,1]
            wnew[:,:,-1]=wnew[:,:,1]

            u[:,:,:]=unew[:,:,:]
            v[:,:,:]=vnew[:,:,:]
            w[:,:,:]=wnew[:,:,:]
            Velocity=float
            Velocity=np.max(u)

                            
        Vel=np.empty(nproc,dtype=float)
        #print("Velocity",Vel)
        Vel=comm.gather(Velocity, root=0)
        if rank==0:
           # print("Velocity",Vel)
           dT=dt_finder(Vel)

        if rank==0:
            send_buffer=dT
        else:
            send_buffer=None

        dt=comm.bcast(send_buffer,root=0)

        

        
        

    #comm.Barrier()

    data_2 =np.empty((int(Nx/nproc),Ny,Nz), dtype=float)
    data_3 =np.empty((int(Nx/nproc),Ny,Nz), dtype=float)
    data_4 =np.empty((int(Nx/nproc),Ny,Nz), dtype=float)
    if rank==0:
            data_2=u[1:-1,:,:]
            data_3=v[1:-1,:,:]
            data_4=w[1:-1,:,:]
    elif rank==nproc-1:
            data_2=u[1:-1,:,:]
            data_3=v[1:-1,:,:]
            data_4=w[1:-1,:,:]
    else :
            data_2=u[1:-1,:,:]
            data_3=v[1:-1,:,:]
            data_4=w[1:-1,:,:]


    #print("data",data)
    #data_2 =np.empty((int(Nx/nproc),Ny,Nz), dtype=float)
    recv_array=np.zeros((Nx,Ny,Nz),dtype=np.float64)
    recv_array2=np.zeros((Nx,Ny,Nz),dtype=np.float64)
    recv_array3=np.zeros((Nx,Ny,Nz),dtype=np.float64)

    recv_array = comm.gather(data_2, root=0)
    recv_array2 = comm.gather(data_3, root=0)
    recv_array3 = comm.gather(data_4, root=0)

    if rank == 0:
        recv_array = np.concatenate(recv_array,axis = 0)
        recv_array2 = np.concatenate(recv_array2,axis = 0)
        recv_array3 = np.concatenate(recv_array3,axis = 0)
        #print(recv_array)       
    
    return  recv_array,recv_array2,recv_array3


def OneD(Xrange):
    Nx=X_division
    Ny=Nz=3
    dx=Xrange/(Nx-1)
    dy=dx
    dz=dx
    u = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    v = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    w = np.zeros((Nx,Ny,Nz),dtype=np.float64)
 #   unew = np.zeros((Nx,Ny,Nz),dtype=np.float64)
 #   vnew = np.zeros((Nx,Ny,Nz),dtype=np.float64)
 #   wnew = np.zeros((Nx,Ny,Nz),dtype=np.float64)

    # Initial condition
    u[int(0.5/ dx):int(1/dx + 1),1,1] = 2 

    U,V,W=Timeloop(u,v,w,dx,dy,dz,Nx,Ny,Nz)
    
    #plot
#    x = np.linspace(0, Xrange,Nx)
#    u = U[:,1,1]  # create a 1xn vector of 1's
#    pyplot.figure()
#    pyplot.plot(x,u)
#    pyplot.show()


def TwoD(Xrange,Yrange):
    Nx=X_division
    Ny=Y_division    
    Nz=3
    
    dx=Xrange/(Nx-1)
    dy=Yrange/(Ny-1)
    dz=dx
    u = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    v = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    w = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    U = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    V = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    W = np.zeros((Nx,Ny,Nz),dtype=np.float64)

    # Initial condition
    u[int(0.5/ dx):int(1/dx + 1),int(0.5/ dy):int(1/dy + 1),1] = 0.5 
    v[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1),1] = 0.5
    #print(u)
    U,V,W=Timeloop(u,v,w,dx,dy,dz,Nx,Ny,Nz)
    #print(U[1][1][1]) 
    if rank==0:
        f = open("Results_2D_80_80_mpi","w")
        for i in  range (0,Nx,1):
          for j in  range (0,Ny,1):
            f.write(str(round(dx*i,8)))
            f.write(" ")
            f.write(str(round(dy*j,8)))
            f.write(" ")
            f.write(str(U[i][j][1]))
            f.write(" ")
            f.write(str(V[i][j][1]))
            f.write("\n")
        f.close()   
    
def ThreeD(Xrange,Yrange,Zrange):
    Nx=X_division
    Ny=Y_division
    Nz=Z_division	 
    dx=Xrange/(Nx-1)
    dy=Yrange/(Ny-1)
    dz=Zrange/(Nz-1)
    u = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    v = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    w = np.zeros((Nx,Ny,Nz),dtype=np.float64)
    #unew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    #vnew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)
    #wnew = numpy.zeros((Nx,Ny,Nz),dtype=numpy.float64)

    # Initial condition
    u[0:int(1 / dx), 0:int(1 / dy), 0:int(1 / dz)] = 2
    ##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    v[0:int(1 / dx), 0:int(1 / dy), 0:int(1 / dz)] = 2
    w[0:int(1 / dx), 0:int(1 / dy), 0:int(1 / dz)] = 2


    U,V,W=Timeloop(u,v,w,dx,dy,dz,Nx,Ny,Nz)
    if rank==0:
        f = open("Results_3D_80_cube_mpi","w")
        for i in  range (0,Nx,1):
          for j in  range (0,Ny,1):
             for k in  range (0,Nz,1):
               f.write(str(round(dx*i,8)))
               f.write(" ")
               f.write(str(round(dy*j,8)))
               f.write(" ")
               f.write(str(round(dz*k,8)))
               f.write(" ")
               f.write(str(U[i][j][1]))
               f.write(" ")
               f.write(str(V[i][j][1]))
               f.write(" ")
               f.write(str(W[i][j][1]))
               f.write("\n")
        f.close()







def dt_finder(speed):

        
    dx=Xrange/(X_division-1)
    dt1=((0.8/dim)*(dx*dx)/(2*nu))

    vmax=np.max(speed)
    
    dt2=((0.8/dim)*(dx/vmax))

    dt=min(dt1,dt2)

    return dt    

#OneD(Xrange)

#TwoD(Xrange, Yrange)

ThreeD(Xrange, Yrange, Zrange)
if rank==0:
	t2 = datetime.now()
	print(t2-t1)

