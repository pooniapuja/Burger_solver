/////The code solves 2D Burgers equation using MPI. It uses malloc function to allocate memory in arrays and is easily scalable for big data points and large number of processors. But it may not be as fast as Blitz++, since it doesn't use vectorization.
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <fstream>
#include <mpi.h>     /* For MPI functions, etc */ 
//using namespace std::chrono;

using namespace std;

double **allocate_array(int row_dim, int col_dim) 
{
  double **result;
  int i;

  result=(double **)malloc(row_dim*sizeof(double *));

  result[0]=(double *)malloc(row_dim*col_dim*sizeof(double));

  for(i=1; i<row_dim; i++)
	result[i]=result[i-1]+col_dim;

  return result;
}


void deallocate_array(double **array, int row_dim) 
{
  int i;
  /* Make sure all the pointers into the array are not pointing to
	 random locations in memory */
  for(i=1; i<row_dim; i++)
	array[i]=NULL;
  /* De-allocate the array */
  free(array[0]);
  /* De-allocate the array of pointers */
  free(array);
}

int main()
{ 
  int       nprocs;              
  int       rank;               
  MPI_Request reqs[8];   
  MPI_Status stats[8];

  MPI_Init(NULL, NULL); 

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

  int N=9;
  int i,j,k,itr,iteration;
  int count=N/(nprocs);
  double kin_visc=0.01,L=2;
  double dx=L/N,dt=dx/8;
  double dy=L/N;
  iteration=1;
  double start = MPI_Wtime(); 
  double **v_new=NULL, **u_new=NULL, **recv_u=NULL, **recv_v=NULL;

  double **u=NULL,**v=NULL,**x=NULL,**y=NULL;
  u=allocate_array(1,1);
  v=allocate_array(1,1);

  double T=0.3;
   
  int u1=0.5/dx+1;
  int u2=1/dx;
  int v1=0.5/dy+1;
  int v2=1/dy;

if (rank==0)
{    
      u = allocate_array(N,N+2);
      v = allocate_array(N,N+2);
      x = allocate_array(N,N+2);
      y = allocate_array(N,N+2);  

      for(i=0;i<N;i++)
      {   
        for(j=0;j<=N+1;j++) //Getting initialised
        {   
          x[i][j]=dx*(i-1);
          y[i][j]=dy*(j-1);

          u[i][j]= 0; 
          v[i][j]= 0;   
        }      
      }

      for(i=u1;i<=u2;i++)
        for(j=v1;j<=v2;j++) 
        {
          u[i][j]=2;
    
          v[i][j]=2;

        }
   
}
    u_new=allocate_array(count+2,N+2);
    v_new=allocate_array(count+2,N+2);
    recv_u=allocate_array(count+2,N+2);
    recv_v=allocate_array(count+2,N+2);
   
     MPI_Scatter (*u,count*(N+2),MPI_DOUBLE,u_new[1],count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
     MPI_Scatter (*v,count*(N+2),MPI_DOUBLE,v_new[1],count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
   
     for(itr=1;itr<=iteration;itr++)
      {//if (rank==2) cout<<itr<<endl;
          MPI_Isend(u_new[1], (N+2), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[0]); 
          MPI_Isend(u_new[count], (N+2), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[1]); 
          MPI_Isend(v_new[1], (N+2), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 2, MPI_COMM_WORLD,&reqs[2]); 
          MPI_Isend(v_new[count], (N+2), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 2, MPI_COMM_WORLD,&reqs[3]); 
          MPI_Irecv(u_new[0], (N+2), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[4]); 
          MPI_Irecv(u_new[count+1], (N+2), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[5]); 
          MPI_Irecv(v_new[0], (N+2), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 2, MPI_COMM_WORLD,&reqs[6]); 
          MPI_Irecv(v_new[count+1], (N+2), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 2 , MPI_COMM_WORLD,&reqs[7]); 
          MPI_Waitall(8, reqs, stats);

        for(i=0;i<count;i++)  
          for(j=0;j<N+2;j++)
          {
            recv_u[i][j]=u_new[i][j];
            recv_v[i][j]=v_new[i][j];
          }

        for(i=1;i<count+1;i++)  
          {
            for(j=1;j<N+1;j++)
              {    //if (rank==2)   cout<<"    "<< u_new[i][j]<<"    ";

            u_new[i][j] = recv_u[i][j] - (dt/(2*dx))*recv_u[i][j]*(recv_u[i+1][j]-recv_u[i-1][j]) \
			        + (kin_visc*dt/(dx*dx))*(recv_u[i+1][j]-2*recv_u[i][j]+recv_u[i-1][j]) \
              - (dt/(2*dy))*recv_v[i][j]*(recv_u[i][j+1]-recv_u[i][j-1]) \
			        + (kin_visc*dt/(dy*dy))*(recv_u[i][j+1]-2*recv_u[i][j]+recv_u[i][j-1]);

            v_new[i][j] = recv_v[i][j] - (dt/(2*dx))*recv_v[i][j]*(recv_v[i+1][j]-recv_v[i-1][j]) \
			        + (kin_visc*dt/(dx*dx))*(recv_v[i+1][j]-2*recv_v[i][j]+recv_v[i-1][j]) \
              -(dt/(2*dy))*recv_v[i][j]*(recv_v[i][j+1]-recv_v[i][j-1]) \
			        + (kin_visc*dt/(dy*dy))*(recv_v[i][j+1]-2*recv_v[i][j]+recv_v[i][j-1]);   
              }
          //   if (rank==2)   cout<<"Next"<<endl;
          }

      }
        MPI_Gather (u_new[1],(count)*(N+2),MPI_DOUBLE,*u,count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Gather (v_new[1],(count)*(N+2),MPI_DOUBLE,*v,count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
   if (rank==0)
   {

      ofstream outdata("Results_BG_2D.txt");
      for(i=0;i<N;i++)
      {
        for(j=1;j<N+1; j++)
        {//cout<<u[i][j]<<"    ";
        outdata<<x[i][j]<<" "<<y[i][j]<< " "<< u[i][j]<< " "<<v[i][j]<<endl;}
       // cout<<endl;
        }
      outdata.close();
      double end=MPI_Wtime();
       cout << end-start << endl; 
      deallocate_array(u,N);
      deallocate_array(v,N);

   }
   deallocate_array(u_new,count+2);
  deallocate_array(v_new,count+2);
   deallocate_array(recv_u,count+2);
   deallocate_array(recv_v,count+2);

  MPI_Finalize();
  return 0;

}