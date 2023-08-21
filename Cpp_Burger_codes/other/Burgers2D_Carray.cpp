///The code is not recommended for large data sizes and for too many processors. Its the simple code which uses traditional C++ arrays. It consumes too much memory since each processor has all variables. Its only for the beginners. An alternate code using malloc is also avaialable in the folder
#include <iostream>
#include <stdio.h>
#include <math.h> 
#include <fstream>
#include <mpi.h>     /* For MPI functions, etc */ 
//using namespace std::chrono;

using namespace std;

int main()
{ 
  int       nprocs;               /* Number of processes    */
  int       rank;               /* My process rank        */
  MPI_Request reqs[8];   // required variable for non-blocking calls
   MPI_Status stats[8];
  /* Start up MPI */
   MPI_Init(NULL, NULL); 

   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 

   /* Get my rank among all the processes */
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
   int N=8,i,j,k,itr,iteration;
   int count=N/(nprocs);
   double recv_u[count+2][N+2],recv_v[count+2][N+2],u_new[count+2][N+2],v_new[count+2][N+2];
   double kin_visc=0.01,L=2;
   double dx=L/N,dt=dx/8;
   double dy=L/N;
   iteration=1;
   double start = MPI_Wtime(); 
    double u[N][N+2],u_initial[N][N+2],x[N][N+2];
      double v[N][N+2],v_initial[N][N+2],y[N][N+2];
  // cout<<iteration;
   if (rank==0)
   {	
    //   cout<<"master"<<endl;
      //u1,u2,v1,v2;
      double T=0.3;
      
      int u1=0.5/dx+1;
      int u2=1/dx;
      int v1=0.5/dy+1;
      int v2=1/dy;
      
      //Setting initial condition 


      for(i=0;i<N;i++)
        for(j=0;j<=N+1;j++) //Getting initialised
        {   
          x[i][j]=dx*(i-1);
          y[i][j]=dy*(j-1);


          u[i][j]= 2*i*j;
          u_initial[i][j]= 0;
          v[i][j]= 0;
          v_initial[i][j]= 0;
      
        }
      for(i=u1;i<=u2;i++)
        for(j=v1;j<=v2;j++) 
        {
          //u[i][j]=2;
          u_initial[i][j]=2;
          v[i][j]=2;
          v_initial[i][j]=2;

        }
        ofstream outdat("Initial_BG_2D.txt");

      for(i=0;i<N;i++)
        for(j=1;j<N+1;j++)
          outdat<< x[i][j]<<" "<<y[i][j] << " "<< u_initial[i][j]<< " " <<v_initial[i][j]<<endl;
      outdat.close();
   
   }
    MPI_Scatter (u,count*(N+2),MPI_DOUBLE,u_new[1],count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Scatter (v,count*(N+2),MPI_DOUBLE,v_new[1],count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
   
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
        for(i=0;i<count+2;i++)  
          {for(j=0;j<=N+1;j++)
          {    //if (rank==3)   cout<<"    "<< u_new[i][j]<<"    ";

            u_new[i][j] = recv_u[i][j] - (dt/(2*dx))*recv_u[i][j]*(recv_u[i+1][j]-recv_u[i-1][j]) \
			        + (kin_visc*dt/(dx*dx))*(recv_u[i+1][j]-2*recv_u[i][j]+recv_u[i-1][j]) \
              - (dt/(2*dy))*recv_v[i][j]*(recv_u[i][j+1]-recv_u[i][j-1]) \
			        + (kin_visc*dt/(dy*dy))*(recv_u[i][j+1]-2*recv_u[i][j]+recv_u[i][j-1]);

            v_new[i][j] = recv_v[i][j] - (dt/(2*dx))*recv_v[i][j]*(recv_v[i+1][j]-recv_v[i-1][j]) \
			        + (kin_visc*dt/(dx*dx))*(recv_v[i+1][j]-2*recv_v[i][j]+recv_v[i-1][j]) \
              -(dt/(2*dy))*recv_v[i][j]*(recv_v[i][j+1]-recv_v[i][j-1]) \
			        + (kin_visc*dt/(dy*dy))*(recv_v[i][j+1]-2*recv_v[i][j]+recv_v[i][j-1]);    
          }
            // if (rank==3)   cout<<"Next"<<endl;
             }

      }
        MPI_Gather (u_new[1],(count)*(N+2),MPI_DOUBLE,u,count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Gather (v_new[1],(count)*(N+2),MPI_DOUBLE,v,count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
   if (rank==0)
   {

      ofstream outdata("Results_BG_2D.txt");
      for(i=0;i<N;i++)
      {
        for(j=1;j<N+1; j++)
        {cout<<u[i][j]<<"    ";
        outdata<<x[i][j]<<" "<<y[i][j]<< " "<< u[i][j]<< " "<<v[i][j]<<endl;}
        cout<<endl;
        }
      outdata.close();
      double end=MPI_Wtime();
       cout << end-start << endl; 


   }
  MPI_Finalize();
  return 0;

}
