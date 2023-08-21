////This code is the step 0 of MPI pragramming. it will not scale well and will consume much memory. The code works as follows. At each time step the master processor distributes data to each slave processor.
//// The slave processors do the computation and return the computed data to master processor. The process is repeated again and again. The processes does not talk to neighbours and communication time is quite significant. 
////But this is the simplest possible MPI for FDM.
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
  /* Start up MPI */
   MPI_Init(NULL, NULL); 

   /* Get the number of processes */
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 

   /* Get my rank among all the processes */
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
   int N=200,i,j,k,itr,iteration;
   int count=N/(nprocs-1)+2;
   double recv_u[count][N+2],recv_v[count][N+2],u_new[count][N+2],v_new[count][N+2];
   double kin_visc=0.01,L=2;
   double dx=L/N,dt=dx/8;
   double dy=L/N;
   iteration=500;
  // cout<<iteration;
   if (rank==0)
   {	double start = MPI_Wtime(); 

    //   cout<<"master"<<endl;
      //u1,u2,v1,v2;
      double T=0.3;
      double u[N+2][N+2],u_initial[N+2][N+2],x[N+2][N+2];
      double v[N+2][N+2],v_initial[N+2][N+2],y[N+2][N+2];
     
      
      int u1=0.5/dx+1;
      int u2=1/dx;
      int v1=0.5/dy+1;
      int v2=1/dy;
      
      //Setting initial condition 


      for(i=0;i<=N+1;i++)
        for(j=0;j<=N+1;j++) //Getting initialised
        {   
          x[i][j]=dx*(i-1);
          y[i][j]=dy*(j-1);


          u[i][j]= 0;
          u_initial[i][j]= 0;
          v[i][j]= 0;
          v_initial[i][j]= 0;
      
        }
      for(i=u1;i<=u2;i++)
        for(j=v1;j<=v2;j++) 
        {
          u[i][j]=2;
          u_initial[i][j]=2;
          v[i][j]=2;
          v_initial[i][j]=2;

        }
      for(itr=1;itr<=iteration;itr++)
      {  
        for (i=1;i<=nprocs-1;i++)
        {      
          MPI_Send(u[(i-1)*(count-2)], count*(N+2), MPI_DOUBLE, i, i, MPI_COMM_WORLD); 
          MPI_Send(v[(i-1)*(count-2)], count*(N+2), MPI_DOUBLE, i, i*2, MPI_COMM_WORLD); 
        }
        for(i=1;i<=nprocs-1;i++)
        { //cout<<i<<endl;
          MPI_Recv(recv_u, count*(N+2), MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Recv(recv_v, count*(N+2), MPI_DOUBLE, i, i*2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          for(j=1;j<count-1;j++)
             {
            for(k=1;k<N+1;k++)
          {
            //cout<<"    "<< recv_u[j][k]<<"    ";
            u[(i-1)*(count-2)+j][k]=recv_u[j][k];
            v[(i-1)*(count-2)+j][k]=recv_v[j][k];

          }
              //    cout<<"Next"<<endl;
                  }
        }
        for(j=1;j<=N+1;j++)
        {
          u[0][j]=u[N][j];
          v[0][j]=v[N][j];
	        u[N+1][j]=u[1][j];
          v[N+1][j]=v[1][j];
        }
  
        for(i=1;i<=N+1;i++)
        {
        u[i][0]=u[i][N];
        v[i][0]=v[i][N];
	      u[i][N+1]=u[i][1];
        v[i][N+1]=v[i][1];
        }
        

      }


      ofstream outdat("Initial_BG_2D.txt");

      for(i=1;i<N+1;i++)
        for(j=1;j<N+1;j++)
          outdat<< x[i][j]<<" "<<y[i][j] << " "<< u_initial[i][j]<< " " <<v_initial[i][j]<<endl;
      outdat.close();

      ofstream outdata("Results_BG_2D.txt");
      for(i=1;i<N+1;i++)
        for(j=1;j<N+1; j++)
        outdata<<x[i][j]<<" "<<y[i][j]<< " "<< u[i][j]<< " "<<v[i][j]<<endl;
      outdata.close();
      double end=MPI_Wtime();
       cout << end-start << endl; 
  MPI_Finalize();

   }
   else
   {
     for(itr=1;itr<=iteration;itr++)
      {//if (rank==2) cout<<itr<<endl;
        MPI_Recv(recv_u, count*(N+2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(recv_v, count*(N+2), MPI_DOUBLE, 0, rank*2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for(i=1;i<count-1;i++)  
          {for(j=1;j<=N;j++)
          {    //    if (rank==2)    cout<<"    "<< recv_u[i][j]<<"    ";

            u_new[i][j] = recv_u[i][j] - (dt/(2*dx))*recv_u[i][j]*(recv_u[i+1][j]-recv_u[i-1][j]) \
			        + (kin_visc*dt/(dx*dx))*(recv_u[i+1][j]-2*recv_u[i][j]+recv_u[i-1][j]) \
              - (dt/(2*dy))*recv_v[i][j]*(recv_u[i][j+1]-recv_u[i][j-1]) \
			        + (kin_visc*dt/(dy*dy))*(recv_u[i][j+1]-2*recv_u[i][j]+recv_u[i][j-1]);

            v_new[i][j] = recv_v[i][j] - (dt/(2*dx))*recv_v[i][j]*(recv_v[i+1][j]-recv_v[i-1][j]) \
			        + (kin_visc*dt/(dx*dx))*(recv_v[i+1][j]-2*recv_v[i][j]+recv_v[i-1][j]) \
              -(dt/(2*dy))*recv_v[i][j]*(recv_v[i][j+1]-recv_v[i][j-1]) \
			        + (kin_visc*dt/(dy*dy))*(recv_v[i][j+1]-2*recv_v[i][j]+recv_v[i][j-1]);    
          }
            //   if (rank==2)     cout<<"Next"<<endl;
             }

          MPI_Send(u_new, count*(N+2), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD); 
          MPI_Send(v_new, count*(N+2), MPI_DOUBLE, 0, rank*2, MPI_COMM_WORLD); 
 
      }
      MPI_Finalize();
   }

  return 0;

}