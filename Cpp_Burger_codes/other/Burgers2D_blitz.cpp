///////The code solves 2D Burgers equation using MPI. It uses Blitz++ for arrays and is easily scalable for big data points and large number of processors. 

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <fstream>
#include <mpi.h>    
#include <blitz/array.h>


using namespace std;
using namespace blitz;


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
  Array <double,2> u_new(count+2,N+2),v_new(count+2,N+2),recv_u(count+2,N+2),recv_v(count+2,N+2);

  Array <double,2> u,v,x,y;

  
  double T=0.3;
   
  int u1=0.5/dx+1;
  int u2=1/dx;
  int v1=0.5/dy+1;
  int v2=1/dy;

if (rank==0)
{    //Range i(0,N),j(0,N+1);
firstIndex i;
secondIndex j;
    u.resize(N,N+2);
    v.resize(N,N+2);
    u=i+j,v=0;
    u(Range(u1,u2),Range(v1,v2))=2;
    v(Range(u1,u2),Range(v1,v2))=2;

}

     MPI_Scatter (u.data(),count*(N+2),MPI_DOUBLE,u_new.data()+N+2,count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
     MPI_Scatter (v.data(),count*(N+2),MPI_DOUBLE,v_new.data()+N+2,count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);

   // if (rank==1) cout<< u_new<<endl;

     for(itr=1;itr<=iteration;itr++)
      {
          MPI_Isend(u_new.data()+N+2, (N+2), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[0]); 
          MPI_Isend(u_new.data()+count*(N+2), (N+2), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[1]); 
          MPI_Isend(v_new.data()+N+2, (N+2), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 2, MPI_COMM_WORLD,&reqs[2]); 
          MPI_Isend(v_new.data()+count*(N+2), (N+2), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 2, MPI_COMM_WORLD,&reqs[3]); 
          MPI_Irecv(u_new.data(), (N+2), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[4]); 
          MPI_Irecv(u_new.data()+(count+1)*(N+2), (N+2), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[5]); 
          MPI_Irecv(v_new.data(), (N+2), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 2, MPI_COMM_WORLD,&reqs[6]); 
          MPI_Irecv(v_new.data()+(count+1)*(N+2), (N+2), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 2 , MPI_COMM_WORLD,&reqs[7]); 
          MPI_Waitall(8, reqs, stats);

          recv_u=u_new(Range::all(),Range::all());
          recv_v=v_new(Range::all(),Range::all());

          Range i(1,count),j(1,N);

            u_new(i,j) = recv_u(i,j) - (dt/(2*dx))*recv_u(i,j)*(recv_u(i+1,j)-recv_u(i-1,j)) \
			        + (kin_visc*dt/(dx*dx))*(recv_u(i+1,j)-2*recv_u(i,j)+recv_u(i-1,j)) \
              - (dt/(2*dy))*recv_v(i,j)*(recv_u(i,j+1)-recv_u(i,j-1)) \
			        + (kin_visc*dt/(dy*dy))*(recv_u(i,j+1)-2*recv_u(i,j)+recv_u(i,j-1));

            v_new(i,j) = recv_v(i,j) - (dt/(2*dx))*recv_v(i,j)*(recv_v(i+1,j)-recv_v(i-1,j)) \
			        + (kin_visc*dt/(dx*dx))*(recv_v(i+1,j)-2*recv_v(i,j)+recv_v(i-1,j)) \
              -(dt/(2*dy))*recv_v(i,j)*(recv_v(i,j+1)-recv_v(i,j-1)) \
			        + (kin_visc*dt/(dy*dy))*(recv_v(i,j+1)-2*recv_v(i,j)+recv_v(i,j-1));   

      }
          MPI_Gather (u_new.data()+N+2,(count)*(N+2),MPI_DOUBLE,u.data(),count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
          MPI_Gather (v_new.data()+N+2,(count)*(N+2),MPI_DOUBLE,v.data(),count*(N+2),MPI_DOUBLE,0,MPI_COMM_WORLD);
   if (rank==0)
   {
  //    ofstream outdata("Results_BG_2D.txt");
    //  for(i=0;i<N;i++)
      //{
        //for(j=1;j<N+1; j++)
        //{//cout<<u(i,j)<<"    ";
        //outdata<<x[i,j)<<" "<<y[i,j)<< " "<< u(i,j)<< " "<<v(i,j)<<endl;}
       // cout<<endl;
        //}
      //outdata.close();
      double end=MPI_Wtime();
       cout << end-start << endl; 


   }
   MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;

}