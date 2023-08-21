///////Its a parallel code which uses MPI for solving 2D or 3D Burgers equation using Euler scheme. Compile using -O3/1/2/fast option. It uses Blitz++ for arrays. 
//////When  running the code please also mention grid size along with it. example: mpiexec -np 4 ./a.out 100 100 100 
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <fstream>
#include <mpi.h>    
#include <blitz/array.h>


using namespace std;
using namespace blitz;

///////////////Specify dimension
 const int dim=2;
 ///////////////////////////////


int main(int argc, char *argv[])
{ 
  int       nprocs;              
  int       rank;               
  MPI_Request reqs[4];   
  MPI_Status stats[4];
  MPI_Status status;



////////////////////Grid size
  Array <int,1>  N(dim);
  int i=0;
  for (i=0;i<dim;i++)
  {
      N(i)=stoi(argv[i+1]);
      
  }
///////////////////////////

///////////////////////Initialize MPI
  MPI_Init(NULL, NULL); 

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
//////////////////////////////////
 
  int j,k,l,m,itr,iteration;

  ////data chunks for each processor
  int count=N(0)/(nprocs);
  
  ///////Viscosity
  double kin_visc=0.01;

  /////Range of X and Y coordinates
  Array <double,1> L(dim);L=2;

  Range all = Range::all();
  
  /////size of each grid box
  Array <double,1> dx(dim);
  dx(all)=L/N(all);

  /////each timestep
  double dt=min(dx)/8;

  ///no. of iterations
  iteration=2000;

  ////start MPI time
  double start = MPI_Wtime();

  //// Defines grid size for array of data used by each processor. N1 is for the total data and N2 for the chunks of data for each processor
  Array <int,1> N1(dim+1),N2(dim+1);
  N1(dim)=N2(dim)=dim; N2(0)=count+2; N1(0)=N(0);
  N1(Range(1,dim-1))=N(Range(1,toEnd))+2;
  N2(Range(1,dim-1))=N(Range(1,toEnd))+2;
  
  
  //////u_new and u_mid will be used by each processor while u will be used by root only. Initially size of u is null and will be defined later by root processor only
  ///u(i,j,k,l): means velocity u_l (where l can be 1,2 or 3) at grid points i, j, k
  Array <double,dim+1> u_new(N2.data()),u_mid(N2.data());
  Array <double,dim+1> u;



if (rank==0)
{    
/////Define the size of u
    u.resize(N1.data());

/////////////////////////////////////////////////////////////////////////////////////
    //Initial condition for shock at the center. user may change it according to his/her requirements.   
  Array <int,1> I1(0.5/dx+1);
  Array <int,1> I2(1/dx);

////for 3D shock
if (dim==3)
    {u(all,all,all,all)=0;
    u(Range(I1(0),I2(0)),Range(I1(1)+1,I2(1)+1),Range(I1(2)+1,I2(2)+1),all)=0.5;}
/////for 2D shock
if (dim==2)
    {u(all,all,all)=0;
    u(Range(I1(0),I2(0)),Range(I1(1)+1,I2(1)+1),all)=0.5;}


//////////////////////////////////////////////////////////////////////////////////

}
////////Scatter data to each procesor

    long int row_data=u_new.numElements()/u_new.extent(0);

     MPI_Scatter (u.data(),count*(row_data),MPI_DOUBLE,u_new.data()+row_data,count*(row_data),MPI_DOUBLE,0,MPI_COMM_WORLD);

     MPI_Barrier(MPI_COMM_WORLD);

    int itr_count=0;
///////////Time loop 

     for(itr=0;itr<=iteration;itr++)
      {
    ////Communication with the neighbours

          MPI_Isend(u_new.data()+row_data, row_data, MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[0]); 
          MPI_Isend(u_new.data()+(count)*(row_data), row_data, MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[1]); 
          MPI_Irecv(u_new.data()+(count+1)*(row_data), (row_data), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[3]); 
          MPI_Irecv(u_new.data(), (row_data), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[2]); 

          MPI_Waitall(4, reqs, stats);
       
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
        //Euler scheme starts from here. RK-2 scheme is available in an alternate code

        //Dynamic updation of time. This step requires communication between each processor and hence is expensive.
        ///// That's why this step is carried only after 10 iterations. User can choose to change it and also can change courant number 
        ///// if scheme becomes unstable or too slow     
        if (itr%10==0){
          double local_max=max(u_new);
          double global_max=1e-10;
          MPI_Reduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD) ;

          if (rank==0) { dt=(0.8/dim)*(min(min(dx)/global_max,0.5*min(dx)*min(dx)/kin_visc)); }

          MPI_Bcast(&dt,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        
            if(rank==0) cout<<itr << "   "<<dt<<"  "<<global_max<<endl;
        }

        /////Euler time marching for 3D
          if (dim==3)
              {
                  Range i(1,count), j(1,N(1)), k(1,N(2));

                //////////////////////////////////////////////////////Boundary condition for Periodic boundary     
                  u_new(all,N(1)+1,all,all)=u_new(all,1,all,all);
                  u_new(all,0,all,all)=u_new(all,N(1),all,all);
                  u_new(all,all,N(2)+1,all)=u_new(all,all,1,all);
                  u_new(all,all,0,all)=u_new(all,all,N(2),all);
                //////////////////////////////////////////////////////////////////
                  u_mid=u_new(all,all,all,all);
                  
                  for (int m=0;m<dim;m++)
                      u_new(i,j,k,m) = u_mid(i,j,k,m) -(dt/(2*dx(0)))*u_mid(i,j,k,0)*(u_mid(i+1,j,k,m)-u_mid(i-1,j,k,m)) \
		                    + (kin_visc*dt/(dx(0)*dx(0)))*(u_mid(i+1,j,k,m)-2*u_mid(i,j,k,m)+u_mid(i-1,j,k,m))   \
                        - (dt/(2*dx(1)))*u_mid(i,j,k,1)*(u_mid(i,j+1,k,m)-u_mid(i,j-1,k,m)) \
                        + (kin_visc*dt/(dx(1)*dx(1)))*(u_mid(i,j+1,k,m)-2*u_mid(i,j,k,m)+u_mid(i,j-1,k,m))\
                        - (dt/(2*dx(2)))*u_mid(i,j,k,2)*(u_mid(i,j,k+1,m)-u_mid(i,j,k-1,m)) \
                        + (kin_visc*dt/(dx(2)*dx(2)))*(u_mid(i,j,k+1,m)-2*u_mid(i,j,k,m)+u_mid(i,j,k-1,m));
              }

         //////////////Euler time marching for 2D case     

          if (dim==2)
              {

                  Range i(1,count), j(1,N(1));

                  //////////////////////////////////////////Boundary Condition for periodic boundary
                  u_new(all,N(1)+1,all)=u_new(all,1,all);
                  u_new(all,0,all)=u_new(all,N(1),all);
                  /////////////////////////////////////////////////////////////

                  u_mid=u_new(all,all,all);

                  for (int m=0;m<dim;m++)
                      u_new(i,j,m) = u_mid(i,j,m) -(dt/(2*dx(0)))*u_mid(i,j,0)*(u_mid(i+1,j,m) -u_mid(i-1,j,m)) \
		                  + (kin_visc*dt/(dx(0)*dx(0)))*(u_mid(i+1,j,m)-2*u_mid(i,j,m)+u_mid(i-1,j,m))   \
                      - (dt/(2*dx(1)))*u_mid(i,j,1)*(u_mid(i,j+1,m)-u_mid(i,j-1,m)) \
		                  + (kin_visc*dt/(dx(1)*dx(1)))*(u_mid(i,j+1,m)-2*u_mid(i,j,m)+u_mid(i,j-1,m));
    
              }

          /////Parallel I/O for writing in the file. Currently only initial and final result is written but user can write after any number of time steps

          if (itr%iteration==0)
              {
                  double end=MPI_Wtime();
                  if (rank==0)cout<< "run time   "<<end-start<<endl;
                  double start =MPI_Wtime();
                  MPI_File fh;
                  MPI_File_open(MPI_COMM_WORLD,"Results_BG_2D.txt",MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
                  MPI_Offset fo=(MPI_Offset)(itr_count*N(0)+rank*count)*row_data*sizeof(double);
                  MPI_Barrier(MPI_COMM_WORLD);
                  MPI_File_write_at(fh,fo, u_new.data()+row_data, (count)*row_data, MPI_DOUBLE, &status);
                  MPI_File_close(&fh);
                  itr_count++;
                  end=MPI_Wtime();
                  if (rank==0) cout<<"parallel I/O timings  "<<"  "<<end-start<<endl;
              }


 }

//time loop ends

// following commented part is necessary only for serial I/O and is not necessary

/////////////////////////////////////////Gathering data to root
  //   start=MPI_Wtime(); 
   //       MPI_Gather (u_new.data()+row_data,(count)*(row_data),MPI_DOUBLE,u.data(),count*(row_data),MPI_DOUBLE,0,MPI_COMM_WORLD);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//// Serial  I/O

   /*//if (rank==0)
  // {
      ofstream outdata("Results_BG_2D_1.txt");
      outdata<< u<<endl;
      outdata.close();
      double end=MPI_Wtime();
       cout << end-start << endl; 
   }*/

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////Wait for all process and fianlize MPI
   MPI_Barrier(MPI_COMM_WORLD);
//free(u.data());
  MPI_Finalize();
  return 0;

}