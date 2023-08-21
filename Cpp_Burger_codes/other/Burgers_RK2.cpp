///////Its a parallel code which uses MPI for solving 2D or 3D Burgers equation using RK2 scheme. Compile using -O3/1/2/fast option. It uses Blitz++ for arrays. 
//////When  running the code please also mention grid size along with it. example: mpiexec -np 4 ./a.out 100 100 100 
//////See Burgers_parallel.cpp for more comments
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
 double kin_visc=0.01;
 double dt=.01;
 Array <double,1> dx(dim);

 ///////////////////////////////Marching in time 

void time_march(Array <double,dim+1> u_new, Array <double,dim+1> u_mid,int count,Array <int,1>  N,double weight)
{
   Range i(1,count), j(1,N(1)), k(1,N(2));

for (int m=0;m<dim;m++)  
{
    if (dim==2){
    u_mid(i,j,m) =(1-weight)*u_mid(i,j,m) + weight*(u_new(i,j,m) -(dt/(2*dx(0)))*u_new(i,j,0)*(u_new(i+1,j,m) -u_new(i-1,j,m)) \
		        + (kin_visc*dt/(dx(0)*dx(0)))*(u_new(i+1,j,m)-2*u_new(i,j,m)+u_new(i-1,j,m))   \
              - (dt/(2*dx(1)))*u_new(i,j,1)*(u_new(i,j+1,m)-u_new(i,j-1,m)) \
		        + (kin_visc*dt/(dx(1)*dx(1)))*(u_new(i,j+1,m)-2*u_new(i,j,m)+u_new(i,j-1,m)));}
if (dim==3){
    u_mid(i,j,k,m) =(1-weight)*u_mid(i,j,k,m) + weight*(u_new(i,j,k,m) -(dt/(2*dx(0)))*u_new(i,j,k,0)*(u_new(i+1,j,k,m)-u_new(i-1,j,k,m)) \
		        + (kin_visc*dt/(dx(0)*dx(0)))*(u_new(i+1,j,k,m)-2*u_new(i,j,k,m)+u_new(i-1,j,k,m))   \
              - (dt/(2*dx(1)))*u_new(i,j,k,1)*(u_new(i,j+1,k,m)-u_new(i,j-1,k,m)) \
		        + (kin_visc*dt/(dx(1)*dx(1)))*(u_new(i,j+1,k,m)-2*u_new(i,j,k,m)+u_new(i,j-1,k,m))\
                - (dt/(2*dx(2)))*u_new(i,j,k,2)*(u_new(i,j,k+1,m)-u_new(i,j,k-1,m)) \
		        + (kin_visc*dt/(dx(2)*dx(2)))*(u_new(i,j,k+1,m)-2*u_new(i,j,k,m)+u_new(i,j,k-1,m)));
}
}
}

////Boundary Condition
void boundary_condition(Array <double,dim+1> u_new,Array <int,1>  N)
{  Range all = Range::all();
if(dim==2){
    u_new(all,N(1)+1,all)=u_new(all,1,all);
    u_new(all,0,all)=u_new(all,N(1),all);
}

if(dim==3){
          u_new(all,N(1)+1,all,all)=u_new(all,1,all,all);
          u_new(all,0,all,all)=u_new(all,N(1),all,all);
          u_new(all,all,N(2)+1,all)=u_new(all,all,1,all);
          u_new(all,all,0,all)=u_new(all,all,N(2),all);
}
          
}




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


  MPI_Init(NULL, NULL); 

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); 

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

 
  int j,k,l,m,itr,iteration;
  int count=N(0)/(nprocs);
  Array <double,1> L(dim);L=2;
  Range all = Range::all();
  dx(all)=L/N(all);
  iteration=1000;
  double start = MPI_Wtime(); 
  Array <int,1> N1(dim+1),N2(dim+1);
  N1(dim)=N2(dim)=dim; N2(0)=count+2; N1(0)=N(0);
  N1(Range(1,dim-1))=N(Range(1,toEnd))+2;
  N2(Range(1,dim-1))=N(Range(1,toEnd))+2;
  Array <double,dim+1> u_new(N2.data()),u_mid(N2.data());
  Array <double,dim+1> u;

  double T=0.3;
   
  Array <int,1> I1(0.5/dx+1);
  Array <int,1> I2(1/dx);

if (rank==0)
{    

    u.resize(N1.data());

/////////////////////////////////////////////////////////////////////////////////////
    //Initial condition
    firstIndex i;
    secondIndex j;

if (dim==3){
    u=0;
    u(Range(I1(0),I2(0)),Range(I1(1)+1,I2(1)+1),Range(I1(2)+1,I2(2)+1),all)=2;
}
if (dim==2)
    {
    u(all,all,all)=0;
    u(Range(I1(0),I2(0)),Range(I1(1)+1,I2(1)+1),all)=0.5;}

//////////////////////////////////////////////////////////////////////////////////

}
long int row_data=u_new.numElements()/u_new.extent(0);

MPI_Scatter (u.data(),count*(row_data),MPI_DOUBLE,u_new.data()+row_data,count*(row_data),MPI_DOUBLE,0,MPI_COMM_WORLD);

MPI_Barrier(MPI_COMM_WORLD);

int itr_count=0;

for(itr=1;itr<=iteration;itr++)
    {
          
        MPI_Isend(u_new.data()+row_data, row_data, MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[0]); 
        MPI_Isend(u_new.data()+(count)*(row_data), row_data, MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[1]); 
        MPI_Irecv(u_new.data()+(count+1)*(row_data), (row_data), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[3]); 
        MPI_Irecv(u_new.data(), (row_data), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[2]); 

        MPI_Waitall(4, reqs, stats);
 
        double local_max=max(u_new);
        double global_max=1e-10;
        MPI_Reduce(&local_max,&global_max,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD) ;

       if(itr%10==0)
    { 
        if (rank==0) 
    
            dt=(0.8/dim)*(min(min(dx)/global_max,0.5*min(dx)*min(dx)/kin_visc));
    
        MPI_Bcast(&dt,1,MPI_DOUBLE,0,MPI_COMM_WORLD);

        if(rank==0) cout<<itr << "   "<<dt<<"  "<<global_max<<endl;
    }
    /////Predictor
        boundary_condition(u_new,N);

        time_march(u_new,u_mid,count,N,1);

        MPI_Isend(u_mid.data()+row_data, row_data, MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[0]); 
        MPI_Isend(u_mid.data()+(count)*(row_data), row_data, MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[1]); 
        MPI_Irecv(u_mid.data()+(count+1)*(row_data), (row_data), MPI_DOUBLE, (rank+1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[3]); 
        MPI_Irecv(u_mid.data(), (row_data), MPI_DOUBLE,(rank-1+nprocs)%nprocs , 1, MPI_COMM_WORLD,&reqs[2]); 

        MPI_Waitall(4, reqs, stats);
////////corrector
        boundary_condition(u_mid,N);

        time_march(u_mid,u_new,count,N,0.5);


        if (itr%iteration==0){
            double start =MPI_Wtime();
            MPI_File fh;
            MPI_File_open(MPI_COMM_WORLD,"Results_BG_2D.txt",MPI_MODE_CREATE | MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
            MPI_Offset fo=(MPI_Offset)(itr_count*N(0)+rank*count)*row_data*sizeof(double);
            MPI_File_write_at_all(fh,fo, u_new.data()+row_data, (count)*row_data, MPI_DOUBLE, &status);
            MPI_Barrier(MPI_COMM_WORLD);
            MPI_File_close(&fh);
            itr_count++;
            double end=MPI_Wtime();
            if (rank==3) cout<<"parallel"<<"  "<<end-start<<endl;
              }


 }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
       //   MPI_Gather (u_new.data()+row_data,(count)*(row_data),MPI_DOUBLE,u.data(),count*(row_data),MPI_DOUBLE,0,MPI_COMM_WORLD);

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
  return 0;

}