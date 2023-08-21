#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <fstream> 
#include <blitz/array.h>
#include <ctime>
#include <chrono>

using namespace std;
using namespace blitz;
using namespace std::chrono;
///////////////Specify dimension
 const int dim=3;
 ///////////////////////////////


int main(int argc, char *argv[])
{ 



////////////////////Grid size
  Array <int,1>  N(dim);
  for (int i=0;i<dim;i++)
  {
      N(i)=stoi(argv[i+1]);
      
  }
///////////////////////////
 
  int k,l,m,itr,iteration;
  double kin_visc=0.01;
  Array <double,1> L(dim);L=2;
  Range all = Range::all();
  Array <double,1> dx(dim);
  dx(all)=L/N(all);
  double dt=min(dx)/8;
  iteration=2000;
  //time_t start = time(NULL) ;
  auto start = high_resolution_clock::now();
  Array <int,1> N1(dim+1);
  N1(dim)=dim;
  N1(Range(0,dim-1))=N(Range(0,toEnd))+2;
  Array <double,dim+1> u(N1.data()),u_mid(N1.data());
  double T=0.3;
  Array <int,1> I1(0.5/dx+1);
  Array <int,1> I2(1/dx);

std::ofstream ofs;
ofs.open("Results_BG_2D_1.txt", std::ofstream::out | std::ofstream::trunc);
ofs.close();

/////////////////////////////////////////////////////////////////////////////////////
    //Initial condition
    firstIndex i;
    secondIndex j;

if (dim==3)
    {u(all,all,all,all)=0;
    u(Range(I1(0)+1,I2(0)+1),Range(I1(1)+1,I2(1)+1),Range(I1(2)+1,I2(2)+1),all)=0.5;}
if (dim==2)
    {u(all,all,all)=0;
    u(Range(I1(0)+1,I2(0)+1),Range(I1(1)+1,I2(1)+1),all)=0.5;}
//u=i;

//////////////////////////////////////////////////////////////////////////////////


    long int row_data=u.numElements()/u.extent(0);

     for(itr=0;itr<=iteration;itr++)
      {
       
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
        //Euler scheme     
          double global_max=max(u);

       if (itr%10==0)
       {   dt=(0.8/dim)*(min(min(dx)/global_max,0.5*min(dx)*min(dx)/kin_visc));

         cout<<itr << "   "<<dt<<"  "<<global_max<<endl;
       }
          if (dim==3)
              {
                  Range i(1,N(0)), j(1,N(1)), k(1,N(2));

                //////////////////////////////////////////////////////Boundary condition      
                  u(all,N(1)+1,all,all)=u(all,1,all,all);
                  u(all,0,all,all)=u(all,N(1),all,all);
                  u(all,all,N(2)+1,all)=u(all,all,1,all);
                  u(all,all,0,all)=u(all,all,N(2),all);
                  u(0,all,all,all)=u(N(0),all,all,all);
                  u(N(1)+1,all,all,all)=u(1,all,all,all);
              //////////////////////////////////////////////////////////////////
                  u_mid=u(all,all,all,all);
                  
                  for (int m=0;m<dim;m++)
                      u(i,j,k,m) = u_mid(i,j,k,m) -(dt/(2*dx(0)))*u_mid(i,j,k,0)*(u_mid(i+1,j,k,m)-u_mid(i-1,j,k,m)) \
		                    + (kin_visc*dt/(dx(0)*dx(0)))*(u_mid(i+1,j,k,m)-2*u_mid(i,j,k,m)+u_mid(i-1,j,k,m))   \
                        - (dt/(2*dx(1)))*u_mid(i,j,k,1)*(u_mid(i,j+1,k,m)-u_mid(i,j-1,k,m)) \
                        + (kin_visc*dt/(dx(1)*dx(1)))*(u_mid(i,j+1,k,m)-2*u_mid(i,j,k,m)+u_mid(i,j-1,k,m))\
                        - (dt/(2*dx(2)))*u_mid(i,j,k,2)*(u_mid(i,j,k+1,m)-u_mid(i,j,k-1,m)) \
                        + (kin_visc*dt/(dx(2)*dx(2)))*(u_mid(i,j,k+1,m)-2*u_mid(i,j,k,m)+u_mid(i,j,k-1,m));
              }

          if (dim==2)
              {

                  Range i(1,N(0)), j(1,N(1));

                  //////////////////////////////////////////Boundary Condition
                  u(all,N(1)+1,all)=u(all,1,all);
                  u(all,0,all)=u(all,N(1),all);
                  u(0,all,all)=u(N(0),all,all);
                  u(N(1)+1,all,all)=u(1,all,all);
                  /////////////////////////////////////////////////////////////

                  u_mid=u(all,all,all);

                  for (int m=0;m<dim;m++)
                      u(i,j,m) = u_mid(i,j,m) -(dt/(2*dx(0)))*u_mid(i,j,0)*(u_mid(i+1,j,m) -u_mid(i-1,j,m)) \
		                  + (kin_visc*dt/(dx(0)*dx(0)))*(u_mid(i+1,j,m)-2*u_mid(i,j,m)+u_mid(i-1,j,m))   \
                      - (dt/(2*dx(1)))*u_mid(i,j,1)*(u_mid(i,j+1,m)-u_mid(i,j-1,m)) \
		                  + (kin_visc*dt/(dx(1)*dx(1)))*(u_mid(i,j+1,m)-2*u_mid(i,j,m)+u_mid(i,j-1,m));
    
              }

    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //time_t   end=time(NULL); 
auto stop = high_resolution_clock::now();
auto duration = duration_cast<microseconds>(stop - start);
cout<<duration.count()*1.0/1e+6<<endl;
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////  I/O
{
      ofstream outdata("Results_BG_2D_1.txt");
        outdata<<std::setprecision(16)<< u<<endl;
      outdata.close(); 
      
}
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  return 0;

}