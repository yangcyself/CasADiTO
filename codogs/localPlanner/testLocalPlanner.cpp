#include "localPlanner.h"
#include <Eigen/Core>
#include <iostream>
#include <vector>
int main(
   int    /*argv*/,
   char** /*argc*/
)
{
   const double pi = acos(-1);
   // Create a new instance of your nlp
   //  (use a SmartPtr, not raw)
   // Eigen::MatrixXd newx;

   std::vector<double> X0 = {0,0,0};
   std::vector<double> Xdes = {1,0,0.1};
   std::vector<double> pa0 = {-1,0,  0,1,  0,-1};
   double pc[] = {-1,0, 0,1, 0,-1};
   double Q[] = {1,0,0, 0,1,0, 0,0,1};
   double r[] = {1,1,1};
   double normAng[] = {pi,pi/2,-pi/2};
   double cylinderObstacles[] = {0, 0, 0, 0,0,0};
   double lineObstacles[] = {0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0};


   parseOutput::x_plot x_out;
   parseOutput::u_plot p_out;

   localPlanner(X0.data(), Xdes.data(), pa0.data(), pc, Q, r, normAng, cylinderObstacles, lineObstacles,
            x_out, p_out);

   int status = 1;

   // for(int i=0; i< 10; i++){
   //  pa[0] += 0.1;
   //  status *= HrlDynm(r, pc, Q, xold, pa, newx);
   //  std::cout << newx[0]<<" " << newx[1]<<" " << newx[2]<<" " <<std::endl;
   //  std::memcpy(xold, newx,   3 * sizeof(double));

   // }
   std::cout<<x_out<<std::endl;
   return (int) status;
}
// [MAIN]
