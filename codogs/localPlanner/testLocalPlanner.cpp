#include "localPlanner.h"
#include <Eigen/Core>
#include <iostream>
int main(
   int    /*argv*/,
   char** /*argc*/
)
{
   const double pi = acos(-1);
   // Create a new instance of your nlp
   //  (use a SmartPtr, not raw)
   // Eigen::MatrixXd newx;

   double X0[] = {0,0,0};
   double Xdes[] = {2,0,0};
   double pa0[] = {-1,0,  0,1,  0,-1};
   double pc[] = {-1,0, 0,1, 0,-1};
   double Q[] = {1,0,0, 0,1,0, 0,0,0.1};
   double r[] = {1,1,1};
   double normAng[] = {pi,pi/2,-pi/2};
   double cylinderObstacles[] = {0, 0, 0, 0,0,0};


   double x_out[90];
   double p_out[180];

   localPlanner(X0, Xdes, pa0, pc, Q, r, normAng, cylinderObstacles, 
            x_out, p_out);

   int status = 1;

   // for(int i=0; i< 10; i++){
   //  pa[0] += 0.1;
   //  status *= HrlDynm(r, pc, Q, xold, pa, newx);
   //  std::cout << newx[0]<<" " << newx[1]<<" " << newx[2]<<" " <<std::endl;
   //  std::memcpy(xold, newx,   3 * sizeof(double));

   // }
   return (int) status;
}
// [MAIN]
