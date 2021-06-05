#include "hrlDyn.h"
#include <Eigen/Core>
#include <iostream>
int main(
   int    /*argv*/,
   char** /*argc*/
)
{
   // Create a new instance of your nlp
   //  (use a SmartPtr, not raw)
   // Eigen::MatrixXd newx;
   double r[] = {1,999999999,999999999};
   double pc[] = {-1,1, -1,-1, 1,0};
   double Q[] = {1,0,0, 0,1,0, 0,0,0.01};
   double xold[] = {1,-3,0};
   double pa[] = {1,-2, -2.05,-1, -0.05,0};



   double newx[3];
   // double r[] = {5,99999,99999};
   // double pc[] = {-1,1, -1,-1, 1,0};
   // double Q[] = {1,0,0, 0,1,0, 0,0,1};
   // double xold[] = {0, 0, 0};
   // double pa[] = {0,0, 0,0, 0,0};
   
   int status = 1;

   for(int i=0; i< 10; i++){
    pa[0] += 0.1;
    status *= HrlDynm(r, pc, Q, xold, pa, newx);
    std::cout << newx[0]<<" " << newx[1]<<" " << newx[2]<<" " <<std::endl;
    std::memcpy(xold, newx,   3 * sizeof(double));

   }
   return (int) status;
}
// [MAIN]
