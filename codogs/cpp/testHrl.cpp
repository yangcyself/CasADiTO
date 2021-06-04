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
   Eigen::MatrixXd newx;
   double r[] = {1,1,1};
   double pc[] = {-1,1, -1,-1, 1,0};
   double Q[] = {1,0,0, 0,1,0, 0,0,1};
   double xold[] = {0,0,0};
   double pa[] = {-2.05,1, -2.05,-1, -0.05,0};
   
   int status = 1;
   for(int i=0; i< 10; i++){
    pa[0] -= 0.001;
    status *= HrlDynm(r, pc, Q, xold, pa, newx);
    std::cout << newx <<std::endl;
   }
   return (int) status;
}
// [MAIN]
