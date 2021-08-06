#include "dogController.h"
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

   std::vector<double> x0 = {0,0,0};
   std::vector<double> refTraj = {1,0, 
                                 1, 0.1, 
                                 1, 0.2, 
                                 1, 0.3, 
                                 1, 0.4, 
                                 1, 0.5, 
                                 1, 0.6, 
                                 1, 0.7, 
                                 1, 0.8, 
                                 1, 0.9};

   parseOutput::x_plot x_out;
   parseOutput::u_plot p_out;

   dogController(x0.data(), refTraj.data(),
            x_out, p_out);

   int status = 1;

   std::cout<<x_out<<std::endl;
   return (int) status;
}
// [MAIN]
