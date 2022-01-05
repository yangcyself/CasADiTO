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

   std::vector<double> x0 = {0,0,1e-2, 0.3, 0, 0};
   std::vector<double> wacc = {1e3, 1e6, 10};
   configDogController(wacc.data(), 0.4, 0.1, 1e-3, 1);
   std::vector<double> refTraj = {1,2, 0.2, 0, 0};
   std::vector<double> obstacles = {0,0,0,0,0,
                                    0,0,0,0,0,
                                    0,0,0,0,0};

   parseOutput::x_plot x_out;
   parseOutput::u_plot p_out;

   dogController(x0.data(), refTraj.data(), obstacles.data(),
            x_out, p_out);

   int status = 1;

   std::cout<<x_out<<std::endl;
   return (int) status;
}
// [MAIN]
