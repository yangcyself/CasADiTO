#include "IpIpoptApplication.hpp"
#include "tonlp.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

using namespace Ipopt;

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");


int main(
   int    /*argv*/,
   char** /*argc*/
)
{
   // Create a new instance of your nlp
   //  (use a SmartPtr, not raw)
   Eigen::MatrixXd X,U,T;
   // double distance = 0.8;
   double hp_cost_u = 0.1;
   double hp_cost_ddq = 0.0001;
   double hp_cost_qreg = 0.1;
   double hp_terrianPoints[] = {-2, 0,    0.45, 0.47, 0.6,  0.8,  2,
                               0.0, 0.0,  0.0,  0.3,  0.3,  0.3,  0.3};
   std::vector<std::pair<std::string, const double* > > hp = {
      // std::make_pair("distance", &hp_dist)
      std::make_pair("costU", &hp_cost_u),
      std::make_pair("costDDQ", &hp_cost_ddq),
      std::make_pair("costQReg", &hp_cost_qreg),
      std::make_pair("terrianPoints", hp_terrianPoints)
   };

   SmartPtr<TNLP> mynlp = new TONLP(X,U,T, hp);

   // Create a new instance of IpoptApplication
   //  (use a SmartPtr, not raw)
   // We are using the factory, since this allows us to compile this
   // example with an Ipopt Windows DLL
   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

   // Change some options
   // Note: The following choices are only examples, they might not be
   //       suitable for your optimization problem.
   app->Options()->SetNumericValue("tol", 1e-7);
   app->Options()->SetStringValue("mu_strategy", "monotone"); // from casadi document, IMPORTANT! related to coredump
   // default is monotone

   // Initialize the IpoptApplication and process the options
   ApplicationReturnStatus status;
   status = app->Initialize();
   if( status != Solve_Succeeded )
   {
      std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
      return (int) status;
   }

   // Ask Ipopt to solve the problem
   status = app->OptimizeTNLP(mynlp);

   if( status == Solve_Succeeded )
   {
      std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
   }
   else
   {
      std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
   }


   std::ofstream fox("X_out.csv");
   fox << X.transpose().format(CSVFormat);

   std::ofstream fou("U_out.csv");
   fou << U.transpose().format(CSVFormat);

   std::ofstream fot("T_out.csv");
   fot << T.transpose().format(CSVFormat);

   // As the SmartPtrs go out of scope, the reference count
   // will be decremented and the objects will automatically
   // be deleted.

   return (int) status;
}
// [MAIN]
