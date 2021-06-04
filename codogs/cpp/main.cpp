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
   Eigen::MatrixXd newx;
   double r[] = {1,1,1};
   double pc[] = {-1,1, -1,-1, 1,0};
   double Q[] = {1,0,0, 0,1,0, 0,0,1};
   double xold[] = {0,0,0};
   double pa[] = {-2.05,1, -2.05,-1, -0.05,0};
   std::vector<std::pair<std::string, const double* > > hp = {
      std::make_pair("r", r),
      std::make_pair("pc", pc),
      std::make_pair("Q", Q),
      std::make_pair("xold", xold),
      std::make_pair("pa", pa)
   };

   SmartPtr<TNLP> mynlp = new TONLP(newx, hp);

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
   app->Options()->SetIntegerValue ("print_level", 0); // from casadi document, IMPORTANT! related to coredump
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


   std::cout<<"newx"<<newx<<std::endl;
   // As the SmartPtrs go out of scope, the reference count
   // will be decremented and the objects will automatically
   // be deleted.

   return (int) status;
}
// [MAIN]
