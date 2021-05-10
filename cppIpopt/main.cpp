#include "IpIpoptApplication.hpp"
#include "tonlp.h"

#include <iostream>
#include <fstream>

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
   SmartPtr<TNLP> mynlp = new TONLP(X,U,T);
//    SmartPtr<TNLP> mynlp = new HS071_NLP();

   // Create a new instance of IpoptApplication
   //  (use a SmartPtr, not raw)
   // We are using the factory, since this allows us to compile this
   // example with an Ipopt Windows DLL
   SmartPtr<IpoptApplication> app = IpoptApplicationFactory();

   // Change some options
   // Note: The following choices are only examples, they might not be
   //       suitable for your optimization problem.
   app->Options()->SetNumericValue("tol", 1e-7);
   app->Options()->SetStringValue("mu_strategy", "adaptive");
   app->Options()->SetStringValue("output_file", "ipopt.out");
   // The following overwrites the default name (ipopt.opt) of the options file
   // app->Options()->SetStringValue("option_file_name", "hs071.opt");

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

   std::cout << "********* X out **********"<<std::endl;
   std::cout << X <<std::endl;

   std::cout << "********* U out **********"<<std::endl;
   std::cout << U <<std::endl;

   std::cout << "********* T out **********"<<std::endl;
   std::cout << T <<std::endl;
   
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