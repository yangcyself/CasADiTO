#include "localPlanner.h"
#include <Eigen/Core>
#include <iostream>
#include <vector>

class Problem{
public:
   Problem(
      std::vector<double> X0,
      std::vector<double> Xdes,
      std::vector<double> pa0,
      std::vector<double> boxObstacles
   ):_X0(X0), _Xdes(Xdes), _pa0(pa0), _boxObstacles(boxObstacles){}

   double* X0(){return _X0.data();}
   double* Xdes(){return _Xdes.data();}
   double* pa0(){return _pa0.data();}
   double* boxObstacles(){return _boxObstacles.data();}

private:
   std::vector<double> _X0;
   std::vector<double> _Xdes;
   std::vector<double> _pa0;
   std::vector<double> _boxObstacles;
};

int main(
   int    /*argv*/,
   char** /*argc*/
)
{
   const double pi = acos(-1);
   // Create a new instance of your nlp
   //  (use a SmartPtr, not raw)
   // Eigen::MatrixXd newx;

   std::vector<Problem> problems;

   problems.push_back(Problem(
      {0,0,0},
      {0.8,0, 0.1},
      {-1.8,0,  0,1.8,  0,-1},
      {0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0})
   );

   problems.push_back(Problem(
      {0,0,0},
      {1,0, 0.1},
      {-1.5,0,  0,1.5,  0,-1.5},
      {0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0})
   );

   problems.push_back(Problem(
      {0,0,0},
      {0.5,0, 2},
      {-1.5,0,  0,1.5,  0,-1.5},
      {0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0})
   );

   problems.push_back(Problem(
      {0,0,0},
      {0.5,0, 2.5},
      {-1.2,0,  0,1.8,  0,-1.5},
      {0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0})
   );

   problems.push_back(Problem(
      {0,0,0},
      {0.5,0, 3},
      {-1.5,0,  0,1.5,  0,-1.5},
      {0,0,0,0,0,  0,0,0,0,0,  0,0,0,0,0})
   );

   problems.push_back(Problem(
      {0,0,0},
      {1,0, 2},
      {-1.5,0,  0,1.5,  0,-1.5},
      {3,3,0,1,1,  0,0,0,0,0,  0,0,0,0,0})
   );

   problems.push_back(Problem(
      {0,0,0},
      {1.2,0, 3},
      {-1.5,0,  0,1.5,  0,-1.5},
      {3,3,0,1,1,  0,0,0,0,0,  0,0,0,0,0})
   );

   // std::vector<double> X0 = {0,0,0};
   // std::vector<double> Xdes = {0.5,0, 2};
   // std::vector<double> pa0 = {-1,0,  0,1,  0,-1};
   double pc[] = {-1,0, 0,1, 0,-1};
   double Q[] = {1,0,0, 0,1,0, 0,0,3};
   double r[] = {1,1,1};
   double normAng[] = {pi,pi/2,-pi/2};
   // double boxObstacles[] = {3,3,0,1,1,  0,0,0,0,0,  0,0,0,0,0};

   parseOutput::x_plot x_out;
   parseOutput::u_plot p_out;

   for(auto p: problems){
      localPlanner(p.X0(), p.Xdes(), p.pa0(), pc, Q, r, normAng, p.boxObstacles(),
               x_out, p_out);
   }
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
