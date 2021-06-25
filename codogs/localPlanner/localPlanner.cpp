#include "IpIpoptApplication.hpp"
#include "tonlp.h"
#include "localPlanner.h"

#include <vector>
#include <utility>
#include <cstring> //memcpy
#include "stdio.h"
/**
 * A wrapper class of ipopt app, takes care of all the init work
 */
using namespace Ipopt;
class localPlanApp{
public:
    double r[3]; 
    double pc[6]; 
    double Q[9]; 
    double X0[3];
    double Xdes[3];
    double pa0[6];
    double normAng[3];
    double cylinderObstacles[6];

    Eigen::MatrixXd x_out;
    Eigen::MatrixXd p_out;
    localPlanApp()
        :_app(IpoptApplicationFactory()), 
         _mynlp(new TONLP(x_out, p_out, {
                std::make_pair("X0", X0),
                std::make_pair("Xdes", Xdes),
                std::make_pair("pa0", pa0),
                std::make_pair("pc", pc),
                std::make_pair("Q", Q),
                std::make_pair("r", r),
                std::make_pair("normAng", normAng),
                std::make_pair("cylinderObstacles", cylinderObstacles)
                })) {
        std::printf("***IPOPT APP SUCCEFFULLY LOADED***\n");
        _app->Options()->SetNumericValue("tol", 1e-7);
        _app->Options()->SetStringValue("mu_strategy", "monotone"); // from casadi document, IMPORTANT! related to coredump
        _app->Options()->SetIntegerValue ("print_level", 0); // from casadi document, IMPORTANT! related to coredump
    }
    SmartPtr<IpoptApplication> app(){return _app;}
    SmartPtr<TNLP> mynlp(){return _mynlp;}
private:
    SmartPtr<IpoptApplication> _app;
    SmartPtr<TNLP> _mynlp;
};


int localPlanner(const double r[3], const double pc[6], const double Q[9], 
            const double X0[3], const double Xdes[3], const double pa0[6], 
            const double normAng[3], const double cylinderObstacles[6], 
            double x_out[90], double u_out[180])
{
    static localPlanApp a;
    const auto app = a.app();
    ApplicationReturnStatus status;
    
    std::memcpy(a.X0, X0,   3 * sizeof(double));
    std::memcpy(a.Xdes, Xdes,   3 * sizeof(double));
    std::memcpy(a.pa0, pa0,   6 * sizeof(double));
    std::memcpy(a.pc, pc,   6 * sizeof(double));
    std::memcpy(a.Q, Q,   9 * sizeof(double));
    std::memcpy(a.r, r,   3 * sizeof(double));
    std::memcpy(a.normAng, normAng,   3 * sizeof(double));
    std::memcpy(a.cylinderObstacles, cylinderObstacles,   6 * sizeof(double));
    
    status = app->Initialize();
    if( status != Solve_Succeeded )
    {
        std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
        return (int) status;
    }

    // Ask Ipopt to solve the problem
    status = app->OptimizeTNLP(a.mynlp());

    if( status == Solve_Succeeded || status == Solved_To_Acceptable_Level)
    {
        // std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
    }
    else
    {
        std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
    }


    std::cout<<"x_out"<<a.x_out<<std::endl;

    return (int) status;
}