#include "IpIpoptApplication.hpp"
#include "tonlp.h"
#include "hrlDyn.h"

#include <vector>
#include <utility>
#include <cstring> //memcpy

/**
 * A wrapper class of ipopt app, takes care of all the init work
 */
using namespace Ipopt;
class ipoptApp{
public:
    double r[3]; 
    double pc[6]; 
    double Q[9]; 
    double xold[3];
    double pa[6];
    Eigen::MatrixXd newx;
    ipoptApp()
        :_app(IpoptApplicationFactory()), 
         _mynlp(new TONLP(newx, {
                std::make_pair("r", r),
                std::make_pair("pc", pc),
                std::make_pair("Q", Q),
                std::make_pair("xold", xold),
                std::make_pair("pa", pa)})) {
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


int HrlDynm(const double r[3], const double pc[6], const double Q[9], 
            const double xold[3], const double pa[6], double newx[3])
{
    static ipoptApp a;
    const auto app = a.app();
    ApplicationReturnStatus status;

    std::memcpy(a.r, r,   3 * sizeof(double));
    std::memcpy(a.pc, pc,   6 * sizeof(double));
    std::memcpy(a.Q, Q,   9 * sizeof(double));
    std::memcpy(a.xold, xold,   3 * sizeof(double));
    std::memcpy(a.pa, pa,   6 * sizeof(double));
    
    status = app->Initialize();
    if( status != Solve_Succeeded )
    {
        std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
        return (int) status;
    }

    // Ask Ipopt to solve the problem
    status = app->OptimizeTNLP(a.mynlp());

    if( status == Solve_Succeeded )
    {
        // std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
    }
    else
    {
        std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
    }


    // std::cout<<"newx"<<a.newx<<std::endl;
    // newx = a.newx;
    newx[0] = a.newx(0);
    newx[1] = a.newx(1);
    newx[2] = a.newx(2);
    return (int) status;
}