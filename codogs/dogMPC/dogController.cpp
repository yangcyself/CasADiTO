#include "IpIpoptApplication.hpp"
#include "tonlp.h"
#include "dogController.h"

#include <vector>
#include <utility>
#include <cstring> //memcpy
#include "stdio.h"
#include <chrono> // count time
#include <ctime>  

/**
 * A wrapper class of ipopt app, takes care of all the init work
 */
using namespace Ipopt;
class dogCtrlApp{
private:
    const double _gamma = 1;
    const double _Wreference = 1e3;
    const double _Wvelref = 1e1;
    const hyperParameters::Wacc _Wacc = {1e3,1e6,2};
    const double _Wrot = 0.03;
    const double _dog_l = 0.65;
    const double _dog_w = 0.35;
    const double _Cvel_forw = 0.35;
    const double _Cvel_side = 0.1;
    SmartPtr<IpoptApplication> _app;
    SmartPtr<TNLP> _mynlp;
public:
    hyperParameters::x0 x0;
    hyperParameters::refTraj refTraj;
    hyperParameters::obstacles obstacles;
    parseOutput::x_plot x_out;
    parseOutput::u_plot p_out;

    dogCtrlApp(): 
        _app(IpoptApplicationFactory()), 
        _mynlp(new TONLP(x_out, p_out, {
                std::make_pair("x0", x0),
                std::make_pair("refTraj", refTraj),
                std::make_pair("dog_l", &_dog_l),
                std::make_pair("dog_w", &_dog_w),
                std::make_pair("obstacles", obstacles),
                std::make_pair("gamma", &_gamma),
                std::make_pair("Cvel_forw", &_Cvel_forw),
                std::make_pair("Cvel_side", &_Cvel_side),
                std::make_pair("Wreference", &_Wreference),
                std::make_pair("Wvelref", &_Wvelref),
                std::make_pair("Wacc", _Wacc),
                std::make_pair("Wrot", &_Wrot)
                })) {
        std::printf("***IPOPT APP SUCCEFFULLY LOADED***\n");
        // _app->Options()->SetNumericValue("tol", 1e-5);
        // _app->Options()->SetNumericValue("constr_viol_tol", 1e-2);
        
        // _app->Options()->SetNumericValue("mumps_dep_tol", 0); // following ipopt documentation
        // _app->Options()->SetStringValue("mu_strategy", "monotone"); // from casadi document, IMPORTANT! related to coredump
        _app->Options()->SetStringValue("check_derivatives_for_naninf", "yes"); // from casadi document, IMPORTANT! related to coredump
        _app->Options()->SetIntegerValue ("max_iter", 1000);
        _app->Options()->SetIntegerValue ("print_level", 0);
    }
    SmartPtr<IpoptApplication> app(){return _app;}
    SmartPtr<TNLP> mynlp(){return _mynlp;}
};


int dogController(	
        const hyperParameters::x0 x0,
        const hyperParameters::refTraj refTraj,
        const hyperParameters::obstacles obstacles,
        parseOutput::x_plot& x_out, 
        parseOutput::u_plot& p_out
)
{
    static dogCtrlApp a; 
    const auto app = a.app();
    ApplicationReturnStatus status;
    
    std::memcpy(a.x0, x0,   sizeof(hyperParameters::x0));
    std::memcpy(a.refTraj, refTraj,   sizeof(hyperParameters::refTraj));

    status = app->Initialize();
    if( status != Solve_Succeeded )
    {
        std::cout << std::endl << std::endl << "*** Error during initialization!" << std::endl;
        return (int) status;
    }

    // Ask Ipopt to solve the problem
    auto start = std::chrono::system_clock::now();
    status = app->OptimizeTNLP(a.mynlp());
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "time consumption: " << elapsed_seconds.count() <<std::endl;

    if( status == Solve_Succeeded || status == Solved_To_Acceptable_Level)
    {
        // std::cout << std::endl << std::endl << "*** The problem solved!" << std::endl;
    }
    else
    {
        std::cout << std::endl << std::endl << "*** The problem FAILED!" << std::endl;
        switch (status)
        {
            case Infeasible_Problem_Detected:
                std::cout <<"Infeasible_Problem_Detected"<<std::endl; break;
            case Search_Direction_Becomes_Too_Small:
                std::cout <<"Search_Direction_Becomes_Too_Small"<<std::endl; break;
            case Diverging_Iterates:
                std::cout <<"Diverging_Iterates"<<std::endl; break;
            case User_Requested_Stop:
                std::cout <<"User_Requested_Stop"<<std::endl; break;
            case Feasible_Point_Found:
                std::cout <<"Feasible_Point_Found"<<std::endl; break;
            case Maximum_Iterations_Exceeded:
                std::cout <<"Maximum_Iterations_Exceeded"<<std::endl; break;
            case Restoration_Failed:
                std::cout <<"Restoration_Failed"<<std::endl; break;
            default:
                std::cout <<"Other Problem"<< status<<std::endl;
                break;
        }
    }


    x_out = a.x_out;
    p_out = a.p_out;


    return (int) status;
}