#include "IpIpoptApplication.hpp"
#include "tonlp.h"
#include "localPlanner.h"

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
class localPlanApp{
public:
    hyperParameters::X0 X0;
    hyperParameters::Xdes Xdes;
    hyperParameters::pa0 pa0;
    hyperParameters::pc pc;
    hyperParameters::Q Q;
    hyperParameters::r r;
    hyperParameters::normAng normAng;
    hyperParameters::boxObstacles boxObstacles;

    Eigen::MatrixXd x_out;
    Eigen::MatrixXd p_out;
    localPlanApp(double Wboxfinal, double WropeNorm, double Wboxstep)
        :_Wboxfinal(Wboxfinal), _WropeNorm(WropeNorm), _Wboxstep(Wboxstep),         _app(IpoptApplicationFactory()), 
         _mynlp(new TONLP(x_out, p_out, {
                std::make_pair("X0", X0),
                std::make_pair("Xdes", Xdes),
                std::make_pair("pa0", pa0),
                std::make_pair("pc", pc),
                std::make_pair("Q", Q),
                std::make_pair("r", r),
                std::make_pair("normAng", normAng),
                std::make_pair("boxObstacles", boxObstacles),
                std::make_pair("Wboxfinal", &_Wboxfinal),
                std::make_pair("WropeNorm", &_WropeNorm),
                std::make_pair("Wboxstep", &_Wboxstep)
                })) {
        std::printf("***IPOPT APP SUCCEFFULLY LOADED***\n");
        // _app->Options()->SetNumericValue("tol", 1e-5);
        // _app->Options()->SetNumericValue("constr_viol_tol", 1e-2);
        
        // _app->Options()->SetNumericValue("mumps_dep_tol", 0); // following ipopt documentation
        // _app->Options()->SetStringValue("mu_strategy", "monotone"); // from casadi document, IMPORTANT! related to coredump
        _app->Options()->SetStringValue("check_derivatives_for_naninf", "yes"); // from casadi document, IMPORTANT! related to coredump
        _app->Options()->SetIntegerValue ("max_iter", 5000);
        _app->Options()->SetIntegerValue ("print_level", 1);
    }
    SmartPtr<IpoptApplication> app(){return _app;}
    SmartPtr<TNLP> mynlp(){return _mynlp;}
private:
    double _Wboxfinal;
	double _WropeNorm;
	double _Wboxstep;
    SmartPtr<IpoptApplication> _app;
    SmartPtr<TNLP> _mynlp;
};


int localPlanner(	
        hyperParameters::X0 X0,
	    hyperParameters::Xdes Xdes,
        hyperParameters::pa0 pa0,
        hyperParameters::pc pc,
        hyperParameters::Q Q,
        hyperParameters::r r,
        hyperParameters::normAng normAng,
        hyperParameters::boxObstacles boxObstacles,
        parseOutput::x_plot& x_out, 
        parseOutput::u_plot& p_out
)
{
    static localPlanApp a(1e3, 1, 1e1); 
    const auto app = a.app();
    ApplicationReturnStatus status;
    
    std::memcpy(a.X0, X0,   sizeof(hyperParameters::X0));
    std::memcpy(a.Xdes, Xdes,   sizeof(hyperParameters::Xdes));
    std::memcpy(a.pa0, pa0,   sizeof(hyperParameters::pa0));
    std::memcpy(a.pc, pc,   sizeof(hyperParameters::pc));
    std::memcpy(a.Q, Q,   sizeof(hyperParameters::Q));
    std::memcpy(a.r, r,   sizeof(hyperParameters::r));
    std::memcpy(a.normAng, normAng,   sizeof(hyperParameters::normAng));
    std::memcpy(a.boxObstacles, boxObstacles,   sizeof(hyperParameters::boxObstacles));

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