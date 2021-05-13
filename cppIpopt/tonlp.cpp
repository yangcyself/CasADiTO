// Author:  Chenyu Yang     YSC    2021-04-28

#include "tonlp.h"
// #include "nlpGen.h"
#include "nlpGen/interface.h"
#include <cmath>

#include <cassert>
#include <iostream>
#include "util.h"

/**
 * Setting up several varialbes in the function:
 * params:
 *  FNAME: the name of the casadi generated function
 *  N_ARGS: the first N_ARGS argument will be put into argW
 *  N_RES: the first N_RES results will be put into resW
 * 
 * Deprecated, allocating memory too often jeopodizes the performance
 */
#define AUTO_SET_UP_WORKING_MEM(FNAME, N_ARGS, N_RES, TAG)\
    casadi_int TAG##sz_arg;\
    casadi_int TAG##sz_res;\
    casadi_int TAG##n_arg;\
    casadi_int TAG##n_res;\
    casadi_int TAG##sz_iw;\
    casadi_int TAG##sz_w;\
    casadi_int TAG##flag=true;\
    FNAME##_work(&TAG##sz_arg, &TAG##sz_res, &TAG##sz_iw, &TAG##sz_w);\
    TAG##n_arg = FNAME##_n_in();\
    TAG##n_res = FNAME##_n_out();\
    SimpleArrayPtr<casadi_int> TAG##argSize(TAG##n_arg);\
    SimpleArrayPtr<casadi_int> TAG##resSize(TAG##n_res);\
    for(casadi_int i = 0; i<TAG##n_arg;i++)\
        TAG##argSize[i] = compCCS_nnz(FNAME##_sparsity_in(i));\
    for(casadi_int i = 0; i<TAG##n_res;i++)\
        TAG##resSize[i] = compCCS_nnz(FNAME##_sparsity_out(i));\
    SimpleArrayPtr<const casadi_real*> TAG##arg(TAG##sz_arg);\
    SimpleArrayPtr<casadi_real*> TAG##res(TAG##sz_res);\
    SimpleArrayPtr<casadi_int> TAG##iw(TAG##sz_iw);\
    SimpleArrayPtr<casadi_real> TAG##w(TAG##sz_w);\
    SimpleArrayPtr<casadi_real> TAG##argW(std::accumulate(TAG##argSize.p(), TAG##argSize+N_ARGS, 0));\
    SimpleArrayPtr<casadi_real> TAG##resW(std::accumulate(TAG##resSize.p(), TAG##resSize+N_RES, 0));\
    for(casadi_int i=0, ind=0; i<N_ARGS; ind+= TAG##argSize[i], ++i)\
        TAG##arg[i] = TAG##argW+ind;\
    for(casadi_int i=0, ind=0; i<N_RES; ind+= TAG##resSize[i], ++i)\
        TAG##res[i] = TAG##resW+ind;\

using namespace Ipopt;

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

typedef int ca_work_t(casadi_int *, casadi_int* , casadi_int *, casadi_int *);

void largestWork(ca_work_t f, casadi_int &sz_arg, casadi_int& sz_res, casadi_int &sz_iw, casadi_int &sz_w)
{
    casadi_int _sz_arg;
    casadi_int _sz_res;
    casadi_int _sz_iw;
    casadi_int _sz_w;
    f(&_sz_arg,  &_sz_res, &_sz_iw, &_sz_w);
    sz_arg = std::max(_sz_arg, sz_arg);
    sz_res = std::max(_sz_res, sz_res);
    sz_iw = std::max(_sz_iw, sz_iw);
    sz_w = std::max(_sz_w, sz_w);
}

inline void largestWork(casadi_int &sz_arg, casadi_int& sz_res, casadi_int &sz_iw, casadi_int &sz_w){}

template<typename T1, typename... T>
inline void largestWork(casadi_int &sz_arg, casadi_int& sz_res, casadi_int &sz_iw, casadi_int &sz_w,
                 T1 f, T... fs)
{
    largestWork(f, sz_arg, sz_res, sz_iw, sz_w);
    largestWork(sz_arg, sz_res, sz_iw, sz_w, fs...);
}


// constructor
TONLP::TONLP(Eigen::MatrixXd& x,    
            Eigen::MatrixXd& u,
            Eigen::MatrixXd& t,
            const hp_t& hyParam):
x_out(x),
u_out(u),
t_out(t)
{
    /*Iterate over all functions and build the largest working array*/
    casadi_int sz_arg=0;
    casadi_int sz_res=0;
    casadi_int sz_iw=0;
    casadi_int sz_w=0;
    largestWork(sz_arg, sz_res, sz_iw, sz_w,
        nlp_info_work,
        bounds_info_work,
        starting_point_work,
        nlp_f_work,
        nlp_grad_f_work,
        nlp_g_work,
        nlp_grad_g_work,
        nlp_h_work,
        parse_x_plot_work,
        parse_u_plot_work,
        parse_t_plot_work
    );
    SimpleArrayPtr<const casadi_real*> arg(sz_arg); _arg = arg;
    SimpleArrayPtr<casadi_real*> res(sz_res); _res = res;
    SimpleArrayPtr<casadi_int> iw(sz_iw); _iw = iw;
    SimpleArrayPtr<casadi_real> w(sz_w); _w = w;

    /** Set the first several arguments of arg **/
    for(int i = 0; i<hyParam.size(); i++){
        assert(std::string(hyParam[i].first) == bounds_info_name_in(i));
        _arg[i] = hyParam[i].second;
    }
}


// destructor
TONLP::~TONLP()
{
}

// [TNLP_get_nlp_info]
// returns the size of the problem
bool TONLP::get_nlp_info(
    Index &n,
    Index &m,
    Index &nnz_jac_g,
    Index &nnz_h_lag,
    IndexStyleEnum &index_style)
{
    std::cout <<" get_nlp_info in" <<std::endl;
    AUTO_SET_UP_WORKING_MEM(nlp_info, 0, 4,);

    flag = nlp_info(arg, res, iw, w, 0);
    if (flag)
        return false;

    // The problem described in TONLP.hpp has 4 variables, x[0] through x[3]
    n = std::round(res[0][0]);

    // one equality constraint and one inequality constraint
    m = std::round(res[1][0]);

    // in this example the jacobian is dense and contains 8 nonzeros
    nnz_jac_g = std::round(res[2][0]);

    // the Hessian is also dense and has 16 total nonzeros, but we
    // only need the lower left corner (since it is symmetric)
    nnz_h_lag = std::round(res[3][0]);

    // use the C style indexing (0-based)
    index_style = TNLP::C_STYLE;
    std::cout<<"n: \t"<< n <<std::endl;
    std::cout<<"m: \t"<< m <<std::endl;
    std::cout<<"nnz_jac_g: \t"<< nnz_jac_g <<std::endl;
    std::cout<<"nnz_h_lag: \t"<< nnz_h_lag <<std::endl;
    std::cout <<" get_nlp_info out" <<std::endl;

    return true;
}
// [TNLP_get_nlp_info]

// [TNLP_get_bounds_info]
// returns the variable bounds
bool TONLP::get_bounds_info(
    Index n,
    Number *x_l,
    Number *x_u,
    Index m,
    Number *g_l,
    Number *g_u)
{
    // here, the n and m we gave IPOPT in get_nlp_info are passed back to us.
    // If desired, we could assert to make sure they are what we think they are.
    // assert(n == 4);
    // assert(m == 2);
    // std::cout <<" get_bounds_info in" <<std::endl;

    // AUTO_SET_UP_WORKING_MEM(bounds_info, n_arg, 0,);
    casadi_int flag = true;
    _res[0] = x_l;
    _res[1] = x_u;
    _res[2] = g_l;
    _res[3] = g_u;
    flag = bounds_info(_arg,_res,_iw,_w,0);
    // std::cout <<" get_bounds_info out" <<std::endl;
    return !flag;
}
// [TNLP_get_bounds_info]

// [TNLP_get_starting_point]
// returns the initial point for the problem
bool TONLP::get_starting_point(
    Index n,
    bool init_x,
    Number *x,
    bool init_z,
    Number *z_L,
    Number *z_U,
    Index m,
    bool init_lambda,
    Number *lambda)
{
    // std::cout <<" get_starting_point in" <<std::endl;

    assert(init_x == true);
    assert(init_z == false);
    assert(init_lambda == false);

    // AUTO_SET_UP_WORKING_MEM(starting_point, n_arg, 0,);
    casadi_int flag = true;

    _res[0] = x;

    flag = starting_point(_arg,_res,_iw,_w,0);
    // std::cout <<" get_starting_point out" <<std::endl;
    return !flag;

}
// [TNLP_get_starting_point]

// [TNLP_eval_f]
// returns the value of the objective function
bool TONLP::eval_f(
    Index n,
    const Number *x,
    bool new_x,
    Number &obj_value)
{
    // std::cout <<" eval_f in" <<std::endl;

    // AUTO_SET_UP_WORKING_MEM(nlp_f, n_arg-1, 1,);
    static casadi_int n_arg = nlp_f_n_in();
    casadi_int flag = true;
    _arg[n_arg-1] = x;
    
    flag = nlp_f(_arg,_res,_iw,_w,0);
    obj_value = _res[0][0];
    // std::cout <<" eval_f out" <<std::endl;
    return !flag;
}
// [TNLP_eval_f]

// [TNLP_eval_grad_f]
// return the gradient of the objective function grad_{x} f(x)
bool TONLP::eval_grad_f(
    Index n,
    const Number *x,
    bool new_x,
    Number *grad_f)
{
    // std::cout <<" eval_grad_f in" <<std::endl;

    // AUTO_SET_UP_WORKING_MEM(nlp_grad_f, n_arg-1, 0,);
    static casadi_int n_arg = nlp_grad_f_n_in();
    casadi_int flag = true;

    _arg[n_arg-1] = x;
    _res[0] = grad_f;

    flag = nlp_grad_f(_arg,_res,_iw,_w,0);
    // std::cout <<" eval_grad_f out" <<std::endl;
    return !flag;
}
// [TNLP_eval_grad_f]

// [TNLP_eval_g]
// return the value of the constraints: g(x)
bool TONLP::eval_g(
    Index n,
    const Number *x,
    bool new_x,
    Index m,
    Number *g)
{
    // std::cout <<" eval_g in" <<std::endl;

    // AUTO_SET_UP_WORKING_MEM(nlp_g, n_arg-1, 0,);
    static casadi_int n_arg = nlp_g_n_in();
    casadi_int flag = true;

    _arg[n_arg-1] = x;
    _res[0] = g;

    flag = nlp_g(_arg,_res,_iw,_w,0);
    // std::cout <<" eval_g out" <<std::endl;
    return !flag;

}
// [TNLP_eval_g]

// [TNLP_eval_jac_g]
// return the structure or values of the Jacobian
bool TONLP::eval_jac_g(
    Index n,
    const Number *x,
    bool new_x,
    Index m,
    Index nele_jac,
    Index *iRow,
    Index *jCol,
    Number *values)
{
    // std::cout <<" eval_jac_g in" <<std::endl;

    if (values == NULL){
        compCCS_Triplet(nlp_grad_g_sparsity_out(0), iRow, jCol);
    }else{ // (values == NULL)
        // AUTO_SET_UP_WORKING_MEM(nlp_grad_g, n_arg-1, 0,);
        static casadi_int n_arg = nlp_grad_g_n_in();
        casadi_int flag = true;

        _arg[n_arg-1] = x;
        _res[0] = values;

        flag = nlp_grad_g(_arg,_res,_iw,_w,0);
        return !flag;
    } //(values == NULL)
    // std::cout <<" eval_jac_g out" <<std::endl;
    return true;
}
// [TNLP_eval_jac_g]

// [TNLP_eval_h]
//return the structure or values of the Hessian
bool TONLP::eval_h(
    Index n,
    const Number *x,
    bool new_x,
    Number obj_factor,
    Index m,
    const Number *lambda,
    bool new_lambda,
    Index nele_hess,
    Index *iRow,
    Index *jCol,
    Number *values)
{
    // std::cout <<" eval_h in" <<std::endl;

    if (values == NULL){
        compCCS_Triplet(nlp_h_sparsity_out(0), iRow, jCol);
    }else{ //(values == NULL)
        // AUTO_SET_UP_WORKING_MEM(nlp_h, n_arg-3, 0,);
        static casadi_int n_arg = nlp_h_n_in();
        casadi_int flag = true;

        _arg[n_arg-3] = x;
        _arg[n_arg-2] = &obj_factor;
        _arg[n_arg-1] = lambda;
        _res[0] = values;

        flag = nlp_h(_arg,_res,_iw,_w,0);
        return !flag;
    }
    // std::cout <<" eval_h out" <<std::endl;
    return true;
}
// [TNLP_eval_h]

// [TNLP_finalize_solution]
void TONLP::finalize_solution(
    SolverReturn status,
    Index n,
    const Number *x,
    const Number *z_L,
    const Number *z_U,
    Index m,
    const Number *g,
    const Number *lambda,
    Number obj_value,
    const IpoptData *ip_data,
    IpoptCalculatedQuantities *ip_cq)
{

    static casadi_int n_arg = parse_x_plot_n_in();
    _arg[n_arg-1] = x;

    AUTO_SET_UP_WORKING_MEM(parse_x_plot, 0, x_n_res, x_);
    parse_x_plot(_arg, x_res, x_iw, x_w, 0);
    x_out = Eigen::MatrixXd::Zero(parse_x_plot_sparsity_out(0)[0],
                    parse_x_plot_sparsity_out(0)[1]);
    compCCS_fillDense(parse_x_plot_sparsity_out(0),
                        x_res[0], x_out);


    AUTO_SET_UP_WORKING_MEM(parse_u_plot, 0, u_n_res, u_);
    parse_u_plot(_arg, u_res, u_iw, u_w, 0);
    u_out = Eigen::MatrixXd::Zero(parse_u_plot_sparsity_out(0)[0],
                    parse_u_plot_sparsity_out(0)[1]);
    compCCS_fillDense(parse_u_plot_sparsity_out(0),
                        u_res[0], u_out);

    AUTO_SET_UP_WORKING_MEM(parse_t_plot, 0, t_n_res, t_);
    parse_t_plot(_arg, t_res, t_iw, t_w, 0);
    t_out = Eigen::MatrixXd::Zero(parse_t_plot_sparsity_out(0)[0],
                    parse_t_plot_sparsity_out(0)[1]);
    compCCS_fillDense(parse_t_plot_sparsity_out(0),
                        t_res[0], t_out);


    std::cout << std::endl
              << std::endl
              << "Objective value" << std::endl;
    std::cout << "f(x*) = " << obj_value << std::endl;

}
// [TNLP_finalize_solution]
