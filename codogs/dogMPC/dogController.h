#ifndef LOCALPLANNER_H
#define LOCALPLANNER_H
#include "generated/interface.h"

int dogController(	
        const hyperParameters::x0 x0,
        const hyperParameters::refTraj refTraj,
        const hyperParameters::obstacles obstacles,
        parseOutput::x_plot& x_out, 
        parseOutput::u_plot& p_out);

void configDogController(
        double wreference, 
        double wvelref,
        double wrot,
        hyperParameters::Wacc wacc,
        double Cvelforw,
        double Cvelside,
        double Caccforw,
        double Caccside,
        double Cvel_yaw,
        double Cacc_yaw,
        double Wvel_yaw
        );
#endif