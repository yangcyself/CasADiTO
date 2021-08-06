#ifndef LOCALPLANNER_H
#define LOCALPLANNER_H
#include "generated/interface.h"

int dogController(	
        hyperParameters::x0 x0,
        hyperParameters::refTraj refTraj,
        parseOutput::x_plot& x_out, 
        parseOutput::u_plot& p_out);

#endif