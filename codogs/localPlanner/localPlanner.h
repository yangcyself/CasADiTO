#ifndef LOCALPLANNER_H
#define LOCALPLANNER_H
#include "generated/interface.h"

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
        parseOutput::u_plot& p_out);

#endif