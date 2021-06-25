#ifndef LOCALPLANNER_H
#define LOCALPLANNER_H

int localPlanner(const double r[3], const double pc[6], const double Q[9], 
            const double X0[3], const double Xdes[3], const double pa0[6], 
            const double normAng[3], const double cylinderObstacles[6], 
            double x_out[90], double u_out[180]);

#endif