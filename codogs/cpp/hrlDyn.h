#ifndef HRLDYN_H
#define HRLDYN_H

int HrlDynm(const double r[3], const double pc[6], const double Q[9], 
        const double xold[3], const double pa[6], double newx[3]);

#endif