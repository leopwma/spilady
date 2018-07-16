#include "spilady.h"

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL

double Jij_gen(double rij){

    double A = 0.870;
    double rc = 3.75;

    double Jij = 0e0;
    if (rij < rc) Jij = A*pow(1.0-rij/rc,3);

    return Jij;
}
#endif

#if defined SDHL || defined SLDHL
double LandauA_gen(double rho){

    double LandauA = -0.744824;
    return LandauA;

}
double LandauB_gen(double rho){

    double LandauB = 0.345295;
    return LandauB;

}

double LandauC_gen(double rho){

    double LandauC = -0.00790205;
    return LandauC;

}

double LandauD_gen(double rho){

    double LandauD = 0e0;
    return LandauD;

}

#endif
