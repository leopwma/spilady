/********************************************************************************
*
*   Copyright (C) 2015 Culham Centre for Fusion Energy,
*   United Kingdom Atomic Energy Authority, Oxfordshire OX14 3DB, UK
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.
*
********************************************************************************
*
*   Program: SPILADY - A Spin-Lattice Dynamics Simulation Program
*   Version: 1.0
*   Date:    Aug 2015
*   Author:  Pui-Wai (Leo) MA
*   Contact: info@spilady.ccfe.ac.uk
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*
********************************************************************************
*
*   Edit notes: 
*   Date:    Oct 2015
*   Author:  Pui-Wai (Leo) Ma
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) Added function double first_derivative(double (*function)(double x), double x);
    2) Added function double interpolation(double x, double input_over_max, double* x_ptr);
*   3) All functions are rewritten using firs_derivative and interpolation for simplicity.
*
********************************************************************************/

#include "spilady.h"

double first_derivative(double (*function)(double x), double x);
double interpolation(double x, double input_over_max, double* x_ptr);

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
double bigf_gen(double rho);
double smallf_gen(double r);
double pair_gen(double r);
#endif

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
double Jij_gen(double r);
#endif

#if defined SDHL || defined SLDHL
double LandauA_gen(double rho);
double LandauB_gen(double rho);
double LandauC_gen(double rho);
double LandauD_gen(double rho);
#endif

void potential_table(){

    rmax = rcut_max;               //Put the max. of r as the max. cutoff of all potentials
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    rhomax = 2e0*smallf_gen(1e0);  //Put the max. of rho as double of the fij of rij = 1 Angstrom
    #endif
    if (rhomax < 1000e0) rhomax = 1000e0;

    finput_over_rmax = finput/rmax;
    finput_over_rhomax = finput/rhomax;

    size_t size_input = ninput*sizeof(double);

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    //Big F  
    bf_ptr   = (double*)malloc(size_input);
    dbf_ptr  = (double*)malloc(size_input);
    
    //small f
    sf_ptr   = (double*)malloc(size_input);
    dsf_ptr  = (double*)malloc(size_input);

    //pair term
    pr_ptr   = (double*)malloc(size_input);
    dpr_ptr  = (double*)malloc(size_input);
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    Jij_ptr  = (double*)malloc(size_input);
    dJij_ptr = (double*)malloc(size_input);
    #endif

    #if defined SDHL || defined SLDHL
    LandauA_ptr = (double*)malloc(size_input);
    LandauB_ptr = (double*)malloc(size_input);
    LandauC_ptr = (double*)malloc(size_input);
    LandauD_ptr = (double*)malloc(size_input);
    #endif
    #if defined SLDHL
    dLandauA_ptr = (double*)malloc(size_input);
    dLandauB_ptr = (double*)malloc(size_input);
    dLandauC_ptr = (double*)malloc(size_input);
    dLandauD_ptr = (double*)malloc(size_input);
    #endif

    for(int n = 0; n < ninput; ++n ){

      double rij  = rmax/double(ninput)*double(n+1);
      double rho  = rhomax/double(ninput)*double(n+1);

      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        *(bf_ptr + n)  = bigf_gen(rho);
        *(dbf_ptr + n) = first_derivative(&bigf_gen,rho);

        *(sf_ptr + n)  = smallf_gen(rij);
        *(dsf_ptr + n) = first_derivative(&smallf_gen,rij);

        *(pr_ptr + n)  = pair_gen(rij);
        *(dpr_ptr + n) = first_derivative(&pair_gen,rij);
      #endif

      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
         *(Jij_ptr + n)  = Jij_gen(rij);
         *(dJij_ptr + n) = first_derivative(&Jij_gen,rij);
         #ifdef magmom
           *(Jij_ptr + n) *= pow(el_g,2);
           *(dJij_ptr + n) *= pow(el_g,2);
         #endif
      #endif
    
      #if defined SDHL || defined SLDHL
         *(LandauA_ptr + n) = LandauA_gen(rho);
         
         *(LandauB_ptr + n) = LandauB_gen(rho);
         *(LandauC_ptr + n) = LandauC_gen(rho);
         *(LandauD_ptr + n) = LandauD_gen(rho);
         #ifdef magmom
           *(LandauA_ptr + n) *= pow(el_g,2);
           *(LandauB_ptr + n) *= pow(el_g,4);
           *(LandauC_ptr + n) *= pow(el_g,6);
           *(LandauD_ptr + n) *= pow(el_g,8);
         #endif
      #endif
      #if defined SLDHL
         *(dLandauA_ptr + n) = first_derivative(&LandauA_gen,rho);
         *(dLandauB_ptr + n) = first_derivative(&LandauB_gen,rho);
         *(dLandauC_ptr + n) = first_derivative(&LandauC_gen,rho);
         *(dLandauD_ptr + n) = first_derivative(&LandauD_gen,rho);
         #ifdef magmom
           *(dLandauA_ptr + n) *= pow(el_g,2);
           *(dLandauB_ptr + n) *= pow(el_g,4);
           *(dLandauC_ptr + n) *= pow(el_g,6);
           *(dLandauD_ptr + n) *= pow(el_g,8);
         #endif
      #endif
    }
}

double first_derivative(double (*function)(double x), double x){

    //accuracy up to numerical exact.
    double dx = x/1000e0;
    double value_p  = (*function)(x + dx);
    double value_m  = (*function)(x - dx);
    double value_2p = (*function)(x + 2*dx);
    double value_2m = (*function)(x - 2*dx);
    double value_3p = (*function)(x + 3*dx);
    double value_3m = (*function)(x - 3*dx);
    return (45e0*(value_p - value_m) - 9e0*(value_2p - value_2m) + (value_3p - value_3m))/(60e0*dx); 
}

double interpolation(double x, double input_over_max, double* x_ptr){

    double fn = x*input_over_max;
    int n = int(fn);
    double ratio = double(n+1)- fn;
    return *(x_ptr+n-1)*ratio + *(x_ptr+n)*(1e0-ratio);

}


#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
double bigf(double rho){

    return interpolation(rho, finput_over_rhomax, bf_ptr);
}

double dbigf(double rho){

    return interpolation(rho, finput_over_rhomax, dbf_ptr);
}

double smallf(double rij){

    return interpolation(rij, finput_over_rmax, sf_ptr);
}

double dsmallf(double rij){

    return interpolation(rij, finput_over_rmax, dsf_ptr);
}

double pairij(double rij){
      
    return interpolation(rij, finput_over_rmax, pr_ptr);
}

double dpair(double rij){
      
    return interpolation(rij, finput_over_rmax, dpr_ptr);
}

#endif

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
double Jij(double rij){
      
    return interpolation(rij, finput_over_rmax, Jij_ptr);
}

double dJij(double rij){
      
    return interpolation(rij, finput_over_rmax, dJij_ptr);
}
#endif

#if defined SDHL || defined SLDHL
double LandauA(double rho){

    return interpolation(rho, finput_over_rhomax, LandauA_ptr);
}

double LandauB(double rho){

    return interpolation(rho, finput_over_rhomax, LandauB_ptr);
}

double LandauC(double rho){

    return interpolation(rho, finput_over_rhomax, LandauC_ptr);
}

double LandauD(double rho){

    return interpolation(rho, finput_over_rhomax, LandauD_ptr);
}
#endif

#if defined SLDHL
double dLandauA(double rho){

    return interpolation(rho, finput_over_rhomax, dLandauA_ptr);
}

double dLandauB(double rho){

    return interpolation(rho, finput_over_rhomax, dLandauB_ptr);
}

double dLandauC(double rho){

    return interpolation(rho, finput_over_rhomax, dLandauC_ptr);
}

double dLandauD(double rho){

    return interpolation(rho, finput_over_rhomax, dLandauD_ptr);
}
#endif

void free_potential_ptr(){

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    free(bf_ptr);
    free(dbf_ptr);
    free(sf_ptr);
    free(dsf_ptr);
    free(pr_ptr);
    free(dpr_ptr);
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    free(Jij_ptr);
    free(dJij_ptr);
    #endif

    #if defined SDHL || defined SLDHL
    free(LandauA_ptr);
    free(LandauB_ptr);
    free(LandauC_ptr);
    free(LandauD_ptr);
    #endif
    
    #if defined SLDHL
    free(dLandauA_ptr);
    free(dLandauB_ptr);
    free(dLandauC_ptr);
    free(dLandauD_ptr);
    #endif
}


