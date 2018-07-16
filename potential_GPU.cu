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
    1) Added function __device__ double interpolation_d(double x, double input_over_max, double* x_ptr);
*   2) All functions are rewritten using interpolation_d for simplicity.
*
********************************************************************************/

#ifdef GPU

#include "spilady.h"
#include "prototype_GPU.h"

void potential_table_GPU(){
 
    size_t size_input = ninput*sizeof(double);

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    //Big F  
    cudaMalloc((void**)&bf_ptr_d, size_input);
    cudaMalloc((void**)&dbf_ptr_d, size_input);
    cudaMemcpy(bf_ptr_d, bf_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(dbf_ptr_d, dbf_ptr, size_input, cudaMemcpyHostToDevice);

    //small f
    cudaMalloc((void**)&sf_ptr_d, size_input);
    cudaMalloc((void**)&dsf_ptr_d, size_input);
    cudaMemcpy(sf_ptr_d, sf_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(dsf_ptr_d, dsf_ptr, size_input, cudaMemcpyHostToDevice);

    //pair term
    cudaMalloc((void**)&pr_ptr_d, size_input);
    cudaMalloc((void**)&dpr_ptr_d, size_input);
    cudaMemcpy(pr_ptr_d, pr_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(dpr_ptr_d, dpr_ptr, size_input, cudaMemcpyHostToDevice);
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    //Jij
    cudaMalloc((void**)&Jij_ptr_d, size_input);
    cudaMalloc((void**)&dJij_ptr_d, size_input);
    cudaMemcpy(Jij_ptr_d, Jij_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(dJij_ptr_d, dJij_ptr, size_input, cudaMemcpyHostToDevice);
    #endif

    #if defined SDHL || defined SLDHL
    cudaMalloc((void**)&LandauA_ptr_d, size_input);
    cudaMalloc((void**)&LandauB_ptr_d, size_input);
    cudaMalloc((void**)&LandauC_ptr_d, size_input);
    cudaMalloc((void**)&LandauD_ptr_d, size_input);
    cudaMemcpy(LandauA_ptr_d, LandauA_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(LandauB_ptr_d, LandauB_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(LandauC_ptr_d, LandauC_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(LandauD_ptr_d, LandauD_ptr, size_input, cudaMemcpyHostToDevice);
    #endif

    #ifdef SLDHL
    cudaMalloc((void**)&dLandauA_ptr_d, size_input);
    cudaMalloc((void**)&dLandauB_ptr_d, size_input);
    cudaMalloc((void**)&dLandauC_ptr_d, size_input);
    cudaMalloc((void**)&dLandauD_ptr_d, size_input);
    cudaMemcpy(dLandauA_ptr_d, dLandauA_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(dLandauB_ptr_d, dLandauB_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(dLandauC_ptr_d, dLandauC_ptr, size_input, cudaMemcpyHostToDevice);
    cudaMemcpy(dLandauD_ptr_d, dLandauD_ptr, size_input, cudaMemcpyHostToDevice);
    #endif
}

__device__ double interpolation_d(double x, double input_over_max, double* x_ptr){

    double fn = x*input_over_max;
    int n = int(fn);
    double ratio = double(n+1)- fn;
    return *(x_ptr+n-1)*ratio + *(x_ptr+n)*(1e0-ratio);

}

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
__device__ double bigf_d(double rho, double *bf_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, bf_ptr_d);
}

__device__ double dbigf_d(double rho, double *dbf_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, dbf_ptr_d);
}

__device__ double smallf_d(double rij, double *sf_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rij, var_ptr_d->finput_over_rmax, sf_ptr_d);
}

__device__ double dsmallf_d(double rij, double *dsf_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rij, var_ptr_d->finput_over_rmax, dsf_ptr_d);
}

__device__ double pair_d(double rij, double *pr_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rij, var_ptr_d->finput_over_rmax, pr_ptr_d);
}

__device__ double dpair_d(double rij, double *dpr_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rij, var_ptr_d->finput_over_rmax, dpr_ptr_d);
}
#endif

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
__device__ double Jij_d(double rij, double *Jij_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rij, var_ptr_d->finput_over_rmax, Jij_ptr_d);
}

__device__ double dJij_d(double rij, double *dJij_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rij, var_ptr_d->finput_over_rmax, dJij_ptr_d);
}
#endif

#if defined SDHL || defined SLDHL
__device__ double LandauA_d(double rho, double *LandauA_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, LandauA_ptr_d);
}

__device__ double LandauB_d(double rho, double *LandauB_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, LandauB_ptr_d);
}

__device__ double LandauC_d(double rho, double *LandauC_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, LandauC_ptr_d);
}

__device__ double LandauD_d(double rho, double *LandauD_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, LandauD_ptr_d);
}
#endif

#ifdef SLDHL

__device__ double dLandauA_d(double rho, double *dLandauA_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, dLandauA_ptr_d);
}

__device__ double dLandauB_d(double rho, double *dLandauB_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, dLandauB_ptr_d);
}

__device__ double dLandauC_d(double rho, double *dLandauC_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, dLandauC_ptr_d);
}

__device__ double dLandauD_d(double rho, double *dLandauD_ptr_d, struct varGPU *var_ptr_d){

    return interpolation_d(rho, var_ptr_d->finput_over_rhomax, dLandauD_ptr_d);
}

#endif

void free_potential_ptr_GPU(){

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    cudaFree(bf_ptr_d);
    cudaFree(dbf_ptr_d);
    cudaFree(sf_ptr_d);
    cudaFree(dsf_ptr_d);
    cudaFree(pr_ptr_d);
    cudaFree(dpr_ptr_d);
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    cudaFree(Jij_ptr_d);
    cudaFree(dJij_ptr_d);
    #endif

    #if defined SDHL || defined SLDHL
    cudaFree(LandauA_ptr_d);
    cudaFree(LandauB_ptr_d);
    cudaFree(LandauC_ptr_d);
    cudaFree(LandauD_ptr_d);
    #endif
    #if defined SLDHL
    cudaFree(dLandauA_ptr_d);
    cudaFree(dLandauB_ptr_d);
    cudaFree(dLandauC_ptr_d);
    cudaFree(dLandauD_ptr_d);
    #endif
}

#endif
