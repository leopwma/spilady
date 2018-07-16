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
********************************************************************************/

#if defined GPU

#include "spilady.h"

#if defined changestep

#include "prototype_GPU.h"

__global__ void LP1ScStp_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d
                              #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                              , double *displace_max_ptr_d
                              #endif
                              #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                              , double *phi_max_ptr_d
                              #endif                          
                              );
__global__ void LP1ScStp_part2(struct varGPU *var_ptr_d
                              #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                              , double *displace_max_ptr_d
                              #endif
                              #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                              , double *phi_max_ptr_d
                              #endif                          
                              );

void scale_step_GPU(){

    size_t size = no_of_MP*no_of_threads*sizeof(double);

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    double displace_max;
    double *displace_max_ptr_d;
    cudaMalloc((void**)&displace_max_ptr_d, size);
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    double phi_max;
    double *phi_max_ptr_d;
    cudaMalloc((void**)&phi_max_ptr_d, size);
    #endif

    LP1ScStp_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d
                                         #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                                         , displace_max_ptr_d
                                         #endif
                                         #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                                         , phi_max_ptr_d
                                         #endif                          
                                         );
    LP1ScStp_part2<<<no_of_MP, no_of_threads>>>(var_ptr_d
                                         #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                                         , displace_max_ptr_d
                                         #endif
                                         #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                                         , phi_max_ptr_d
                                         #endif                          
                                         );
                                         
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    cudaMemcpy(&displace_max, displace_max_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    displace_max = sqrt(displace_max)/atmass*step;
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    cudaMemcpy(&phi_max, phi_max_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    phi_max = sqrt(phi_max)/hbar*step;
    #endif

    cout
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    << "displace_max = " << displace_max << "(Angstrom)"
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    << " phi_max = " << phi_max << "(rad.)"
    #endif
    <<'\n';

    int switch_lattice = 0;
    int switch_spin = 0;

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    if (displace_max > displace_limit) switch_lattice = 1;
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    if (phi_max > phi_limit) switch_spin = 1;
    #endif
    if (switch_spin + switch_lattice > 0){
        step *= 0.80;
    } else {
        step *= 1.05;
    }

    #ifdef lattice
    cudaFree(displace_max_ptr_d);
    #endif
    #ifdef spin
    cudaFree(phi_max_ptr_d);
    #endif

}

void scale_step(){
    scale_step_GPU();
}

__global__ void LP1ScStp_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d
                              #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                              , double *displace_max_ptr_d
                              #endif
                              #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                              , double *phi_max_ptr_d
                              #endif
                              ){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    *(displace_max_ptr_d + i) = 0.0;
    #endif
    #ifdef spin
    *(phi_max_ptr_d + i) = 0.0;
    #endif
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
           struct atom_struct *atom_ptr;
           atom_ptr = first_atom_ptr_d + m;
           #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
           double p_sq = vec_sq_d(atom_ptr->p);
           if (p_sq > *(displace_max_ptr_d + i)) *(displace_max_ptr_d + i) = p_sq ;
           #endif
           #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
           double omega_sq = vec_sq_d(atom_ptr->Heff_H);
           if (omega_sq > *(phi_max_ptr_d + i)) *(phi_max_ptr_d + i) = omega_sq;
           #endif
        }
    }

    __syncthreads();
}

__global__ void LP1ScStp_part2(struct varGPU *var_ptr_d
                              #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                              , double *displace_max_ptr_d
                              #endif
                              #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                              , double *phi_max_ptr_d
                              #endif
                              ){

    int depth = blockIdx.x*blockDim.x; 
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j){
            if (*(displace_max_ptr_d + depth + j) > *(displace_max_ptr_d + depth)) *(displace_max_ptr_d + depth) = *(displace_max_ptr_d + depth + j);
        }
    }
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    if (threadIdx.x == 1){
        for (int j = 1; j < blockDim.x; ++j){
            if (*(phi_max_ptr_d + depth + j) > *(phi_max_ptr_d + depth)) *(phi_max_ptr_d + depth) = *(phi_max_ptr_d + depth + j);
        }
    }
    #endif

    __threadfence();

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j){
            if (*(displace_max_ptr_d + j*blockDim.x) > *displace_max_ptr_d) *displace_max_ptr_d = *(displace_max_ptr_d + j*blockDim.x);
        }
    }
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    if (blockIdx.x == 0 && threadIdx.x == 1){
        for (int j = 1; j < gridDim.x; ++j){
            if (*(phi_max_ptr_d + j*blockDim.x) > *phi_max_ptr_d) *phi_max_ptr_d = *(phi_max_ptr_d + j*blockDim.x);
        }
    }
    #endif
}

#endif
#endif
