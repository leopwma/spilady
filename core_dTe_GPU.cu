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
*   Date:    Apr 2016
*   Author:  Pui-Wai (Leo) MA
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) variable "kappa_e" is added. Previously, it was a constant.
*
********************************************************************************/

#if defined GPU

#include "spilady.h"

#if defined eltemp

#include "prototype_GPU.h"

/***************************************************************************************
* GPU prototypes
***************************************************************************************/
__global__ void LP1dTe(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double dt);

__global__ void LP2dTe(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double dt);

__global__ void LP3dTe_part1(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d,
                              double *total_Ee_ptr_d);
__global__ void LP3dTe_part2(struct varGPU *var_ptr_d, double *total_Ee_ptr_d);

__global__ void LP4dTe(int i, int *allocate_threads_ptr_d, struct cell_struct **allocate_cell_ptr_ptr_d,
                        int *max_no_of_members_ptr_d, struct cell_struct *first_cell_ptr_d,
                        double dx_sq, double dy_sq, double dz_sq, double prefactor);

__global__ void LP5dTe(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double renormalize_factor);

/***************************************************************************************
* CPU codes
***************************************************************************************/
void  core_dTe_GPU_A(double dt);
void  core_dTe_GPU_B(double dt);
void  core_dTe_GPU_C(double dt);

void core_dTe_GPU(double dt){


    int nloop = 20;
    double dt_over_nloop = dt/nloop;
    // C dTe/dt = Kappa Lapacian Te + Ges (Ts-Te) + Gel (Tl - Te)
    for(int i = 0; i < nloop; ++i){
        core_dTe_GPU_A(dt_over_nloop/2e0);   //C dTe/dt = Ges Ts + Gel Tl
        core_dTe_GPU_B(dt_over_nloop/2e0);   //C dTe/dt = -(Ges + Gel) Te
        core_dTe_GPU_C(dt_over_nloop);       //C dTe/dt = Kappa Lapacian Te
        core_dTe_GPU_B(dt_over_nloop/2e0);   //C dTe/dt = -(Ges + Gel) Te
        core_dTe_GPU_A(dt_over_nloop/2e0);   //C dTe/dt = Ges Ts + Gel Tl
    }
}

void core_dTe(double dt){
    core_dTe_GPU(dt);
}

void core_dTe_GPU_A(double dt){
    LP1dTe<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_cell_ptr_d, dt);
}

void core_dTe_GPU_B(double dt){
    LP2dTe<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_cell_ptr_d, dt);
}

void core_dTe_GPU_C(double dt){

    size_t size2 = no_of_MP*no_of_threads*sizeof(double);

    double start_total_Ee = 0e0;
    double *start_total_Ee_ptr_d;
    cudaMalloc((void**)&start_total_Ee_ptr_d, size2);

    LP3dTe_part1<<<no_of_MP,no_of_threads>>>(var_ptr_d, first_cell_ptr_d, start_total_Ee_ptr_d);
    LP3dTe_part2<<<no_of_MP,no_of_threads>>>(var_ptr_d, start_total_Ee_ptr_d);

    cudaMemcpy(&start_total_Ee, start_total_Ee_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(start_total_Ee_ptr_d);

    double thermal_conductivity = kappa_e/eVtoJ/1e10;    //J s^-1 m^-1 K ^-1
    double dx_sq = pow(box_length.x/no_of_link_cell_x,2);
    double dy_sq = pow(box_length.y/no_of_link_cell_y,2);
    double dz_sq = pow(box_length.z/no_of_link_cell_z,2);
    double atomic_volume = box_volume/double(natom);
    double volume_factor = atomic_volume/(double(natom)/double(ncells));
    
    double prefactor = dt/2e0*thermal_conductivity*volume_factor/boltz;

    for (int i = 0 ; i < ngroups ; ++i){
        LP4dTe<<<no_of_blocks_members, no_of_threads>>>(i, allocate_threads_ptr_d, allocate_cell_ptr_ptr_d,
                                                        max_no_of_members_ptr_d, first_cell_ptr_d,
                                                        dx_sq, dy_sq, dz_sq, prefactor);
    }
    for (int i = ngroups - 1 ; i >=0 ; --i){
        LP4dTe<<<no_of_blocks_members, no_of_threads>>>(i, allocate_threads_ptr_d, allocate_cell_ptr_ptr_d,
                                                        max_no_of_members_ptr_d, first_cell_ptr_d,
                                                        dx_sq, dy_sq, dz_sq, prefactor);
    }

    double end_total_Ee = 0e0;
    double *end_total_Ee_ptr_d;
    cudaMalloc((void**)&end_total_Ee_ptr_d, size2);

    LP3dTe_part1<<<no_of_MP,no_of_threads>>>(var_ptr_d, first_cell_ptr_d, end_total_Ee_ptr_d);
    LP3dTe_part2<<<no_of_MP,no_of_threads>>>(var_ptr_d, end_total_Ee_ptr_d);

    cudaMemcpy(&end_total_Ee, end_total_Ee_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(end_total_Ee_ptr_d);

    double renormalize_factor = start_total_Ee/end_total_Ee;

    LP5dTe<<<no_of_blocks_cell, no_of_threads>>>(var_ptr_d, first_cell_ptr_d, renormalize_factor);
}

/***************************************************************************************
* GPU codes
***************************************************************************************/

__global__ void LP1dTe(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double dt){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->ncells){

        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr_d + i;

        cell_ptr->Ee = Te_to_Ee_d(cell_ptr->Te);

        double delta_Ee = 0e0;
        #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined lattlang
        delta_Ee += cell_ptr->Gel*cell_ptr->Tl;
        #endif
        #ifdef spinlang
          #if defined SDH || defined SLDH
            delta_Ee += cell_ptr->Ges*cell_ptr->Ts_R;
          #endif
          #if defined SDHL || defined SLDHL
            delta_Ee += cell_ptr->Ges*cell_ptr->Ts_L;
          #endif
        #endif
        cell_ptr->Ee += dt*delta_Ee;
        cell_ptr->Te = Ee_to_Te_d(cell_ptr->Ee);
    }
}

__global__ void LP2dTe(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double dt){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->ncells){

        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr_d + i;

        double Ges_plus_Gel_dt_over_boltz = 0e0;
        #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined lattlang
        Ges_plus_Gel_dt_over_boltz += cell_ptr->Gel;
        #endif
        #if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined spinlang
        Ges_plus_Gel_dt_over_boltz += cell_ptr->Ges;
        #endif
        Ges_plus_Gel_dt_over_boltz *= dt/boltz;

        //RK2
        //double Te_temp = cell_ptr->Te;
        //double Te_half = Te_temp*exp(-0.5e0*Ges_plus_Gel_dt_over_boltz/Ce_d(Te_temp));
        //cell_ptr->Te = Te_temp*exp(-Ges_plus_Gel_dt_over_boltz/Ce_d(Te_half));

        //RK4
        double T0 = cell_ptr->Te;
        double k1 = T0*(exp(-Ges_plus_Gel_dt_over_boltz/Ce_d(T0)) - 1e0)/dt;
        double T1 = T0 + k1*dt/2e0;
        double k2 = T1*(exp(-Ges_plus_Gel_dt_over_boltz/Ce_d(T1)) - 1e0)/dt;
        double T2 = T0 + k2*dt/2e0;
        double k3 = T2*(exp(-Ges_plus_Gel_dt_over_boltz/Ce_d(T2)) - 1e0)/dt;
        double T3 = T0 + k3*dt;
        double k4 = T3*(exp(-Ges_plus_Gel_dt_over_boltz/Ce_d(T3)) - 1e0)/dt;
        cell_ptr->Te += dt/6e0*(k1 + 2e0*k2 + 2e0*k3 + k4);

     }
}

__global__ void LP3dTe_part1(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d,
                              double *total_Ee_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(total_Ee_ptr_d + i) = 0e0;

    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->ncells - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->ncells) {
            struct cell_struct *cell_ptr;
            cell_ptr = first_cell_ptr_d + m;
            cell_ptr->Ee = Te_to_Ee_d(cell_ptr->Te);
            *(total_Ee_ptr_d + i) += cell_ptr->Ee;
        }
    }
    __syncthreads();
}
__global__ void LP3dTe_part2(struct varGPU *var_ptr_d, double *total_Ee_ptr_d){

    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(total_Ee_ptr_d + depth) += *(total_Ee_ptr_d + depth + j);
    }
     __threadfence();
    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *total_Ee_ptr_d += *(total_Ee_ptr_d + j*blockDim.x);
    }
}

__global__ void LP4dTe(int i, int *allocate_threads_ptr_d, struct cell_struct **allocate_cell_ptr_ptr_d,
                        int *max_no_of_members_ptr_d, struct cell_struct *first_cell_ptr_d,
                        double dx_sq, double dy_sq, double dz_sq, double prefactor){

    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < *(allocate_threads_ptr_d + i)){

        struct cell_struct *cell_ptr;
        cell_ptr = *(allocate_cell_ptr_ptr_d + i*(*max_no_of_members_ptr_d) + j);

        cell_ptr->Ee = Te_to_Ee_d(cell_ptr->Te);;

        cell_ptr->Ee += prefactor
                    *( ((first_cell_ptr_d+(cell_ptr->neigh_cell[0]))->Te
                       +(first_cell_ptr_d+(cell_ptr->neigh_cell[13]))->Te
                       -2e0*cell_ptr->Te)/dx_sq
                      +((first_cell_ptr_d+(cell_ptr->neigh_cell[2]))->Te
                       +(first_cell_ptr_d+(cell_ptr->neigh_cell[14]))->Te
                       -2e0*cell_ptr->Te)/dy_sq
                      +((first_cell_ptr_d+(cell_ptr->neigh_cell[12]))->Te
                       +(first_cell_ptr_d+(cell_ptr->neigh_cell[15]))->Te
                       -2e0*cell_ptr->Te)/dz_sq);

        cell_ptr->Te = Ee_to_Te_d(cell_ptr->Ee);

    }
}

__global__ void LP5dTe(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double renormalize_factor){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->ncells){

        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr_d + i;

        cell_ptr->Ee *= renormalize_factor;
        cell_ptr->Te = Ee_to_Te_d(cell_ptr->Ee);
    }
}

#endif
#endif
