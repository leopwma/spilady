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

#ifdef GPU

#include "spilady.h"
#include "prototype_GPU.h"

/******************************************************************************
* GPU prototype
******************************************************************************/

__global__ void LP1ChEn_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d
                       #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                       , double *ave_pe_ptr_d, double *ave_ke_ptr_d
                       #endif
                       #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                       , double *ave_me_ptr_d, double *ave_me0_ptr_d
                       #endif
                       #ifdef eltemp
                       , double *ave_Ee_ptr_d, struct cell_struct *first_cell_ptr_d
                       #endif
                       );
__global__ void LP1ChEn_part2(struct varGPU *var_ptr_d
                       #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                       , double *ave_pe_ptr_d, double *ave_ke_ptr_d
                       #endif
                       #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                       , double *ave_me_ptr_d, double *ave_me0_ptr_d
                       #endif
                       #ifdef eltemp
                       , double *ave_Ee_ptr_d
                       #endif
                       );
#ifdef eltemp
__global__ void LP2ChEn(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double numerical_error_ave_energy);
#endif
/******************************************************************************
* CPU codes
******************************************************************************/

void check_energy_GPU(int current_step){

    size_t size = no_of_MP*no_of_threads*sizeof(double);

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    double ave_pe;
    double ave_ke;
    double *ave_pe_ptr_d;
    double *ave_ke_ptr_d;
    cudaMalloc((void**)&ave_pe_ptr_d, size);
    cudaMalloc((void**)&ave_ke_ptr_d, size);
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    double ave_me;
    double ave_me0;
    double *ave_me_ptr_d;
    double *ave_me0_ptr_d;
    cudaMalloc((void**)&ave_me_ptr_d, size);
    cudaMalloc((void**)&ave_me0_ptr_d, size);
    #endif

    #ifdef eltemp
    double ave_Ee;
    double *ave_Ee_ptr_d;
    cudaMalloc((void**)&ave_Ee_ptr_d, size);
    #endif


    LP1ChEn_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d
                                         #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                                         , ave_pe_ptr_d, ave_ke_ptr_d
                                         #endif
                                         #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                                         , ave_me_ptr_d, ave_me0_ptr_d
                                         #endif
                                         #ifdef eltemp
                                         , ave_Ee_ptr_d, first_cell_ptr_d
                                         #endif
                                         );
    LP1ChEn_part2<<<no_of_MP, no_of_threads>>>(var_ptr_d
                                         #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                                         , ave_pe_ptr_d, ave_ke_ptr_d
                                         #endif
                                         #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                                         , ave_me_ptr_d, ave_me0_ptr_d
                                         #endif
                                         #ifdef eltemp
                                         , ave_Ee_ptr_d
                                         #endif
                                         );

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    cudaMemcpy(&ave_pe, ave_pe_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_ke, ave_ke_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    ave_pe   /= natom;
    ave_ke   /= natom;
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    cudaMemcpy(&ave_me, ave_me_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_me0, ave_me0_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    ave_me  /= natom;
    ave_me0 /= natom;
     #endif

    #ifdef eltemp
    cudaMemcpy(&ave_Ee, ave_Ee_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    ave_Ee  /= natom;
    
    #ifdef renormalizeEnergy
    double numerical_error_ave_energy = 0e0;
    if (current_step == -1){
        initial_ave_energy = 0e0;
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        initial_ave_energy += ave_pe;
        initial_ave_energy += ave_ke;
        #endif
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        initial_ave_energy += ave_me;
        initial_ave_energy += ave_me0;
        #endif
        initial_ave_energy += ave_Ee;
    } else {
        double current_ave_energy = 0e0;
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        current_ave_energy += ave_pe;
        current_ave_energy += ave_ke;
        #endif
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        current_ave_energy += ave_me;
        current_ave_energy += ave_me0;
        #endif
        current_ave_energy += ave_Ee;
        numerical_error_ave_energy = current_ave_energy - initial_ave_energy;
    }

    LP2ChEn<<<no_of_blocks_cell, no_of_threads>>>(var_ptr_d, first_cell_ptr_d, numerical_error_ave_energy);

    #endif

    #endif



    char out_enr_front[] = "enr-";
    char out_enr[256];
    strcpy(out_enr,out_enr_front);
    strcat(out_enr,out_body);
    strcat(out_enr,".dat");

    ofstream out_file(out_enr,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step             //1
             << " " << total_time        //2
             #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
             << " " << ave_pe            //3
             << " " << ave_ke            //4
             #endif
             #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
             << " " << ave_me            //5
             << " " << ave_me0           //6
             #endif
             #ifdef eltemp
             << " " << ave_Ee            //7
             #endif
             << '\n';

    out_file.close();

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    cudaFree(ave_pe_ptr_d);
    cudaFree(ave_ke_ptr_d);
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    cudaFree(ave_me_ptr_d);
    cudaFree(ave_me0_ptr_d);
    #endif
    #ifdef eltemp
    cudaFree(ave_Ee_ptr_d);
    #endif
    
}

void check_energy(int current_step){
    check_energy_GPU(current_step);
}

/******************************************************************************
* GPU codes
******************************************************************************/

__global__ void LP1ChEn_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d
                       #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                       , double *ave_pe_ptr_d, double *ave_ke_ptr_d
                       #endif
                       #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                       , double *ave_me_ptr_d, double *ave_me0_ptr_d
                       #endif  
                       #ifdef eltemp
                       , double *ave_Ee_ptr_d, struct cell_struct *first_cell_ptr_d
                       #endif
                       ){


    int i = blockIdx.x*blockDim.x + threadIdx.x;
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    *(ave_pe_ptr_d + i) = 0e0;
    *(ave_ke_ptr_d + i) = 0e0;
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    *(ave_me_ptr_d + i) = 0e0;
    *(ave_me0_ptr_d + i) = 0e0;
    #endif
    #ifdef eltemp
    *(ave_Ee_ptr_d + i) = 0e0;
    #endif
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
            struct atom_struct *atom_ptr;
            atom_ptr = first_atom_ptr_d + m;
            #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
            *(ave_pe_ptr_d + i) += atom_ptr->pe;
            *(ave_ke_ptr_d + i) += atom_ptr->ke;
            #endif
            #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
            *(ave_me_ptr_d + i) += atom_ptr->me;
            *(ave_me0_ptr_d + i) += atom_ptr->me0;
            #endif
        }
        #ifdef eltemp
        if (m < var_ptr_d->ncells){
            struct cell_struct *cell_ptr;
            cell_ptr = first_cell_ptr_d + m;
            cell_ptr->Ee = Te_to_Ee_d(cell_ptr->Te);
            *(ave_Ee_ptr_d + i) += (double(var_ptr_d->natom)/double(var_ptr_d->ncells))*cell_ptr->Ee;
        }
        #endif

    }

    __syncthreads();
}

__global__ void LP1ChEn_part2(struct varGPU *var_ptr_d
                       #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                       , double *ave_pe_ptr_d, double *ave_ke_ptr_d
                       #endif
                       #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                       , double *ave_me_ptr_d, double *ave_me0_ptr_d
                       #endif  
                       #ifdef eltemp
                       , double *ave_Ee_ptr_d
                       #endif
                       ){

    int depth = blockIdx.x*blockDim.x;
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(ave_pe_ptr_d + depth) += *(ave_pe_ptr_d + depth + j);
    }
    if (threadIdx.x == 1){
        for (int j = 1; j < blockDim.x; ++j) *(ave_ke_ptr_d + depth) += *(ave_ke_ptr_d + depth + j);
    }
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    if (threadIdx.x == 2){
        for (int j = 1; j < blockDim.x; ++j) *(ave_me_ptr_d + depth) += *(ave_me_ptr_d + depth + j);
    }
    if (threadIdx.x == 3){
        for (int j = 1; j < blockDim.x; ++j) *(ave_me0_ptr_d + depth) += *(ave_me0_ptr_d + depth + j);
    }
    #endif
 
    #ifdef eltemp
    if (threadIdx.x == 4){
        for (int j = 1; j < blockDim.x; ++j) *(ave_Ee_ptr_d + depth) += *(ave_Ee_ptr_d + depth + j);
    }
    #endif

     __threadfence();

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *ave_pe_ptr_d += *(ave_pe_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 1){
        for (int j = 1; j < gridDim.x; ++j) *ave_ke_ptr_d += *(ave_ke_ptr_d + j*blockDim.x);
    }
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    if (blockIdx.x == 0 && threadIdx.x == 2){
        for (int j = 1; j < gridDim.x; ++j) *ave_me_ptr_d += *(ave_me_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 3){
        for (int j = 1; j < gridDim.x; ++j) *ave_me0_ptr_d += *(ave_me0_ptr_d + j*blockDim.x);
    }
    #endif

    #ifdef eltemp
    if (blockIdx.x == 0 && threadIdx.x == 4){
        for (int j = 1; j < gridDim.x; ++j) *ave_Ee_ptr_d += *(ave_Ee_ptr_d + j*blockDim.x);
    }
    #endif
}

#ifdef eltemp
__global__ void LP2ChEn(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double numerical_error_ave_energy){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->ncells){

        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr_d + i;
        cell_ptr->Ee -= numerical_error_ave_energy;
        cell_ptr->Te = Ee_to_Te_d(cell_ptr->Ee);
    }
}
#endif

#endif
