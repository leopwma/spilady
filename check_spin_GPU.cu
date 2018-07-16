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

#if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined GPU

#include "spilady.h"
#include "prototype_GPU.h"

/****************************************************************************
* GPU prototypes
****************************************************************************/

__global__ void LP1ChSp_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                              double *ave_sx_ptr_d, double *ave_sy_ptr_d, double *ave_sz_ptr_d);
__global__ void LP1ChSp_part2(double *ave_sx_ptr_d, double *ave_sy_ptr_d, double *ave_sz_ptr_d);

/****************************************************************************
* CPU codes
****************************************************************************/

void check_spin_GPU(int current_step){
 
    size_t size = no_of_MP*no_of_threads*sizeof(double);

    ave_s = vec_zero();
    ave_m = vec_zero();

    double *ave_sx_ptr_d;
    double *ave_sy_ptr_d;
    double *ave_sz_ptr_d;

    cudaMalloc((void**)&ave_sx_ptr_d, size);
    cudaMalloc((void**)&ave_sy_ptr_d, size);
    cudaMalloc((void**)&ave_sz_ptr_d, size);

    LP1ChSp_part1<<<no_of_MP,no_of_threads>>>(var_ptr_d, first_atom_ptr_d,
                                              ave_sx_ptr_d, ave_sy_ptr_d, ave_sz_ptr_d);
    LP1ChSp_part2<<<no_of_MP,no_of_threads>>>(ave_sx_ptr_d, ave_sy_ptr_d, ave_sz_ptr_d);

    cudaMemcpy(&ave_s.x, ave_sx_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_s.y, ave_sy_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_s.z, ave_sz_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);

    ave_s = vec_divide(ave_s, natom);
    double ave_s0 = vec_length(ave_s);

    ave_m = vec_times(-el_g, ave_s);
    double ave_m0 = vec_length(ave_m);

    char out_spn_front[] = "spn-";
    char out_spn[256];
    strcpy(out_spn,out_spn_front);
    strcat(out_spn,out_body);
    strcat(out_spn,".dat");

    ofstream out_file(out_spn,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step
             << " " << total_time
             << " " << ave_s.x
             << " " << ave_s.y
             << " " << ave_s.z
             << " " << ave_s0
             << " " << ave_m.x
             << " " << ave_m.y
             << " " << ave_m.z
             << " " << ave_m0
             << '\n';
    out_file.close();

    cudaFree(ave_sx_ptr_d);
    cudaFree(ave_sy_ptr_d);
    cudaFree(ave_sz_ptr_d);

}

void check_spin(int current_step){
    check_spin_GPU(current_step);
}

/****************************************************************************
* GPU codes
****************************************************************************/

__global__ void LP1ChSp_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                       double *ave_sx_ptr_d, double *ave_sy_ptr_d, double *ave_sz_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(ave_sx_ptr_d + i) = 0.0;
    *(ave_sy_ptr_d + i) = 0.0;
    *(ave_sz_ptr_d + i) = 0.0;
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
           struct atom_struct *atom_ptr;
           atom_ptr = first_atom_ptr_d + m;
           *(ave_sx_ptr_d + i) += atom_ptr->s.x;
           *(ave_sy_ptr_d + i) += atom_ptr->s.y;
           *(ave_sz_ptr_d + i) += atom_ptr->s.z;
        }
    }

    __syncthreads();
}

__global__ void LP1ChSp_part2(double *ave_sx_ptr_d, double *ave_sy_ptr_d, double *ave_sz_ptr_d){

    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(ave_sx_ptr_d + depth) += *(ave_sx_ptr_d + depth + j);
    }
    if (threadIdx.x == 1){
        for (int j = 1; j < blockDim.x; ++j) *(ave_sy_ptr_d + depth) += *(ave_sy_ptr_d + depth + j);
    }
    if (threadIdx.x == 2){
        for (int j = 1; j < blockDim.x; ++j) *(ave_sz_ptr_d + depth) += *(ave_sz_ptr_d + depth + j);
    }
    
     __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *ave_sx_ptr_d += *(ave_sx_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 1){
        for (int j = 1; j < gridDim.x; ++j) *ave_sy_ptr_d += *(ave_sy_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 2){
        for (int j = 1; j < gridDim.x; ++j) *ave_sz_ptr_d += *(ave_sz_ptr_d + j*blockDim.x);
    }
}

#endif
