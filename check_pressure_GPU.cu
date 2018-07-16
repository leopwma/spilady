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

#if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) &&  defined GPU

#include "spilady.h"
#include "prototype_GPU.h"

__global__ void LP1ChPr_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *sum_ke_ptr_d);
__global__ void LP1ChPr_part2(double *sum_ke_ptr_d);

void check_pressure_GPU(int current_step){

    double sum_ke;
    double *sum_ke_ptr_d;
    cudaMalloc((void**)&sum_ke_ptr_d, no_of_MP*no_of_threads*sizeof(double));

    LP1ChPr_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, sum_ke_ptr_d);
    LP1ChPr_part2<<<no_of_MP, no_of_threads>>>(sum_ke_ptr_d);

    cudaMemcpy(&sum_ke, sum_ke_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);

    double tmp = 2e0/3e0*sum_ke/natom;
    pressure0 = density*(tmp-virial/(3e0*natom));
    pressure0 *=160.217653e0;
    
    char out_prs_front[] = "prs-";
    char out_prs[256];
    strcpy(out_prs,out_prs_front);
    strcat(out_prs,out_body);
    strcat(out_prs,".dat");

    ofstream out_file(out_prs,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step << " " << total_time
            << " " << d.xx << " " << d.yx << " " << d.yy
            << " " << d.zx << " " << d.zy << " " << d.zz
            << " " << density
            << " " << pressure0
            << '\n';

    out_file.close();
   
    cudaFree(sum_ke_ptr_d);
}

void check_pressure(int current_step){
    check_pressure_GPU(current_step);
}


__global__ void LP1ChPr_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *sum_ke_ptr_d){
 
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(sum_ke_ptr_d + i) = 0.0;
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
           *(sum_ke_ptr_d + i) += (first_atom_ptr_d + m)->ke;
        }
    }

    __syncthreads();
}

__global__ void LP1ChPr_part2(double *sum_ke_ptr_d){

    int depth = blockIdx.x*blockDim.x; 
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(sum_ke_ptr_d + depth) += *(sum_ke_ptr_d + depth + j);
    }

    __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *sum_ke_ptr_d += *(sum_ke_ptr_d + j*blockDim.x);
    }
}


#endif
