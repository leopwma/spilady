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

/***************************************************************************
* GPU prototypes
****************************************************************************/
    
__global__ void LP1ChSt_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                              double *ave_stress11_ptr_d, double *ave_stress22_ptr_d, double *ave_stress33_ptr_d,
                              double *ave_stress12_ptr_d, double *ave_stress23_ptr_d, double *ave_stress31_ptr_d);
__global__ void LP1ChSt_part2(double *ave_stress11_ptr_d, double *ave_stress22_ptr_d, double *ave_stress33_ptr_d,
                              double *ave_stress12_ptr_d, double *ave_stress23_ptr_d, double *ave_stress31_ptr_d);

/****************************************************************************
* CPU codes
****************************************************************************/

void check_stress_GPU(int current_step){

    size_t size = no_of_MP*no_of_threads*sizeof(double);

    double *ave_stress11_ptr_d;
    double *ave_stress22_ptr_d;
    double *ave_stress33_ptr_d;
    double *ave_stress12_ptr_d;
    double *ave_stress23_ptr_d;
    double *ave_stress31_ptr_d;
    cudaMalloc((void**)&ave_stress11_ptr_d, size);
    cudaMalloc((void**)&ave_stress22_ptr_d, size);
    cudaMalloc((void**)&ave_stress33_ptr_d, size);
    cudaMalloc((void**)&ave_stress12_ptr_d, size);
    cudaMalloc((void**)&ave_stress23_ptr_d, size);
    cudaMalloc((void**)&ave_stress31_ptr_d, size);
    

    LP1ChSt_part1<<<no_of_MP,no_of_threads>>>(var_ptr_d, first_atom_ptr_d,
                                              ave_stress11_ptr_d, ave_stress22_ptr_d, ave_stress33_ptr_d,
                                              ave_stress12_ptr_d, ave_stress23_ptr_d, ave_stress31_ptr_d);
    LP1ChSt_part2<<<no_of_MP,no_of_threads>>>(ave_stress11_ptr_d, ave_stress22_ptr_d, ave_stress33_ptr_d,
                                              ave_stress12_ptr_d, ave_stress23_ptr_d, ave_stress31_ptr_d);
    
    cudaMemcpy(&ave_stress11, ave_stress11_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_stress22, ave_stress22_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_stress33, ave_stress33_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_stress12, ave_stress12_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_stress23, ave_stress23_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ave_stress31, ave_stress31_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);

    ave_stress11 *= 160.217653e0/natom; //1 eV/A^3 = 160.217653 GPa
    ave_stress22 *= 160.217653e0/natom;
    ave_stress33 *= 160.217653e0/natom;
    ave_stress12 *= 160.217653e0/natom;
    ave_stress23 *= 160.217653e0/natom;
    ave_stress31 *= 160.217653e0/natom;

    char out_str_front[] = "str-";
    char out_str[256];
    strcpy(out_str,out_str_front);
    strcat(out_str,out_body);
    strcat(out_str,".dat");

    ofstream out_file(out_str,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step << " " << total_time
             << " " << d.xx << " " << d.yx << " " << d.yy
             << " " << d.zx << " " << d.zy << " " << d.zz
             << " " << density
             << " " << (ave_stress11 + ave_stress22 + ave_stress33)/3e0
             << " " << ave_stress11
             << " " << ave_stress22
             << " " << ave_stress33
             << " " << ave_stress12
             << " " << ave_stress23
             << " " << ave_stress31
             << '\n';

    out_file.close();

    cudaFree(ave_stress11_ptr_d);
    cudaFree(ave_stress22_ptr_d);
    cudaFree(ave_stress33_ptr_d);
    cudaFree(ave_stress12_ptr_d);
    cudaFree(ave_stress23_ptr_d);
    cudaFree(ave_stress31_ptr_d);

}

void check_stress(int current_step){
    check_stress_GPU(current_step);
}

/****************************************************************************
* GPU codes
****************************************************************************/

__global__ void LP1ChSt_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                        double *ave_stress11_ptr_d, double *ave_stress22_ptr_d, double *ave_stress33_ptr_d,
                        double *ave_stress12_ptr_d, double *ave_stress23_ptr_d, double *ave_stress31_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(ave_stress11_ptr_d + i) = 0e0;
    *(ave_stress22_ptr_d + i) = 0e0;
    *(ave_stress33_ptr_d + i) = 0e0;
    *(ave_stress12_ptr_d + i) = 0e0;
    *(ave_stress23_ptr_d + i) = 0e0;
    *(ave_stress31_ptr_d + i) = 0e0;
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
           *(ave_stress11_ptr_d + i) += (first_atom_ptr_d + m)->stress11;
           *(ave_stress22_ptr_d + i) += (first_atom_ptr_d + m)->stress22;
           *(ave_stress33_ptr_d + i) += (first_atom_ptr_d + m)->stress33;
           *(ave_stress12_ptr_d + i) += (first_atom_ptr_d + m)->stress12;
           *(ave_stress23_ptr_d + i) += (first_atom_ptr_d + m)->stress23;
           *(ave_stress31_ptr_d + i) += (first_atom_ptr_d + m)->stress31;
        }
    }

    __syncthreads();
}

__global__ void LP1ChSt_part2(double *ave_stress11_ptr_d, double *ave_stress22_ptr_d, double *ave_stress33_ptr_d,
                              double *ave_stress12_ptr_d, double *ave_stress23_ptr_d, double *ave_stress31_ptr_d){

    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(ave_stress11_ptr_d + depth) += *(ave_stress11_ptr_d + depth + j);
    }
    if (threadIdx.x == 1){
        for (int j = 1; j < blockDim.x; ++j) *(ave_stress22_ptr_d + depth) += *(ave_stress22_ptr_d + depth + j);
    }
    if (threadIdx.x == 2){
        for (int j = 1; j < blockDim.x; ++j) *(ave_stress33_ptr_d + depth) += *(ave_stress33_ptr_d + depth + j);
    }
    if (threadIdx.x == 3){
        for (int j = 1; j < blockDim.x; ++j) *(ave_stress12_ptr_d + depth) += *(ave_stress12_ptr_d + depth + j);
    }
    if (threadIdx.x == 4){
        for (int j = 1; j < blockDim.x; ++j) *(ave_stress23_ptr_d + depth) += *(ave_stress23_ptr_d + depth + j);
    }
    if (threadIdx.x == 5){
        for (int j = 1; j < blockDim.x; ++j) *(ave_stress31_ptr_d + depth) += *(ave_stress31_ptr_d + depth + j);
    }

    __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *ave_stress11_ptr_d += *(ave_stress11_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 1){
        for (int j = 1; j < gridDim.x; ++j) *ave_stress22_ptr_d += *(ave_stress22_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 2){
        for (int j = 1; j < gridDim.x; ++j) *ave_stress33_ptr_d += *(ave_stress33_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 3){
        for (int j = 1; j < gridDim.x; ++j) *ave_stress12_ptr_d += *(ave_stress12_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 4){
        for (int j = 1; j < gridDim.x; ++j) *ave_stress23_ptr_d += *(ave_stress23_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 5){
        for (int j = 1; j < gridDim.x; ++j) *ave_stress31_ptr_d += *(ave_stress31_ptr_d + j*blockDim.x);
    }


}

#endif
