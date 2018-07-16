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

/****************************************************************************
* GPU prototypes
****************************************************************************/

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
__global__ void LP1ChTm_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *sum_ke_ptr_d);
__global__ void LP1ChTm_part2(struct varGPU *var_ptr_d, double *sum_ke_ptr_d);
#endif

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
__global__ void LP2ChTm_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                              struct cell_struct *first_cell_ptr_d
                              #ifndef eltemp
                              , double *Jij_ptr_d
                              #endif
                              , double *sum_R_up_ptr_d, double *sum_R_dn_ptr_d);
__global__ void LP2ChTm_part2(struct varGPU *var_ptr_d, double *sum_R_up_ptr_d, double *sum_R_dn_ptr_d);
#endif

#if defined SDHL || defined SLDHL
__global__ void LP3ChTm_part1(struct varGPU *var_ptr_d
                              , struct cell_struct *first_cell_ptr_d
                              #ifndef eltemp
                              , struct atom_struct *first_atom_ptr_d
                              , double *Jij_ptr_d
                              , double *LandauA_ptr_d
                              , double *LandauB_ptr_d
                              , double *LandauC_ptr_d
                              , double *LandauD_ptr_d
                              #endif
                              , double *sum_L_up_ptr_d, double *sum_L_dn_ptr_d);
__global__ void LP3ChTm_part2(struct varGPU *var_ptr_d, double *sum_L_up_ptr_d, double *sum_L_dn_ptr_d);
#endif

#ifdef eltemp
__global__ void LP4ChTm_part1(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double *Te_ptr_d);
__global__ void LP4ChTm_part2(struct varGPU *var_ptr_d, double *Te_ptr_d);
#endif

/****************************************************************************
* CPU codes
****************************************************************************/

void check_temperature_GPU(int current_step){

    size_t size = no_of_MP*no_of_threads*sizeof(double);

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    double sum_ke = 0e0;
    double *sum_ke_ptr_d;
    cudaMalloc((void**)&sum_ke_ptr_d, size);
    LP1ChTm_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, sum_ke_ptr_d);
    LP1ChTm_part2<<<no_of_MP, no_of_threads>>>(var_ptr_d, sum_ke_ptr_d);
    cudaMemcpy(&sum_ke, sum_ke_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    double Tl = sum_ke*2e0/3e0/natom/boltz;
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    double sum_R_up = 0e0;
    double sum_R_dn = 0e0;
    double *sum_R_up_ptr_d;
    double *sum_R_dn_ptr_d;
    cudaMalloc((void**)&sum_R_up_ptr_d, size);
    cudaMalloc((void**)&sum_R_dn_ptr_d, size);
    LP2ChTm_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, first_cell_ptr_d
                                               #ifndef eltemp
                                               , Jij_ptr_d
                                               #endif
                                               , sum_R_up_ptr_d, sum_R_dn_ptr_d);
    LP2ChTm_part2<<<no_of_MP, no_of_threads>>>(var_ptr_d, sum_R_up_ptr_d, sum_R_dn_ptr_d);
    cudaMemcpy(&sum_R_up, sum_R_up_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_R_dn, sum_R_dn_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    double Ts_R = sum_R_up/sum_R_dn/2e0/boltz;
    #endif

    #if defined SDHL || defined SLDHL
    double sum_L_up = 0e0;
    double sum_L_dn = 0e0;
    double *sum_L_up_ptr_d;
    double *sum_L_dn_ptr_d;
    cudaMalloc((void**)&sum_L_up_ptr_d, size);
    cudaMalloc((void**)&sum_L_dn_ptr_d, size);
    LP3ChTm_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d
                                               , first_cell_ptr_d
                                               #ifndef eltemp
                                               , first_atom_ptr_d
                                               , Jij_ptr_d
                                               , LandauA_ptr_d
                                               , LandauB_ptr_d
                                               , LandauC_ptr_d
                                               , LandauD_ptr_d
                                               #endif
                                               , sum_L_up_ptr_d, sum_L_dn_ptr_d);
    LP3ChTm_part2<<<no_of_MP, no_of_threads>>>(var_ptr_d, sum_L_up_ptr_d, sum_L_dn_ptr_d);
    cudaMemcpy(&sum_L_up, sum_L_up_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&sum_L_dn, sum_L_dn_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    double Ts_L = sum_L_up/sum_L_dn/boltz;
    #endif

    #ifdef eltemp
    double Te = 0e0;
    double *Te_ptr_d;
    cudaMalloc((void**)&Te_ptr_d, size);
    LP4ChTm_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_cell_ptr_d, Te_ptr_d);
    LP4ChTm_part2<<<no_of_MP, no_of_threads>>>(var_ptr_d, Te_ptr_d);
    cudaMemcpy(&Te, Te_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    Te /= ncells;
    Te /= boltz;
    #endif

    char out_tmp_front[] = "tmp-";
    char out_tmp[256];
    strcpy(out_tmp,out_tmp_front);
    strcat(out_tmp,out_body);
    strcat(out_tmp,".dat");

    ofstream out_file(out_tmp,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step
             << " " << total_time
             #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
             << " " << Tl
             #endif
             #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
             << " " << Ts_R
             #endif
             #if defined SDHL || defined SLDHL
             << " " << Ts_L
             #endif
             #ifdef eltemp
             << " " << Te
             #endif
             << '\n';

     out_file.close();

     #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
     cudaFree(sum_ke_ptr_d);
     #endif
     #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
     cudaFree(sum_R_up_ptr_d);
     cudaFree(sum_R_dn_ptr_d);
     #endif
     #if defined SDHL || defined SLDHL
     cudaFree(sum_L_up_ptr_d);
     cudaFree(sum_L_dn_ptr_d);
     #endif
     #ifdef eltemp
     cudaFree(Te_ptr_d);
     #endif
}

void check_temperature(int current_step){
    check_temperature_GPU(current_step);
}

/****************************************************************************
* GPU codes
****************************************************************************/

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
__global__ void LP1ChTm_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *sum_ke_ptr_d)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(sum_ke_ptr_d + i) = 0e0;
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

__global__ void LP1ChTm_part2(struct varGPU *var_ptr_d, double *sum_ke_ptr_d)
{

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

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
__global__ void LP2ChTm_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                              struct cell_struct *first_cell_ptr_d
                              #ifndef eltemp
                              , double *Jij_ptr_d
                              #endif
                              , double *sum_R_up_ptr_d, double *sum_R_dn_ptr_d)
{


    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(sum_R_up_ptr_d + i) = 0e0;
    *(sum_R_dn_ptr_d + i) = 0e0;
    int area = blockDim.x*gridDim.x;
    #ifdef eltemp
      int k = (var_ptr_d->ncells - 1)/area + 1;
      for (int j = 0; j < k; ++j){
          int m = i + j*area;
          if (m < var_ptr_d->ncells){
              *(sum_R_up_ptr_d + i) += (first_cell_ptr_d + m)->sum_R_up;
              *(sum_R_dn_ptr_d + i) += (first_cell_ptr_d + m)->sum_R_dn;
          }
      }
    #else
      int k = (var_ptr_d->natom - 1)/area + 1;
      for (int j = 0; j < k; ++j){
          int m = i + j*area;
          if (m < var_ptr_d->natom) {
             struct atom_struct *atom_ptr;
             atom_ptr = first_atom_ptr_d + m;
             #ifdef extfield
             atom_ptr->Heff_H = atom_ptr->Hext;
             #else
             atom_ptr->Heff_H = vec_zero_d();
             #endif
             inner_spin_d(var_ptr_d, atom_ptr, first_cell_ptr_d, Jij_ptr_d);//calculate the effective field of current atom

             *(sum_R_up_ptr_d + i) += vec_sq_d(vec_cross_d(atom_ptr->s, atom_ptr->Heff_H));
             *(sum_R_dn_ptr_d + i) += vec_dot_d(atom_ptr->s,atom_ptr->Heff_H);
          }
      }
    #endif

    __syncthreads();
}

__global__ void LP2ChTm_part2(struct varGPU *var_ptr_d, double *sum_R_up_ptr_d, double *sum_R_dn_ptr_d)
{

    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(sum_R_up_ptr_d + depth) += *(sum_R_up_ptr_d + depth + j);
    }
    if (threadIdx.x == 1){
        for (int j = 1; j < blockDim.x; ++j) *(sum_R_dn_ptr_d + depth) += *(sum_R_dn_ptr_d + depth + j);
    }

     __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *sum_R_up_ptr_d += *(sum_R_up_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 1){
        for (int j = 1; j < gridDim.x; ++j) *sum_R_dn_ptr_d += *(sum_R_dn_ptr_d + j*blockDim.x);
    }
}
#endif

#if defined SDHL || defined SLDHL
__global__ void LP3ChTm_part1(struct varGPU *var_ptr_d
                              , struct cell_struct *first_cell_ptr_d
                              #ifndef eltemp
                              , struct atom_struct *first_atom_ptr_d
                              , double *Jij_ptr_d
                              , double *LandauA_ptr_d
                              , double *LandauB_ptr_d
                              , double *LandauC_ptr_d
                              , double *LandauD_ptr_d
                              #endif
                              , double *sum_L_up_ptr_d, double *sum_L_dn_ptr_d)
{


    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(sum_L_up_ptr_d + i) = 0e0;
    *(sum_L_dn_ptr_d + i) = 0e0;
    int area = blockDim.x*gridDim.x;
    #ifdef eltemp
      int k = (var_ptr_d->ncells - 1)/area + 1;
      for (int j = 0; j < k; ++j){
          int m = i + j*area;
          if (m < var_ptr_d->ncells){
              *(sum_L_up_ptr_d + i) += (first_cell_ptr_d + m)->sum_L_up;
              *(sum_L_dn_ptr_d + i) += (first_cell_ptr_d + m)->sum_L_dn;
          }
      }
    #else
      int k = (var_ptr_d->natom - 1)/area + 1;
      for (int j = 0; j < k; ++j){
          int m = i + j*area;
          if (m < var_ptr_d->natom) {
             struct atom_struct *atom_ptr;
             atom_ptr = first_atom_ptr_d + m;
             
             #ifdef SLDHL
             double A = LandauA_d(atom_ptr->rho, LandauA_ptr_d, var_ptr_d);
             double B = LandauB_d(atom_ptr->rho, LandauB_ptr_d, var_ptr_d);
             double C = LandauC_d(atom_ptr->rho, LandauC_ptr_d, var_ptr_d);
             double D = LandauD_d(atom_ptr->rho, LandauD_ptr_d, var_ptr_d);
             #endif
             #ifdef SDHL
             double A = LandauA_d(1, LandauA_ptr_d, var_ptr_d);
             double B = LandauB_d(1, LandauB_ptr_d, var_ptr_d);
             double C = LandauC_d(1, LandauC_ptr_d, var_ptr_d);
             double D = LandauD_d(1, LandauD_ptr_d, var_ptr_d);
             #endif

             #ifdef SLDHL
             atom_ptr->sum_Jij_sj = 0e0;
             inner_sum_Jij_sj_d(var_ptr_d, atom_ptr, first_cell_ptr_d, Jij_ptr_d);
             *(sum_L_dn_ptr_d + i) += 2e0*atom_ptr->sum_Jij_sj/vec_length_d(atom_ptr->s);
             #endif

             double s_sq = vec_sq_d(atom_ptr->s);
             #if defined SDHL || defined SLDHL
             atom_ptr->Heff_L = vec_zero_d();
             #endif
             atom_ptr->Heff_L = vec_times_d(-(2e0*A + 4e0*B*s_sq + 6e0*C*pow(s_sq,2) + 8e0*D*pow(s_sq,3)), atom_ptr->s);
             #ifdef SLDHL
             atom_ptr->Heff_HC = vec_times_d(-atom_ptr->sum_Jij_sj/vec_length_d(atom_ptr->s), atom_ptr->s);
             atom_ptr->Heff_L = vec_add_d(atom_ptr->Heff_L, atom_ptr->Heff_HC);
             #endif

             *(sum_L_up_ptr_d + i) += vec_sq_d(vec_add_d(atom_ptr->Heff_H, atom_ptr->Heff_L));
             *(sum_L_dn_ptr_d + i) += 6e0*A + 20e0*B*s_sq + 42e0*C*pow(s_sq,2) + 72e0*D*pow(s_sq,3);

          }
      }
    #endif

    __syncthreads();
}

__global__ void LP3ChTm_part2(struct varGPU *var_ptr_d, double *sum_L_up_ptr_d, double *sum_L_dn_ptr_d)
{

    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(sum_L_up_ptr_d + depth) += *(sum_L_up_ptr_d + depth + j);
    }
    if (threadIdx.x == 1){
        for (int j = 1; j < blockDim.x; ++j) *(sum_L_dn_ptr_d + depth) += *(sum_L_dn_ptr_d + depth + j);
    }

     __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *sum_L_up_ptr_d += *(sum_L_up_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 1){
        for (int j = 1; j < gridDim.x; ++j) *sum_L_dn_ptr_d += *(sum_L_dn_ptr_d + j*blockDim.x);
    }
}
#endif

#ifdef eltemp
__global__ void LP4ChTm_part1(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d, double *Te_ptr_d)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(Te_ptr_d + i) = 0e0;
    int area = blockDim.x*gridDim.x;
      int k = (var_ptr_d->ncells - 1)/area + 1;
      for (int j = 0; j < k; ++j){
          int m = i + j*area;
          if (m < var_ptr_d->ncells){
              *(Te_ptr_d + i) += (first_cell_ptr_d + m)->Te;
          }
      }

    __syncthreads();
}

__global__ void LP4ChTm_part2(struct varGPU *var_ptr_d, double *Te_ptr_d)
{

    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(Te_ptr_d + depth) += *(Te_ptr_d + depth + j);
    }
     __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *Te_ptr_d += *(Te_ptr_d + j*blockDim.x);
    }
}

#endif

#endif


