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

/**********************************************************************
* GPU prototype
***********************************************************************/

__global__ void LP1ForEn(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d);

__global__ void LP2ForEn(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, struct cell_struct *first_cell_ptr_d
                      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                      , double *bf_ptr_d, double *sf_ptr_d, double *pr_ptr_d
                      , double *dbf_ptr_d, double *dsf_ptr_d, double *dpr_ptr_d
                      #endif
                      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                      ,double *Jij_ptr_d, double *dJij_ptr_d
                      #endif
                      #if defined SDHL || defined SLDHL
                      ,double *LandauA_ptr_d, double *LandauB_ptr_d, double *LandauC_ptr_d, double *LandauD_ptr_d
                      #endif
                      #if defined SLDHL
                      ,double *dLandauA_ptr_d, double *dLandauB_ptr_d, double *dLandauC_ptr_d, double *dLandauD_ptr_d
                      #endif
                      );

__global__ void LP3ForEn(struct varGPU *var_ptr_d, struct atom_struct* first_atom_ptr_d);

#ifdef localvol
__global__ void LP4ForEn_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *sum_volume_ptr_d);
__global__ void LP4ForEn_part2(double *sum_volume_ptr_d);

__global__ void LP5ForEn(struct varGPU *var_ptr_d, struct atom_struct* first_atom_ptr_d, double volume_factor);
#endif

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
__global__ void LP6ForEn(struct varGPU *var_ptr_d, struct atom_struct* first_atom_ptr_d);

__global__ void LP7ForEn_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *virial_ptr_d);
__global__ void LP7ForEn_part2(double *virial_ptr_d);
#endif

__device__ void inner_loop_d(struct varGPU *var_ptr_d, struct atom_struct *atom_ptr, struct cell_struct *first_cell_ptr_d
                      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                      , double *bf_ptr_d, double *sf_ptr_d, double *pr_ptr_d
                      , double *dbf_ptr_d, double *dsf_ptr_d, double *dpr_ptr_d
                      #endif
                      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                      ,double *Jij_ptr_d, double *dJij_ptr_d
                      #endif
                      #if defined SDHL || defined SLDHL
                      ,double *LandauA_ptr_d, double *LandauB_ptr_d, double *LandauC_ptr_d, double *LandauD_ptr_d
                      #endif
                      #if defined SLDHL
                      ,double *dLandauA_ptr_d, double *dLandauB_ptr_d, double *dLandauC_ptr_d, double *dLandauD_ptr_d
                      #endif
                      );

/**********************************************************************
* CPU codes
***********************************************************************/
void calculate_force_energy_GPU(){

    LP1ForEn<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d);

    LP2ForEn<<<no_of_blocks, no_of_threads>>>(var_ptr_d
        , first_atom_ptr_d, first_cell_ptr_d
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        , bf_ptr_d, sf_ptr_d, pr_ptr_d
        , dbf_ptr_d, dsf_ptr_d, dpr_ptr_d
        #endif
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        , Jij_ptr_d, dJij_ptr_d
        #endif
        #if defined SDHL || defined SLDHL
        , LandauA_ptr_d, LandauB_ptr_d, LandauC_ptr_d, LandauD_ptr_d
        #endif
        #if defined SLDHL
        , dLandauA_ptr_d, dLandauB_ptr_d, dLandauC_ptr_d, dLandauD_ptr_d
        #endif
        );

    LP3ForEn<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d);

    #ifdef localvol
      double sum_volume= 0e0;
      double* sum_volume_ptr_d;
      cudaMalloc((void**)&sum_volume_ptr_d, no_of_MP*no_of_threads*sizeof(double));
      LP4ForEn_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, sum_volume_ptr_d);
      LP4ForEn_part2<<<no_of_MP, no_of_threads>>>(sum_volume_ptr_d);
      cudaMemcpy(&sum_volume, sum_volume_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(sum_volume_ptr_d);
      
      cudaMemcpy(&box_volume, &(var_ptr_d->box_volume), sizeof(double), cudaMemcpyDeviceToHost);
      double volume_factor = box_volume/sum_volume;
      LP5ForEn<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, volume_factor);
    #endif

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
      LP6ForEn<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d);

      double *virial_ptr_d;
      cudaMalloc((void**)&virial_ptr_d, no_of_MP*no_of_threads*sizeof(double));

      LP7ForEn_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, virial_ptr_d);
      LP7ForEn_part2<<<no_of_MP, no_of_threads>>>(virial_ptr_d);

      cudaMemcpy(&virial, virial_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
      cudaFree(virial_ptr_d);
    #endif
}

void calculate_force_energy(){
    calculate_force_energy_GPU();

}

/**************************************************************************
* GPU codes
**************************************************************************/

__global__ void LP1ForEn(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        atom_ptr->f   = vec_zero_d();
        atom_ptr->pe  = 0e0;
        atom_ptr->vir = 0e0;
        atom_ptr->stress11 = 0e0;
        atom_ptr->stress22 = 0e0;
        atom_ptr->stress33 = 0e0;
        atom_ptr->stress12 = 0e0;
        atom_ptr->stress23 = 0e0;
        atom_ptr->stress31 = 0e0;
        #endif

        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        atom_ptr->me  = 0e0;
        atom_ptr->me0 = 0e0;
        #endif

        #ifdef localvol
        atom_ptr->sum_rij_m1 = 0e0; //Sum rij^-1
        atom_ptr->sum_rij_m2 = 0e0; //Sum rij^-2
        #endif
    }
}

__global__ void LP2ForEn(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, struct cell_struct *first_cell_ptr_d
                      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                      , double *bf_ptr_d, double *sf_ptr_d, double *pr_ptr_d
                      , double *dbf_ptr_d, double *dsf_ptr_d, double *dpr_ptr_d
                      #endif
                      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                      ,double *Jij_ptr_d, double *dJij_ptr_d
                      #endif
                      #if defined SDHL || defined SLDHL
                      ,double *LandauA_ptr_d, double *LandauB_ptr_d, double *LandauC_ptr_d, double *LandauD_ptr_d
                      #endif
                      #if defined SLDHL
                      ,double *dLandauA_ptr_d, double *dLandauB_ptr_d, double *dLandauC_ptr_d, double *dLandauD_ptr_d
                      #endif
                      )
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom)
        inner_loop_d(var_ptr_d, first_atom_ptr_d + i, first_cell_ptr_d
                     #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                     , bf_ptr_d, sf_ptr_d, pr_ptr_d
                     , dbf_ptr_d, dsf_ptr_d, dpr_ptr_d
                     #endif
                     #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                     , Jij_ptr_d, dJij_ptr_d
                     #endif
                     #if defined SDHL || defined SLDHL
                     , LandauA_ptr_d, LandauB_ptr_d, LandauC_ptr_d, LandauD_ptr_d
                     #endif
                     #if defined SLDHL
                     , dLandauA_ptr_d, dLandauB_ptr_d, dLandauC_ptr_d, dLandauD_ptr_d
                     #endif
                     );
}

__global__ void LP3ForEn(struct varGPU *var_ptr_d, struct atom_struct* first_atom_ptr_d)
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        #ifdef localvol
          double local_radius = 0.5e0*atom_ptr->sum_rij_m1/atom_ptr->sum_rij_m2;
          atom_ptr->local_volume = 4e0*Pi_num/3e0*pow(local_radius, 3e0); //it is only an estimation!!!
        #else
          atom_ptr->local_volume = var_ptr_d->box_volume/var_ptr_d->natom;
        #endif
    }

}

#ifdef localvol
__global__ void LP4ForEn_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *sum_volume_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(sum_volume_ptr_d + i) = 0.0;
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
           *(sum_volume_ptr_d + i) += (first_atom_ptr_d + m)->local_volume;
        }
    }
    __syncthreads();
}

__global__ void LP4ForEn_part2(double *sum_volume_ptr_d){

    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(sum_volume_ptr_d + depth) += *(sum_volume_ptr_d + depth + j);
    }

    __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *sum_volume_ptr_d += *(sum_volume_ptr_d + j*blockDim.x);
    }
}

__global__ void LP5ForEn(struct varGPU *var_ptr_d, struct atom_struct* first_atom_ptr_d, double volume_factor){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom) (first_atom_ptr_d + i)->local_volume *= volume_factor;
}
#endif

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
__global__ void LP6ForEn(struct varGPU *var_ptr_d, struct atom_struct* first_atom_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        atom_ptr->stress11 = (pow(atom_ptr->p.x,2)/var_ptr_d->atmass + atom_ptr->stress11/2e0)/atom_ptr->local_volume;
        atom_ptr->stress22 = (pow(atom_ptr->p.y,2)/var_ptr_d->atmass + atom_ptr->stress22/2e0)/atom_ptr->local_volume;
        atom_ptr->stress33 = (pow(atom_ptr->p.z,2)/var_ptr_d->atmass + atom_ptr->stress33/2e0)/atom_ptr->local_volume;
        atom_ptr->stress12 = ((atom_ptr->p.x*atom_ptr->p.y)/var_ptr_d->atmass + atom_ptr->stress12/2e0)/atom_ptr->local_volume;
        atom_ptr->stress23 = ((atom_ptr->p.y*atom_ptr->p.z)/var_ptr_d->atmass + atom_ptr->stress23/2e0)/atom_ptr->local_volume;
        atom_ptr->stress31 = ((atom_ptr->p.z*atom_ptr->p.x)/var_ptr_d->atmass + atom_ptr->stress31/2e0)/atom_ptr->local_volume;
    }

}

__global__ void LP7ForEn_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *virial_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(virial_ptr_d + i) = 0.0;
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
           *(virial_ptr_d + i) += (first_atom_ptr_d + m)->vir;
        }
    }

    __syncthreads();
}

__global__ void LP7ForEn_part2(double *virial_ptr_d){

    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(virial_ptr_d + depth) += *(virial_ptr_d + depth + j);
    }

    __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *virial_ptr_d += *(virial_ptr_d + j*blockDim.x);
    }
}
#endif



__device__ void inner_loop_d(struct varGPU *var_ptr_d, struct atom_struct *atom_ptr, struct cell_struct *first_cell_ptr_d
                      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                      , double *bf_ptr_d, double *sf_ptr_d, double *pr_ptr_d
                      , double *dbf_ptr_d, double *dsf_ptr_d, double *dpr_ptr_d
                      #endif
                      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                      ,double *Jij_ptr_d, double *dJij_ptr_d
                      #endif
                      #if defined SDHL || defined SLDHL
                      ,double *LandauA_ptr_d, double *LandauB_ptr_d, double *LandauC_ptr_d, double *LandauD_ptr_d
                      #endif
                      #if defined SLDHL
                      ,double *dLandauA_ptr_d, double *dLandauB_ptr_d, double *dLandauC_ptr_d, double *dLandauD_ptr_d
                      #endif
                      )
{
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
      double si_sq = vec_sq_d(atom_ptr->s);
    #endif

    struct atom_struct *work_ptr;

    struct cell_struct *ccell_ptr;
    struct cell_struct *wcell_ptr;

    ccell_ptr = first_cell_ptr_d + atom_ptr->new_cell_index;

    for (int i = 0; i <= 26; ++i){
        if (i == 26)
            wcell_ptr = ccell_ptr;
        else
            wcell_ptr = first_cell_ptr_d + (ccell_ptr->neigh_cell[i]);

        work_ptr = wcell_ptr->head_ptr;
        while (work_ptr != NULL){

            vector rij = vec_sub_d(atom_ptr->r, work_ptr->r);

            //find image of j closest to i
            find_image_d(rij, var_ptr_d);

            double rsq  = vec_sq_d(rij);

            if (rsq < var_ptr_d->rcut_max_sq && atom_ptr != work_ptr){

                double rij0 = sqrt(rsq);

                #ifdef localvol
                if (rij0 < var_ptr_d->rcut_vol){
                    atom_ptr->sum_rij_m1 += 1e0/rij0;
                    atom_ptr->sum_rij_m2 += 1e0/rsq;
                }
                #endif

                #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                double pair_enr = 0e0;
                double dudr = 0e0;
                #endif

                #if defined SLDH || defined SLDHL
                double dudr_spin = 0e0;
                #endif

                #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                  double sj_sq = vec_sq_d(work_ptr->s);
                #endif

                #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                if (rij0 < var_ptr_d->rcut_pot){

                    double dsmallf_rij = dsmallf_d(rij0, dsf_ptr_d, var_ptr_d);

                    dudr = (dbigf_d(atom_ptr->rho, dbf_ptr_d, var_ptr_d)
                          + dbigf_d(work_ptr->rho, dbf_ptr_d, var_ptr_d))*dsmallf_rij
                          + dpair_d(rij0, dpr_ptr_d, var_ptr_d);

                    #if  defined SLDHL
                      dudr += (dLandauA_d(atom_ptr->rho, dLandauA_ptr_d, var_ptr_d)*si_sq
                             + dLandauB_d(atom_ptr->rho, dLandauB_ptr_d, var_ptr_d)*pow(si_sq,2)
                             + dLandauC_d(atom_ptr->rho, dLandauC_ptr_d, var_ptr_d)*pow(si_sq,3)
                             + dLandauD_d(atom_ptr->rho, dLandauD_ptr_d, var_ptr_d)*pow(si_sq,4))*dsmallf_rij;
                      dudr += (dLandauA_d(work_ptr->rho, dLandauA_ptr_d, var_ptr_d)*sj_sq
                             + dLandauB_d(work_ptr->rho, dLandauB_ptr_d, var_ptr_d)*pow(sj_sq,2)
                             + dLandauC_d(work_ptr->rho, dLandauC_ptr_d, var_ptr_d)*pow(sj_sq,3)
                             + dLandauD_d(work_ptr->rho, dLandauD_ptr_d, var_ptr_d)*pow(sj_sq,4))*dsmallf_rij;
                    #endif

                    pair_enr = pair_d(rij0, pr_ptr_d, var_ptr_d);
                    atom_ptr->pe += 0.5e0*pair_enr;
                }
                #endif

                #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                if (rij0 < var_ptr_d->rcut_mag){

                    double si_dot_sj = vec_dot_d(atom_ptr->s, work_ptr->s);                 //Si.Sj
                    double si_times_sj = vec_length_d(atom_ptr->s)*vec_length_d(work_ptr->s); //|Si|.|Sj|

                    #if defined SLDH || defined SLDHL
                    double dJijdr = dJij_d(rij0, dJij_ptr_d, var_ptr_d);
                    dudr_spin = -dJijdr*(si_dot_sj - si_times_sj); // -dJdr_ij(Si dot Sj  - |Si||Sj|)
                    #endif

                    double Jij_half = Jij_d(rij0, Jij_ptr_d, var_ptr_d)/2e0;

                    double J_times =  Jij_half*si_times_sj;
                    double J_dot   = -Jij_half*si_dot_sj;

                    atom_ptr->me0 += J_times;
                    atom_ptr->me  += J_dot;

                }
                #if defined SLDH || defined SLDHL
                dudr += dudr_spin;
                #endif
                #endif

                #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                double force = -dudr/rij0;
                vector fij = vec_times_d(force, rij);
                atom_ptr->f = vec_add_d(atom_ptr->f, fij);
                atom_ptr->stress11 += fij.x*rij.x;
                atom_ptr->stress22 += fij.y*rij.y;
                atom_ptr->stress33 += fij.z*rij.z;
                atom_ptr->stress12 += fij.x*rij.y;
                atom_ptr->stress23 += fij.y*rij.z;
                atom_ptr->stress31 += fij.z*rij.x;
                atom_ptr->vir += -force*rsq/2e0;
                #endif
              }
              work_ptr = work_ptr->next_atom_ptr;
          }              
     }
    #if defined MD || defined SLDH || defined SLDHL
    atom_ptr->pe += bigf_d(atom_ptr->rho, bf_ptr_d, var_ptr_d);
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        #ifdef extfield
        atom_ptr->me -= vec_dot_d(atom_ptr->s, atom_ptr->Hext);
        #endif
        #ifdef SLDHL
        atom_ptr->me += LandauA_d(atom_ptr->rho, LandauA_ptr_d, var_ptr_d)*si_sq
                      + LandauB_d(atom_ptr->rho, LandauB_ptr_d, var_ptr_d)*pow(si_sq,2)
                      + LandauC_d(atom_ptr->rho, LandauC_ptr_d, var_ptr_d)*pow(si_sq,3)
                      + LandauD_d(atom_ptr->rho, LandauD_ptr_d, var_ptr_d)*pow(si_sq,4);
        #endif
        #ifdef SDHL
        atom_ptr->me += LandauA_d(1, LandauA_ptr_d, var_ptr_d)*si_sq
                      + LandauB_d(1, LandauB_ptr_d, var_ptr_d)*pow(si_sq,2)
                      + LandauC_d(1, LandauC_ptr_d, var_ptr_d)*pow(si_sq,3)
                      + LandauD_d(1, LandauD_ptr_d, var_ptr_d)*pow(si_sq,4);
        #endif
    #endif
}

#endif
