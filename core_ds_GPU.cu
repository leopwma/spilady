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

#if (defined SDH || defined SDHL || defined SLDH || defined SLDHL) && defined GPU

#include "spilady.h"
#include "prototype_GPU.h"

/***************************************************************************************
* GPU prototypes
***************************************************************************************/
#ifdef magmom
__global__ void LP1ds(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d);
#endif

__global__ void LP2ds(int i, double dt_half, curandState *rand_state_ptr_d,
                       int *allocate_threads_ptr_d,
                       struct cell_struct **allocate_cell_ptr_ptr_d,
                       int *max_no_of_members_ptr_d,
                       struct varGPU *var_ptr_d,
                       struct cell_struct *first_cell_ptr_d,
                       double *Jij_ptr_d
                       #if defined SDHL || defined SLDHL
                       , double *LandauA_ptr_d
                       , double *LandauB_ptr_d
                       , double *LandauC_ptr_d
                       , double *LandauD_ptr_d
                       #endif
                       );

__global__ void LP3ds(int i, double dt_half, curandState *rand_state_ptr_d,
                       int *allocate_threads_ptr_d,
                       struct cell_struct **allocate_cell_ptr_ptr_d,
                       int *max_no_of_members_ptr_d,
                       struct varGPU *var_ptr_d,
                       struct cell_struct *first_cell_ptr_d,
                       double *Jij_ptr_d
                       #if defined SDHL || defined SLDHL
                       , double *LandauA_ptr_d
                       , double *LandauB_ptr_d
                       , double *LandauC_ptr_d
                       , double *LandauD_ptr_d
                       #endif
                       );

#ifdef magmom
__global__ void LP4ds(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d);
#endif

/***************************************************************************************
* CPU codes
***************************************************************************************/

void core_ds_GPU(double dt){

    double dt_half = dt/2e0;

    #ifdef magmom
    LP1ds<<<no_of_blocks,no_of_threads>>>(var_ptr_d, first_atom_ptr_d);
    #endif

    for (int i = 0 ; i < ngroups ; ++i){
        LP2ds<<<no_of_blocks_members, no_of_threads>>>(i, dt_half, rand_state_ptr_d,
                                                       allocate_threads_ptr_d,
                                                       allocate_cell_ptr_ptr_d,
                                                       max_no_of_members_ptr_d,
                                                       var_ptr_d,
                                                       first_cell_ptr_d,
                                                       Jij_ptr_d
                                                       #if defined SDHL || defined SLDHL
                                                       , LandauA_ptr_d
                                                       , LandauB_ptr_d
                                                       , LandauC_ptr_d
                                                       , LandauD_ptr_d
                                                       #endif
                                                       );
    }

    for (int i = ngroups - 1 ; i >=0 ; --i){
        LP3ds<<<no_of_blocks_members, no_of_threads>>>(i, dt_half, rand_state_ptr_d,
                                                       allocate_threads_ptr_d,
                                                       allocate_cell_ptr_ptr_d,
                                                       max_no_of_members_ptr_d,
                                                       var_ptr_d,
                                                       first_cell_ptr_d,
                                                       Jij_ptr_d
                                                       #if defined SDHL || defined SLDHL
                                                       , LandauA_ptr_d
                                                       , LandauB_ptr_d
                                                       , LandauC_ptr_d
                                                       , LandauD_ptr_d
                                                       #endif
                                                       );
    }

    #ifdef magmom
    LP4ds<<<no_of_blocks,no_of_threads>>>(var_ptr_d, first_atom_ptr_d);
    #endif
}

void core_ds(double dt){
    core_ds_GPU(dt);
}

/***************************************************************************************
* GPU codes
***************************************************************************************/
#ifdef magmom
__global__ void LP1ds(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        atom_ptr->s  = vec_divide_d(atom_ptr->m,-el_g);
        atom_ptr->s0 = vec_length_d(atom_ptr->s);
    }
}
#endif

__global__ void LP2ds(int i, double dt_half, curandState *rand_state_ptr_d,
                       int *allocate_threads_ptr_d,
                       struct cell_struct **allocate_cell_ptr_ptr_d,
                       int *max_no_of_members_ptr_d,
                       struct varGPU *var_ptr_d,
                       struct cell_struct *first_cell_ptr_d,
                       double *Jij_ptr_d
                       #if defined SDHL || defined SLDHL
                       , double *LandauA_ptr_d
                       , double *LandauB_ptr_d
                       , double *LandauC_ptr_d
                       , double *LandauD_ptr_d
                       #endif
                       )
{

        int j = blockIdx.x*blockDim.x + threadIdx.x;
        if (j < *(allocate_threads_ptr_d + i)){

            struct atom_struct *atom_ptr;
            atom_ptr = (*(allocate_cell_ptr_ptr_d + i*(*max_no_of_members_ptr_d) + j))->head_ptr;
            while(atom_ptr != NULL){
                calculate_spin_d(j, rand_state_ptr_d,
                                 var_ptr_d,
                                 atom_ptr,
                                 first_cell_ptr_d,
                                 dt_half,
                                 Jij_ptr_d
                                 #if defined SDHL || defined SLDHL
                                 , LandauA_ptr_d
                                 , LandauB_ptr_d
                                 , LandauC_ptr_d
                                 , LandauD_ptr_d
                                 #endif
                                 );
                atom_ptr = atom_ptr->next_atom_ptr;
            }
        }
}

__global__ void LP3ds(int i, double dt_half, curandState *rand_state_ptr_d,
                       int *allocate_threads_ptr_d,
                       struct cell_struct **allocate_cell_ptr_ptr_d,
                       int *max_no_of_members_ptr_d,
                       struct varGPU *var_ptr_d,
                       struct cell_struct *first_cell_ptr_d,
                       double *Jij_ptr_d
                       #if defined SDHL || defined SLDHL
                       , double *LandauA_ptr_d
                       , double *LandauB_ptr_d
                       , double *LandauC_ptr_d
                       , double *LandauD_ptr_d
                       #endif
                       )
{

        int j = blockIdx.x*blockDim.x + threadIdx.x;
        if (j < *(allocate_threads_ptr_d + i)){

            struct atom_struct *atom_ptr;
            atom_ptr = (*(allocate_cell_ptr_ptr_d + i*(*max_no_of_members_ptr_d) + j))->tail_ptr;
            while(atom_ptr != NULL){
                calculate_spin_d(j, rand_state_ptr_d,
                                 var_ptr_d,
                                 atom_ptr,
                                 first_cell_ptr_d,
                                 dt_half,
                                 Jij_ptr_d
                                 #if defined SDHL || defined SLDHL
                                 , LandauA_ptr_d
                                 , LandauB_ptr_d
                                 , LandauC_ptr_d
                                 , LandauD_ptr_d
                                 #endif
                                 );
                atom_ptr = atom_ptr->prev_atom_ptr;
            }
        }
}

#ifdef magmom
__global__ void LP4ds(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        atom_ptr->m  = vec_times_d(-el_g, atom_ptr->s);
        atom_ptr->m0 = vec_length_d(atom_ptr->m);     }
}
#endif

#endif /*GPU*/
