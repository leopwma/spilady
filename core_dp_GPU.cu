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
*********************************************************************************
*
*   Edit notes:
*   Date:    Apr 2016
*   Author:  Pui-Wai (Leo) MA
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) variable "Msteps_quantum" and "Nfrequency_quantum" are added.
*   2) Now the quantum noise change every "Msteps_quantum" steps. 
*
*******************************************************************************/

#if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined GPU

#include "spilady.h"
#include "prototype_GPU.h"

/************************************************************************
* GPU prototypes
************************************************************************/

#if defined lattlang && defined localcolmot
__global__ void LP1dp(int i,
                      int *allocate_threads_ptr_d,
                      struct cell_struct **allocate_cell_ptr_ptr_d,
                      int *max_no_of_members_ptr_d,
                      struct varGPU *var_ptr_d,
                      struct atom_struct *first_atom_ptr_d,
                      struct cell_struct *first_cell_ptr_d,
                      curandState *rand_state_ptr_d,
                      double dt);
                      
__global__ void LP2dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                      struct cell_struct *first_cell_ptr_d, double dt);
#endif

#ifdef lattlang
__global__ void LP3dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                      struct cell_struct *first_cell_ptr_d, curandState *rand_state_ptr_d,
                      #ifdef quantumnoise
                      double* quantum_rand_memory_ptr_d, double* H_ptr_d,
                      double* quantum_noise_ptr_d, int* quantum_count_ptr_d,
                      int Msteps, int Nf2,
                      #endif
                      double dt);
                       
__global__ void LP4dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                      struct cell_struct *first_cell_ptr_d, double dt);
#endif

#if defined lattlang && defined localcolmot
__global__ void LP5dp(int i,
                      int *allocate_threads_ptr_d,
                      struct cell_struct **allocate_cell_ptr_ptr_d,
                      int *max_no_of_members_ptr_d,
                      struct varGPU *var_ptr_d,
                      struct atom_struct *first_atom_ptr_d,
                      struct cell_struct *first_cell_ptr_d,
                      double dt);
                       
__global__ void LP6dp(int i,
                      int *allocate_threads_ptr_d,
                      struct cell_struct **allocate_cell_ptr_ptr_d,
                      int *max_no_of_members_ptr_d,
                      struct varGPU *var_ptr_d,
                      struct atom_struct *first_atom_ptr_d,
                      struct cell_struct *first_cell_ptr_d,
                      double dt);
#endif

#ifndef lattlang
__global__ void LP7dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double dt);
#endif

__global__ void LP8dp_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, vector *ave_p_ptr_d);
__global__ void LP8dp_part2(struct varGPU *var_ptr_d, vector *ave_p_ptr_d);
__global__ void LP9dp_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *total_ke_old_ptr_d, double *total_ke_new_ptr_d, vector ave_p);
__global__ void LP9dp_part2(struct varGPU *var_ptr_d, double *total_ke_old_ptr_d, double *total_ke_new_ptr_d);
__global__ void LP10dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double factor);
__global__ void LP11dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d);


/************************************************************************
* CPU codes
************************************************************************/

#ifdef lattlang
void core_dp_A_GPU(double dt);
void core_dp_B_GPU(double dt);
#endif

#if defined lattlang && defined localcolmot
void core_dp_C1_GPU(double dt);
void core_dp_C2_GPU(double dt);
#endif

void rescale_momentum();

void core_dp_GPU(double dt){

    #ifdef lattlang
      #ifdef localcolmot
      core_dp_C1_GPU(dt/2e0);  //subtract average momentum in a cell
      #endif
      core_dp_B_GPU(dt/2e0);   // solution of dp/dt = -gamma/mass*p
      core_dp_A_GPU(dt);       // add (forces + noise)*dt and substract average noise in a cel ifdef localcolmot
      core_dp_B_GPU(dt/2e0);   // solution of dp/dt = -gamma/mass*p
      #ifdef localcolmot
      core_dp_C2_GPU(dt/2e0);  //subtract average momentum in a cell
      #endif
    #else
      LP7dp<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, dt);
    #endif

    rescale_momentum();

    LP11dp<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d);

}

void core_dp(double dt){
    core_dp_GPU(dt);
}

#ifdef lattlang
void core_dp_A_GPU(double dt){

    #ifdef localcolmot
        for (int i = 0 ; i < ngroups ; ++i){
            LP1dp<<<no_of_blocks_members, no_of_threads>>>(i,
                                                           allocate_threads_ptr_d,
                                                           allocate_cell_ptr_ptr_d,
                                                           max_no_of_members_ptr_d,
                                                           var_ptr_d,
                                                           first_atom_ptr_d,
                                                           first_cell_ptr_d,
                                                           rand_state_ptr_d,
                                                           dt);
        }
        LP2dp<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d,
                                               first_cell_ptr_d, dt);
    #else
        LP3dp<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d,
                                               first_cell_ptr_d, rand_state_ptr_d,
                                               #ifdef quantumnoise
                                               quantum_rand_memory_ptr_d, H_ptr_d,
                                               quantum_noise_ptr_d, quantum_count_ptr_d,
                                               Msteps_quantum, Nfrequency_quantum_2,
                                               #endif
                                               dt);
    #endif

}

void core_dp_B_GPU(double dt){
    LP4dp<<<no_of_blocks,no_of_threads>>>(var_ptr_d, first_atom_ptr_d, first_cell_ptr_d, dt);
}
#endif

#if defined lattlang && defined localcolmot
void core_dp_C1_GPU(double dt){

    for (int i = 0 ; i < ngroups ; ++i){
        LP5dp<<<no_of_blocks_members, no_of_threads>>>(i,
                                                       allocate_threads_ptr_d,
                                                       allocate_cell_ptr_ptr_d,
                                                       max_no_of_members_ptr_d,
                                                       var_ptr_d,
                                                       first_atom_ptr_d,
                                                       first_cell_ptr_d,
                                                       dt);
    }
}

void core_dp_C2_GPU(double dt){

    for (int i = ngroups - 1 ; i >=0 ; --i){
        LP6dp<<<no_of_blocks_members, no_of_threads>>>(i,
                                                       allocate_threads_ptr_d,
                                                       allocate_cell_ptr_ptr_d,
                                                       max_no_of_members_ptr_d,
                                                       var_ptr_d,
                                                       first_atom_ptr_d,
                                                       first_cell_ptr_d,
                                                       dt);
    }
}

#endif

void rescale_momentum(){

    size_t size1 = no_of_MP*no_of_threads*sizeof(vector);

    vector ave_p = vec_zero();
    vector *ave_p_ptr_d;
    cudaMalloc((void**)&ave_p_ptr_d, size1);
    LP8dp_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, ave_p_ptr_d);
    LP8dp_part2<<<no_of_MP, no_of_threads>>>(var_ptr_d, ave_p_ptr_d);
    cudaMemcpy(&ave_p, ave_p_ptr_d, sizeof(vector), cudaMemcpyDeviceToHost);

    ave_p = vec_divide(ave_p, double(natom));

    size_t size2 = no_of_MP*no_of_threads*sizeof(double);

    double total_ke_old = 0e0;
    double total_ke_new = 0e0;
    double *total_ke_old_ptr_d;
    double *total_ke_new_ptr_d;
    cudaMalloc((void**)&total_ke_old_ptr_d, size2);
    cudaMalloc((void**)&total_ke_new_ptr_d, size2);
    LP9dp_part1<<<no_of_MP, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, total_ke_old_ptr_d, total_ke_new_ptr_d, ave_p);
    LP9dp_part2<<<no_of_MP, no_of_threads>>>(var_ptr_d, total_ke_old_ptr_d, total_ke_new_ptr_d);
    cudaMemcpy(&total_ke_old, total_ke_old_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&total_ke_new, total_ke_new_ptr_d, sizeof(double), cudaMemcpyDeviceToHost);

    double factor = sqrt(total_ke_old/total_ke_new);
    if(total_ke_new < 1e-10) factor = 1e0;

    LP10dp<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d, factor);

    cudaFree(ave_p_ptr_d);
    cudaFree(total_ke_old_ptr_d);
    cudaFree(total_ke_new_ptr_d);

}



/**************************************************************************************
* GPU codes
**************************************************************************************/

#if defined lattlang && defined localcolmot

__global__ void LP1dp(int i,
                      int *allocate_threads_ptr_d,
                      struct cell_struct **allocate_cell_ptr_ptr_d,
                      int *max_no_of_members_ptr_d,
                      struct varGPU *var_ptr_d,
                      struct atom_struct *first_atom_ptr_d,
                      struct cell_struct *first_cell_ptr_d,
                      curandState *rand_state_ptr_d,
                      double dt){

    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < *(allocate_threads_ptr_d + i)){

        struct atom_struct *atom_ptr;
        atom_ptr = (*(allocate_cell_ptr_ptr_d + i*(*max_no_of_members_ptr_d) + j))->head_ptr;
        struct cell_struct *cell_ptr;
        
        bool ave_activated = 0;

        if (atom_ptr != NULL){
            cell_ptr = first_cell_ptr_d + atom_ptr->new_cell_index;
            cell_ptr->ave_fluct_force = vec_zero_d();
            ave_activated = 1;
        }

        while(atom_ptr != NULL){

            #ifdef extforce
            atom_ptr->f = vec_add_d(atom_ptr->f, atom_ptr->fext);
            #endif

            //generating random numbers
            vector fluct_force;
            fluct_force.x = normal_rand_d(rand_state_ptr_d + j);
            fluct_force.y = normal_rand_d(rand_state_ptr_d + j);
            fluct_force.z = normal_rand_d(rand_state_ptr_d + j);

            #ifdef eltemp
            double fluct_force_length = sqrt(2e0*cell_ptr->Te*var_ptr_d->gamma_L/dt);
            #else
            double fluct_force_length = sqrt(2e0*var_ptr_d->temperature*var_ptr_d->gamma_L/dt);
            #endif

            fluct_force = vec_times_d(fluct_force_length, fluct_force);

            atom_ptr->p = vec_add_d(atom_ptr->p, vec_times_d(dt, vec_add_d(atom_ptr->f, fluct_force)));
            cell_ptr->ave_fluct_force = vec_add_d(cell_ptr->ave_fluct_force, fluct_force);

            atom_ptr = atom_ptr->next_atom_ptr;
        }

        if (ave_activated){
            cell_ptr->ave_fluct_force = vec_divide_d(cell_ptr->ave_fluct_force, cell_ptr->no_of_atoms_in_cell);
        }      
        
    }
}

__global__ void LP2dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                       struct cell_struct *first_cell_ptr_d, double dt){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;

        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr_d + (atom_ptr->new_cell_index);

        atom_ptr->p = vec_sub_d(atom_ptr->p, vec_times_d(dt, cell_ptr->ave_fluct_force));
    }
}

#endif

#ifdef lattlang

__global__ void LP3dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                       struct cell_struct *first_cell_ptr_d, curandState *rand_state_ptr_d,
                       #ifdef quantumnoise
                       double* quantum_rand_memory_ptr_d, double* H_ptr_d, 
                       double* quantum_noise_ptr_d, int* quantum_count_ptr_d,
                       int Msteps, int Nf2,
                       #endif
                       double dt){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;

        #ifdef extforce
        atom_ptr->f = vec_add_d(atom_ptr->f, atom_ptr->fext);
        #endif
        vector fluct_force;

        //generating random numbers
        #ifdef quantumnoise
          double h = dt*Msteps;
          double fluct_force_length = sqrt(2e0*var_ptr_d->gamma_L/h);
          fluct_force.x = quantum_noise_d(3*i,   i,  quantum_rand_memory_ptr_d, rand_state_ptr_d, H_ptr_d, 
                                          quantum_noise_ptr_d, quantum_count_ptr_d, Msteps, Nf2);
          fluct_force.y = quantum_noise_d(3*i+1, i,  quantum_rand_memory_ptr_d, rand_state_ptr_d, H_ptr_d,
                                          quantum_noise_ptr_d, quantum_count_ptr_d, Msteps, Nf2);
          fluct_force.z = quantum_noise_d(3*i+2, i,  quantum_rand_memory_ptr_d, rand_state_ptr_d, H_ptr_d,
                                          quantum_noise_ptr_d, quantum_count_ptr_d, Msteps, Nf2);
        #else
          #ifdef eltemp
          struct cell_struct *cell_ptr;
          cell_ptr = first_cell_ptr_d + (atom_ptr->new_cell_index);
          double fluct_force_length = sqrt(2e0*cell_ptr->Te*var_ptr_d->gamma_L/dt);
          #else
          double fluct_force_length = sqrt(2e0*var_ptr_d->temperature*var_ptr_d->gamma_L/dt);
          #endif
          fluct_force.x = normal_rand_d(rand_state_ptr_d + i);
          fluct_force.y = normal_rand_d(rand_state_ptr_d + i);
          fluct_force.z = normal_rand_d(rand_state_ptr_d + i);
        #endif

        fluct_force = vec_times_d(fluct_force_length, fluct_force);

        atom_ptr->p = vec_add_d(atom_ptr->p, vec_times_d(dt, vec_add_d(atom_ptr->f, fluct_force)));

    }
}

__global__ void LP4dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                       struct cell_struct *first_cell_ptr_d, double dt){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        #ifdef localcolmot
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr_d + atom_ptr->new_cell_index;
        double exp_dt = exp(-(var_ptr_d->gamma_L)/(var_ptr_d->atmass)*dt*(1e0-1e0/cell_ptr->no_of_atoms_in_cell));
        #else
        double exp_dt = exp(-(var_ptr_d->gamma_L)/(var_ptr_d->atmass)*dt);
        #endif

        atom_ptr->p = vec_times_d(exp_dt, atom_ptr->p);
    }
}

#endif

#if defined lattlang && defined localcolmot

__global__ void LP5dp(int i,
                       int *allocate_threads_ptr_d,
                       struct cell_struct **allocate_cell_ptr_ptr_d,
                       int *max_no_of_members_ptr_d,
                       struct varGPU *var_ptr_d,
                       struct atom_struct *first_atom_ptr_d,
                       struct cell_struct *first_cell_ptr_d,
                       double dt){

    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < *(allocate_threads_ptr_d + i)){

        struct atom_struct *atom_ptr;
        atom_ptr = (*(allocate_cell_ptr_ptr_d + i*(*max_no_of_members_ptr_d) + j))->head_ptr;
        while(atom_ptr != NULL){
            vector sum_p;
            sum_p = vec_zero_d();
            struct atom_struct *work_ptr;
            work_ptr = (*(allocate_cell_ptr_ptr_d + i*(*max_no_of_members_ptr_d) + j))->head_ptr;
            while(work_ptr != NULL){
                sum_p = vec_add_d(sum_p, work_ptr->p);
                work_ptr = work_ptr->next_atom_ptr;
            }
            double factor = var_ptr_d->gamma_L/var_ptr_d->atmass*dt/
                      ((first_cell_ptr_d + atom_ptr->new_cell_index)->no_of_atoms_in_cell);

            atom_ptr->p = vec_add_d(atom_ptr->p, vec_times_d(factor, vec_sub_d(sum_p, atom_ptr->p)));
            atom_ptr = atom_ptr->next_atom_ptr;
        }
    }
}

__global__ void LP6dp(int i,
                      int *allocate_threads_ptr_d,
                      struct cell_struct **allocate_cell_ptr_ptr_d,
                      int *max_no_of_members_ptr_d,
                      struct varGPU *var_ptr_d,
                      struct atom_struct *first_atom_ptr_d,
                      struct cell_struct *first_cell_ptr_d,
                      double dt){

    int j = blockIdx.x*blockDim.x + threadIdx.x;
    if (j < *(allocate_threads_ptr_d + i)){

        struct atom_struct *atom_ptr;
        atom_ptr = (*(allocate_cell_ptr_ptr_d + i*(*max_no_of_members_ptr_d) + j))->tail_ptr;
        while(atom_ptr != NULL){
            vector sum_p;
            sum_p = vec_zero_d();
            struct atom_struct *work_ptr;
            work_ptr = (*(allocate_cell_ptr_ptr_d + i*(*max_no_of_members_ptr_d) + j))->tail_ptr;
            while(work_ptr != NULL){
                sum_p = vec_add_d(sum_p, work_ptr->p);
                work_ptr = work_ptr->prev_atom_ptr;
            }
            double factor = var_ptr_d->gamma_L/var_ptr_d->atmass*dt/
                            ((first_cell_ptr_d + atom_ptr->new_cell_index)->no_of_atoms_in_cell);

            atom_ptr->p = vec_add_d(atom_ptr->p, vec_times_d(factor, vec_sub_d(sum_p, atom_ptr->p)));
            atom_ptr = atom_ptr->prev_atom_ptr;
        }
    }
}

#endif

#ifndef lattlang
__global__ void LP7dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double dt){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        #ifdef quench
        atom_ptr->p = vec_zero_d();
        #endif
        atom_ptr->p = vec_add_d(atom_ptr->p, vec_times_d(dt,atom_ptr->f));

     }
}
#endif

__global__ void LP8dp_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, vector *ave_p_ptr_d)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(ave_p_ptr_d + i) = vec_zero_d();
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
            struct atom_struct *atom_ptr;
            atom_ptr = first_atom_ptr_d + m;
            *(ave_p_ptr_d + i) = vec_add_d(*(ave_p_ptr_d + i), atom_ptr->p);
        }
    }

    __syncthreads();
}

__global__ void LP8dp_part2(struct varGPU *var_ptr_d, vector *ave_p_ptr_d)
{

    int depth = blockIdx.x*blockDim.x;

    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(ave_p_ptr_d + depth) = vec_add_d(*(ave_p_ptr_d + depth), *(ave_p_ptr_d + depth + j));
    }
     __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *ave_p_ptr_d = vec_add_d(*ave_p_ptr_d, *(ave_p_ptr_d + j*blockDim.x));
    }
}



__global__ void LP9dp_part1(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double *total_ke_old_ptr_d, double *total_ke_new_ptr_d, vector ave_p)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    *(total_ke_old_ptr_d + i) = 0e0;
    *(total_ke_new_ptr_d + i) = 0e0;
    int area = blockDim.x*gridDim.x;
    int k = (var_ptr_d->natom - 1)/area + 1;
    for (int j = 0; j < k; ++j){
        int m = i + j*area;
        if (m < var_ptr_d->natom) {
            struct atom_struct *atom_ptr;
            atom_ptr = first_atom_ptr_d + m;
            atom_ptr->ke = vec_sq_d(atom_ptr->p)/2e0/var_ptr_d->atmass;
            *(total_ke_old_ptr_d + i) += atom_ptr->ke;
            atom_ptr->p = vec_sub_d(atom_ptr->p, ave_p);
            atom_ptr->ke = vec_sq_d(atom_ptr->p)/2e0/var_ptr_d->atmass;
            *(total_ke_new_ptr_d + i) += atom_ptr->ke;
        }
    }

    __syncthreads();
}

__global__ void LP9dp_part2(struct varGPU *var_ptr_d, double *total_ke_old_ptr_d, double *total_ke_new_ptr_d)
{
    int depth = blockIdx.x*blockDim.x;
    if (threadIdx.x == 0){
        for (int j = 1; j < blockDim.x; ++j) *(total_ke_old_ptr_d + depth) += *(total_ke_old_ptr_d + depth + j);
    }
    if (threadIdx.x == 1){
        for (int j = 1; j < blockDim.x; ++j) *(total_ke_new_ptr_d + depth) += *(total_ke_new_ptr_d + depth + j);
    }
    __threadfence();

    if (blockIdx.x == 0 && threadIdx.x == 0){
        for (int j = 1; j < gridDim.x; ++j) *total_ke_old_ptr_d += *(total_ke_old_ptr_d + j*blockDim.x);
    }
    if (blockIdx.x == 0 && threadIdx.x == 1){
        for (int j = 1; j < gridDim.x; ++j) *total_ke_new_ptr_d += *(total_ke_new_ptr_d + j*blockDim.x);
    }
}

__global__ void LP10dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double factor)
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        atom_ptr->p = vec_times_d(factor,atom_ptr->p);
     }
}

__global__ void LP11dp(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d)
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        atom_ptr->ke = vec_sq_d(atom_ptr->p)/2e0/var_ptr_d->atmass;
     }
}


#endif
