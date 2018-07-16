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
* GPU Prototype
******************************************************************************/

__global__ void LP0allocate(struct cell_struct *first_cell_ptr_d, int ncells);
__global__ void LP1allocate(struct cell_struct *first_cell_ptr_d, int ncells,
                            int *max_no_of_members_ptr_d, int *no_of_groups_ptr_d);
__global__ void LP2allocate(struct cell_struct *first_cell_ptr_d, int ncells,
                            int *max_no_of_members_ptr_d, int *no_of_groups_ptr_d,
                            struct cell_struct **allocate_cell_ptr_ptr_d,
                            int *allocate_threads_ptr_d);

/******************************************************************************
* CPU codes
******************************************************************************/

void allocate_cells_GPU(){
    
    no_of_blocks_cell = (ncells + no_of_threads - 1)/no_of_threads;

    LP0allocate<<<no_of_blocks_cell, no_of_threads>>>(first_cell_ptr_d, ncells);

    max_no_of_members = 0;
    cudaMalloc((void**)&max_no_of_members_ptr_d, sizeof(int));
    cudaMemcpy(max_no_of_members_ptr_d, &max_no_of_members, sizeof(int), cudaMemcpyHostToDevice);

    int no_of_groups = 0;
    int *no_of_groups_ptr_d = 0;
    cudaMalloc((void**)&no_of_groups_ptr_d, sizeof(int));
    cudaMemcpy(no_of_groups_ptr_d, &no_of_groups, sizeof(int), cudaMemcpyHostToDevice);
        
    LP1allocate<<<1,1>>>(first_cell_ptr_d, ncells, max_no_of_members_ptr_d, no_of_groups_ptr_d);

    cudaMemcpy(&max_no_of_members, max_no_of_members_ptr_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&no_of_groups, no_of_groups_ptr_d, sizeof(int), cudaMemcpyDeviceToHost);

    // at this point, the no_of_groups is the total no. of group
    cout << "No. of group in parallel programming for symplectic method = " << no_of_groups << '\n';
    
    cudaMalloc((void**)&allocate_cell_ptr_ptr_d, no_of_groups*max_no_of_members*sizeof(cell_struct*));
    cudaMalloc((void**)&allocate_threads_ptr_d, no_of_groups*sizeof(int));

    LP0allocate<<<no_of_blocks_cell, no_of_threads>>>(first_cell_ptr_d, ncells);
    
    no_of_groups = 0; //the no_of_groups is reinitialized
    cudaMemcpy(no_of_groups_ptr_d, &no_of_groups, sizeof(int), cudaMemcpyHostToDevice);

    LP2allocate<<<1,1>>>(first_cell_ptr_d, ncells, max_no_of_members_ptr_d, no_of_groups_ptr_d,
                         allocate_cell_ptr_ptr_d, allocate_threads_ptr_d);


    cudaMemcpy(&no_of_groups, no_of_groups_ptr_d, sizeof(int), cudaMemcpyDeviceToHost);
    ngroups = no_of_groups;

    no_of_blocks_members = (max_no_of_members + no_of_threads - 1)/no_of_threads;

    cudaFree(no_of_groups_ptr_d);
}

void free_allocate_memory_GPU(){

    cudaFree(allocate_cell_ptr_ptr_d);
    cudaFree(allocate_threads_ptr_d);
    cudaFree(max_no_of_members_ptr_d);
}

void allocate_cells(){
    allocate_cells_GPU();
}

void free_allocate_memory(){
    free_allocate_memory_GPU();
}


/******************************************************************************
* GPU codes
******************************************************************************/

__global__ void LP0allocate(struct cell_struct *first_cell_ptr_d, int ncells){
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < ncells) (first_cell_ptr_d + i)->type = 0;
}


__global__ void LP1allocate(struct cell_struct *first_cell_ptr_d, int ncells,
                            int *max_no_of_members_ptr_d, int *no_of_groups_ptr_d){

    int no_of_cells_left = ncells;
    struct cell_struct *cell_ptr;
    while(1){

        int no_of_members = 0;
        for (int i = 0 ; i < ncells; ++i){

            cell_ptr = first_cell_ptr_d + i;

            if (cell_ptr->type == 0 ){
                ++no_of_members;
                --no_of_cells_left;
                cell_ptr->type = 1;
                for (int j_neigh = 0 ; j_neigh < 26; ++j_neigh){
                    if ((first_cell_ptr_d+(cell_ptr->neigh_cell[j_neigh]))->type != 1)
                        (first_cell_ptr_d+(cell_ptr->neigh_cell[j_neigh]))->type = -1;
                }
            }
        }
        ++(*(no_of_groups_ptr_d));

        for (int i = 0 ; i < ncells; ++i){
            if ((first_cell_ptr_d+i)->type == -1) (first_cell_ptr_d+i)->type = 0;
        }   
        if (no_of_members > *max_no_of_members_ptr_d) *max_no_of_members_ptr_d = no_of_members;
        if (no_of_cells_left == 0) break;
    }
}

__global__ void LP2allocate(struct cell_struct *first_cell_ptr_d, int ncells,
                            int *max_no_of_members_ptr_d, int *no_of_groups_ptr_d,
                            struct cell_struct **allocate_cell_ptr_ptr_d,
                            int *allocate_threads_ptr_d){

    int no_of_cells_left = ncells;
    struct cell_struct *cell_ptr;
    while(1){

        int no_of_members = 0;
        for (int i = 0 ; i < ncells; ++i){
            cell_ptr = first_cell_ptr_d+i;
            if (cell_ptr->type == 0 ){
                *(allocate_cell_ptr_ptr_d + (*no_of_groups_ptr_d)*(*max_no_of_members_ptr_d) + no_of_members) = cell_ptr;
                cell_ptr->type = 1;
                ++no_of_members;
                --no_of_cells_left;
                for (int j_neigh = 0 ; j_neigh < 26; ++j_neigh){
                    if ((first_cell_ptr_d+(cell_ptr->neigh_cell[j_neigh]))->type != 1)
                        (first_cell_ptr_d+(cell_ptr->neigh_cell[j_neigh]))->type = -1;
                }
            }
            *(allocate_threads_ptr_d + *(no_of_groups_ptr_d)) = no_of_members;
        }
        ++(*no_of_groups_ptr_d);

        for (int i = 0 ; i < ncells; ++i){
            if ((first_cell_ptr_d+i)->type == -1) (first_cell_ptr_d+i)->type = 0;
        }
        if (no_of_cells_left == 0) break;
    }
}


#endif
