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

/**************************************************************************
* GPU prototypes
**************************************************************************/

__global__ void LP1links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d);
                         
__global__ void LP2links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d,
                         struct cell_struct *first_cell_ptr_d);
                         
__global__ void LP3links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d,
                         int *no_of_relink_ptr_d);
                         
__global__ void LP4links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d,
                         int *delink_atom_index_ptr_d,
                         int *delink_atom_number_ptr_d,
                         int no_of_relink);
                         
__global__ void LP5links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d,
                         struct cell_struct *first_cell_ptr_d,
                         int *delink_atom_index_ptr_d,
                         int *delink_atom_number_ptr_d,
                         int no_of_MP,
                         int no_of_relink);

/**************************************************************************
* CPU codes
**************************************************************************/

void initial_links_GPU(){

    LP1links<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d);

    //Creating double link list. too complicated, only serial runs
    LP2links<<<1,1>>>(var_ptr_d, first_atom_ptr_d, first_cell_ptr_d);

}

void links_GPU(){

    int no_of_relink = 0; //no. of atoms that need to be relink;
    int *no_of_relink_ptr_d;

    cudaMalloc((void**)&no_of_relink_ptr_d, sizeof(int));
    cudaMemcpy(no_of_relink_ptr_d, &no_of_relink, sizeof(int), cudaMemcpyHostToDevice);
 

    LP3links<<<no_of_blocks, no_of_threads>>>(var_ptr_d,
                                              first_atom_ptr_d,
                                              no_of_relink_ptr_d);

    cudaMemcpy(&no_of_relink, no_of_relink_ptr_d, sizeof(int), cudaMemcpyDeviceToHost);

    if (no_of_relink != 0) {

        int *delink_atom_index_ptr_d;
        int *delink_atom_number_ptr_d;
 
        cudaMalloc((void**)&delink_atom_index_ptr_d,  no_of_MP*no_of_relink*sizeof(int));
        cudaMalloc((void**)&delink_atom_number_ptr_d, no_of_MP*sizeof(int));

        LP4links<<<no_of_MP, 1>>>(var_ptr_d, first_atom_ptr_d,
                                  delink_atom_index_ptr_d,
                                  delink_atom_number_ptr_d,
                                  no_of_relink);
      
        //too complicated, run in serial run.
        LP5links<<<1,1>>>(var_ptr_d,
                          first_atom_ptr_d,
                          first_cell_ptr_d,
                          delink_atom_index_ptr_d,
                          delink_atom_number_ptr_d,
                          no_of_MP,
                          no_of_relink);
    
        cudaFree(delink_atom_index_ptr_d);
        cudaFree(delink_atom_number_ptr_d);
    }
    cudaFree(no_of_relink_ptr_d);
}

/**************************************************************************
* GPU codes
**************************************************************************/

__global__ void LP1links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d){
         
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        //initialize some variable for the following
        atom_ptr->this_atom_ptr = atom_ptr;
        atom_ptr->next_atom_ptr = NULL;
        atom_ptr->prev_atom_ptr = NULL;

        atom_ptr->old_cell_index = -1; // give it a fake value
        atom_ptr->new_cell_index = -2; // give it a fake value

        periodic_d(atom_ptr->r,  var_ptr_d);
    }
}

__global__ void LP2links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d,
                         struct cell_struct *first_cell_ptr_d){

    if (blockIdx.x == 0 && threadIdx.x == 0){

        struct atom_struct *atom_ptr;
        struct cell_struct *cell_ptr;

        for (int i = 0 ; i < (var_ptr_d->natom) ; ++i){

            atom_ptr = first_atom_ptr_d + i;
       
            vector q;
            q.x = var_ptr_d->Inv_d.xx*atom_ptr->r.x + var_ptr_d->Inv_d.yx*atom_ptr->r.y + var_ptr_d->Inv_d.zx*atom_ptr->r.z;
            q.y =                                     var_ptr_d->Inv_d.yy*atom_ptr->r.y + var_ptr_d->Inv_d.zy*atom_ptr->r.z;
            q.z =                                                                         var_ptr_d->Inv_d.zz*atom_ptr->r.z;

            int icell = int(q.x*var_ptr_d->no_of_link_cell_x)
                      + int(q.y*var_ptr_d->no_of_link_cell_y)*(var_ptr_d->no_of_link_cell_x)
                      + int(q.z*var_ptr_d->no_of_link_cell_z)*(var_ptr_d->no_of_link_cell_x)*(var_ptr_d->no_of_link_cell_y);

            cell_ptr = first_cell_ptr_d + icell;

            atom_ptr->new_cell_index = icell; // On which cell that such atom is a member
        
            atom_ptr->prev_atom_ptr = NULL;
            atom_ptr->next_atom_ptr = cell_ptr->head_ptr;
            cell_ptr->head_ptr      = atom_ptr;
            ++(cell_ptr->no_of_atoms_in_cell);
       
            if (atom_ptr->next_atom_ptr == NULL)
                cell_ptr->tail_ptr = atom_ptr;
            else
                atom_ptr->next_atom_ptr->prev_atom_ptr = atom_ptr;
        }
    }
}

__global__ void LP3links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d,
                         int *no_of_relink_ptr_d){


    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;

        atom_ptr->old_cell_index = atom_ptr->new_cell_index;

        vector q;
        q.x = var_ptr_d->Inv_d.xx*atom_ptr->r.x + var_ptr_d->Inv_d.yx*atom_ptr->r.y + var_ptr_d->Inv_d.zx*atom_ptr->r.z;
        q.y =                                     var_ptr_d->Inv_d.yy*atom_ptr->r.y + var_ptr_d->Inv_d.zy*atom_ptr->r.z;
        q.z =                                                                         var_ptr_d->Inv_d.zz*atom_ptr->r.z;

        int linkx = int(q.x*var_ptr_d->no_of_link_cell_x);
        int linky = int(q.y*var_ptr_d->no_of_link_cell_y);
        int linkz = int(q.z*var_ptr_d->no_of_link_cell_z);

        if (linkx < 0 ) linkx = 0;
        if (linky < 0 ) linky = 0;
        if (linkz < 0 ) linkz = 0;
        if (linkx >= var_ptr_d->no_of_link_cell_x) linkx = var_ptr_d->no_of_link_cell_x - 1;
        if (linky >= var_ptr_d->no_of_link_cell_y) linky = var_ptr_d->no_of_link_cell_y - 1;
        if (linkz >= var_ptr_d->no_of_link_cell_z) linkz = var_ptr_d->no_of_link_cell_z - 1;

        int icell = linkx
                  + linky*(var_ptr_d->no_of_link_cell_x)
                  + linkz*(var_ptr_d->no_of_link_cell_x)*(var_ptr_d->no_of_link_cell_y);

        atom_ptr->new_cell_index = icell; // On which cell that such atom is a member

        if (icell != atom_ptr->old_cell_index)  atomicAdd(no_of_relink_ptr_d, 1); //must use Nvidia GPU architeture >= 1.1
    }

}

__global__ void LP4links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d,
                         int *delink_atom_index_ptr_d,
                         int *delink_atom_number_ptr_d,
                         int no_of_relink){

    int k = blockIdx.x;
    if (threadIdx.x == 0){
        *(delink_atom_number_ptr_d + blockIdx.x) = 0;
        struct atom_struct *atom_ptr;
        int area = (var_ptr_d->natom - 1)/gridDim.x + 1;
        for (int j = 0; j < area; ++j){
            int i = k*area + j;
            if (i < var_ptr_d->natom){
                atom_ptr = first_atom_ptr_d + i;
                if (atom_ptr->new_cell_index != atom_ptr->old_cell_index){
                    *(delink_atom_index_ptr_d + blockIdx.x*no_of_relink + *(delink_atom_number_ptr_d + blockIdx.x)) = i;
                    ++(*(delink_atom_number_ptr_d + blockIdx.x));
                }
            }
        }
    }
}

__global__ void LP5links(struct varGPU *var_ptr_d,
                         struct atom_struct *first_atom_ptr_d,
                         struct cell_struct *first_cell_ptr_d,
                         int *delink_atom_index_ptr_d,
                         int *delink_atom_number_ptr_d,
                         int no_of_MP,
                         int no_of_relink){

    if (blockIdx.x == 0 && threadIdx.x == 0){
        struct atom_struct *atom_ptr;
        for (int i = 0 ; i < no_of_MP ; ++i){
            for (int j = 0; j < *(delink_atom_number_ptr_d + i); ++j){
                atom_ptr = first_atom_ptr_d + *(delink_atom_index_ptr_d + i*no_of_relink + j);
            
                //detach the previous and next atom pointers
                if (atom_ptr->prev_atom_ptr == NULL)
                    (first_cell_ptr_d + (atom_ptr->old_cell_index))->head_ptr = atom_ptr->next_atom_ptr;
                else
                    atom_ptr->prev_atom_ptr->next_atom_ptr = atom_ptr->next_atom_ptr;
 
                if (atom_ptr->next_atom_ptr == NULL)
                    (first_cell_ptr_d + (atom_ptr->old_cell_index))->tail_ptr = atom_ptr->prev_atom_ptr;
                else
                    atom_ptr->next_atom_ptr->prev_atom_ptr = atom_ptr->prev_atom_ptr;
 
                //attach to new cell at the front
                atom_ptr->prev_atom_ptr = NULL;
                atom_ptr->next_atom_ptr = (first_cell_ptr_d + (atom_ptr->new_cell_index))->head_ptr;
                (first_cell_ptr_d + (atom_ptr->new_cell_index))->head_ptr  = atom_ptr;
                   
                if (atom_ptr->next_atom_ptr == NULL)
                    (first_cell_ptr_d + (atom_ptr->new_cell_index))->tail_ptr = atom_ptr;
                else
                    atom_ptr->next_atom_ptr->prev_atom_ptr = atom_ptr;
 
                --((first_cell_ptr_d+(atom_ptr->old_cell_index))->no_of_atoms_in_cell);
                ++((first_cell_ptr_d+(atom_ptr->new_cell_index))->no_of_atoms_in_cell);
            }
        }
    }
}

void initial_links(){
      initial_links_GPU();
}

void links(){
      links_GPU();
}


#endif
