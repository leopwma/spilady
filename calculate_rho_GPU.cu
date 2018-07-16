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

#if defined GPU && (defined MD || defined SLDH || defined SLDHL || defined SLDNC)

#include "spilady.h"
#include "prototype_GPU.h"

/****************************************************************************
* GPU prototyps
*****************************************************************************/
__global__ void LP1rho(struct varGPU *var_ptr_d,
                       struct atom_struct *first_atom_ptr_d);

__global__ void LP2rho(struct varGPU *var_ptr_d,
                       struct atom_struct *first_atom_ptr_d,
                       struct cell_struct *first_cell_ptr_d,
                       double *sf_ptr_d);

__device__ void embedded_rho_d(struct varGPU *var_ptr_d,
                               struct atom_struct *atom_ptr,
                               struct cell_struct *first_cell_ptr_d,
                               double *sf_ptr_d);
                       
/****************************************************************************
* CPU codes
*****************************************************************************/

void calculate_rho_GPU(){

    LP1rho<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d);

    LP2rho<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d,
                                            first_cell_ptr_d, sf_ptr_d);

}

void calculate_rho(){
    calculate_rho_GPU();
}

/****************************************************************************
* GPU codes
*****************************************************************************/


__global__ void LP1rho(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom)  (first_atom_ptr_d+i)->rho = 0e0;
}

__global__ void LP2rho(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d,
                       struct cell_struct *first_cell_ptr_d, double *sf_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom)  embedded_rho_d(var_ptr_d, first_atom_ptr_d + i,
                                              first_cell_ptr_d, sf_ptr_d);
}

__device__ void embedded_rho_d(struct varGPU *var_ptr_d, struct atom_struct *atom_ptr,
                              struct cell_struct *first_cell_ptr_d, double *sf_ptr_d){

     struct atom_struct *work_ptr;

     struct cell_struct *ccell_ptr;   //cell pointer for current atom i
     struct cell_struct *wcell_ptr;   //cell pointer for work_ptr atom j

     ccell_ptr = first_cell_ptr_d + atom_ptr->new_cell_index;
     for (int i = 0; i <= 26; ++i){ //In GPU, we run 26 neighbour cells. We give up the speed up by fij=fji in CPU case.
          if (i == 26)
             wcell_ptr = ccell_ptr;
          else
             wcell_ptr = first_cell_ptr_d + (ccell_ptr->neigh_cell[i]);

          work_ptr = wcell_ptr->head_ptr;
          while (work_ptr != NULL){

              vector rij = vec_sub_d(atom_ptr->r, work_ptr->r);

              //find image of j closest to i
              find_image_d(rij, var_ptr_d);

              double rij0 = vec_length_d(rij);

              if (rij0 < (var_ptr_d->rcut_pot) &&  atom_ptr != work_ptr){
                 double smallf_rij = smallf_d(rij0, sf_ptr_d, var_ptr_d);
                 atom_ptr->rho += smallf_rij;
              }
              work_ptr = work_ptr->next_atom_ptr;
          }
     }
}

#endif
