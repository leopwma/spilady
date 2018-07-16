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

#if defined CPU && (defined MD || defined SLDH || defined SLDHL || defined SLDNC)

#include "spilady.h"

void embedded_rho(atom_struct *atom_ptr){

     struct atom_struct *work_ptr;

     struct cell_struct *ccell_ptr; //current cell pointer
     struct cell_struct *wcell_ptr; //working cell pointer

     ccell_ptr = first_cell_ptr + atom_ptr->new_cell_index;
     for (int i = 0; i <= 13; ++i){
          if (i == 13)
             wcell_ptr = ccell_ptr;
          else
             wcell_ptr = first_cell_ptr + (ccell_ptr->neigh_cell[i]);

          work_ptr = wcell_ptr->head_ptr;
          while (work_ptr != NULL){

              if (work_ptr == atom_ptr && i == 13) break;

              vector rij = vec_sub(atom_ptr->r, work_ptr->r);

              //find image of j closest to i
              find_image(rij);

              double rij0 = vec_length(rij);

              if (rij0 < rcut_pot &&  atom_ptr != work_ptr){
                 double smallf_rij = smallf(rij0);
                 #pragma omp atomic
                 atom_ptr->rho += smallf_rij;
                 #pragma omp atomic
                 work_ptr->rho += smallf_rij;
              }
              work_ptr = work_ptr->next_atom_ptr;
          }
     }
}

void calculate_rho_CPU(){

    #pragma omp parallel for
    for (int i = 0; i < natom; ++i) (first_atom_ptr+i)->rho = 0e0;

    for (int i = 0 ; i < ngroups ; ++i){
        #pragma omp parallel for
        for (int j = 0 ; j < *(allocate_threads_ptr+i); ++j){
            struct atom_struct *atom_ptr;
            atom_ptr = (*(allocate_cell_ptr_ptr + i*max_no_of_members + j))->head_ptr;

            while(atom_ptr != NULL){
                embedded_rho(atom_ptr); // calculate the rho
                atom_ptr = atom_ptr->next_atom_ptr;
            }
        }
    }
    
    //#pragma omp parallel for
    //for (int i = 0; i < natom; ++i) embedded_rho((first_atom_ptr+i));

}

void calculate_rho(){
    calculate_rho_CPU();
}

#endif
