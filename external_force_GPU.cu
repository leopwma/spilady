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

#if defined GPU

#include "spilady.h"

#if defined extforce

#include "prototype_GPU.h"

/************************************************************************
* GPU prototypes
************************************************************************/

__global__ void LP1extforce(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d);

/************************************************************************
* CPU codes
************************************************************************/

void external_force_GPU(int current_step){

    static bool infile_extforce = 0;

    if (current_step ==  -1){
        ifstream infile("extforce.in");

        if (infile) {
            cout << "Reading external forces file!!!" << '\n';
            infile_extforce = 1;

            int temp;
            infile >> natom;
            for (int i = 0; i < natom; ++i){
                struct atom_struct* atom_ptr;
                atom_ptr = first_atom_ptr + i;
                infile >> temp >> atom_ptr->fext.x >> atom_ptr->fext.y >> atom_ptr->fext.z;
            }
        }
    }

    if (infile_extforce == 0){

        if (current_step == 0) cout <<  "User defined external forces apply." << '\n';
        LP1extforce<<<no_of_blocks, no_of_threads>>>(var_ptr_d, first_atom_ptr_d);
    }
    
}

void external_force(int current_step){
    external_force_GPU(current_step);
}

/**************************************************************************************
* GPU codes
**************************************************************************************/

__global__ void LP1extforce(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d){

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        double pushing_force = 8e8;
        double time_const = 4e-11; // = total_time
        struct atom_struct* atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        if (atom_ptr->r.y < 5e0*var_ptr_d->unit_cell_edge_y || atom_ptr->r.y > 41e0*var_ptr_d->unit_cell_edge_y){
            atom_ptr->fext.x = pushing_force*time_const *(atom_ptr->r.y - var_ptr_d->box_length_half.y)/var_ptr_d->box_length_half.y;
        } else {
            atom_ptr->fext.x = 0e0;
        }
        atom_ptr->fext.y = 0e0;
        atom_ptr->fext.z = 0e0;
    }
}

#endif
#endif



