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

#if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined GPU

#include "spilady.h"
#include "prototype_GPU.h"

__global__ void LP1dr(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double dt){

    double dt_over_atmass = dt/var_ptr_d->atmass;

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        atom_ptr->r = vec_add_d(atom_ptr->r, vec_times_d(dt_over_atmass, atom_ptr->p));
        periodic_d(atom_ptr->r,  var_ptr_d);
    }
}

void core_dr_GPU(double dt){

    LP1dr<<<no_of_blocks,no_of_threads>>>(var_ptr_d, first_atom_ptr_d, dt);

}

void core_dr(double dt){
    core_dr_GPU(dt);
}

#endif



