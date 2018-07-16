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

#if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined CPU

#include "spilady.h"

vector spin_rotation(vector Heff, vector s, double dt);

void core_ds_CPU(double dt){

    double dt_half = dt/2e0;
    #ifdef magmom
    #pragma omp parallel for
    for(int i = 0; i < natom ; ++i) {
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        atom_ptr->s  = vec_divide(atom_ptr->m,-el_g);
        atom_ptr->s0 = vec_length(atom_ptr->s);
    }
    #endif

    //Suzuki-Trotter decompsition. Forward and backward.
    for (int i = 0 ; i < ngroups ; ++i){
        #pragma omp parallel for
        for (int j = 0 ; j < *(allocate_threads_ptr+i); ++j){
            struct atom_struct *atom_ptr;
            atom_ptr = (*(allocate_cell_ptr_ptr + i*max_no_of_members + j))->head_ptr;
            while(atom_ptr != NULL){
                calculate_spin(atom_ptr, dt_half);
                atom_ptr = atom_ptr->next_atom_ptr;
            }
        }
    }

    for (int i = ngroups - 1 ; i >=0 ; --i){
        #pragma omp parallel for
        for (int j = 0 ; j < *(allocate_threads_ptr+i); ++j){
            struct atom_struct *atom_ptr;
            atom_ptr = (*(allocate_cell_ptr_ptr + i*max_no_of_members + j))->tail_ptr;
            while(atom_ptr != NULL){
                calculate_spin(atom_ptr, dt_half);
                atom_ptr = atom_ptr->prev_atom_ptr;
            }
        }
    }
    
    #ifdef magmom
    #pragma omp parallel for
    for(int i = 0; i < natom ; ++i) {
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        atom_ptr->m  = vec_times(-el_g, atom_ptr->s);
        atom_ptr->m0 = vec_length(atom_ptr->m);
    }
    #endif
    
}

void core_ds(double dt){
    core_ds_CPU(dt);
}


vector spin_rotation(vector Heff, vector s, double dt){

    vector omega  = vec_divide(Heff, -hbar);
    double omega0 = vec_length(omega);

    if (omega0 > 0e0){
        omega = vec_divide(omega, omega0);
    } else {
        omega = vec_zero();
    }

    double omega_12 = omega.x*omega.y;
    double omega_23 = omega.y*omega.z;
    double omega_13 = omega.x*omega.z;

    double omega1_sq = omega.x*omega.x;
    double omega2_sq = omega.y*omega.y;
    double omega3_sq = omega.z*omega.z;

    double A = sin(omega0*dt);
    double B = 1e0 - cos(omega0*dt);

    vector s_temp;
    s_temp.x = s.x
             + (s.x*B*(-omega2_sq - omega3_sq)
             +  s.y*(B*omega_12 - A*omega.z)
             +  s.z*(A*omega.y + B*omega_13));
    s_temp.y = s.y
             + (s.y*B*(-omega1_sq - omega3_sq)
             +  s.z*(B*omega_23 - A*omega.x)
             +  s.x*(A*omega.z + B*omega_12));
    s_temp.z = s.z
             + (s.z*B*(-omega1_sq - omega2_sq)
             +  s.x*(B*omega_13 - A*omega.y)
             +  s.y*(A*omega.x + B*omega_23));

    return s_temp;
}
#endif
