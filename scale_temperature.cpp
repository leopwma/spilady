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

#if defined MD ||  defined SLDH || defined SLDHL || defined SLDNC

#include "spilady.h"

#if defined initmomentum

void scale_temperature(){

    //scale factor for total temperature
    double total_ke = 3e0/2e0*natom*initTl; // 3/2*NkT
    double sum_ke = 0e0;

    #ifdef OMP
    #pragma omp parallel for reduction(+:sum_ke)
    #endif
    for (int i = 0; i < natom ; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        sum_ke += vec_sq(atom_ptr->p)/2e0/atmass;
    }

    double factor = sqrt(total_ke/sum_ke);

    //apply the scaling to individual atom
    #ifdef OMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < natom; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        atom_ptr->p = vec_times(factor, atom_ptr->p);
        atom_ptr->ke = vec_sq(atom_ptr->p)/2e0/atmass;
    }      
}

#endif
#endif
