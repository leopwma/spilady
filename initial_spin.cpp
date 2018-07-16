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

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC

#include "spilady.h"

#if defined initspin

void initial_spin(){

    int mseed = 25;
    srand(mseed);

    int ncase = 1;

    if (ncase == 1){
    //ferromagnetic
        for (int i = 0 ; i < natom ; ++i){
            struct atom_struct *atom_ptr;
            atom_ptr = first_atom_ptr + i ;

            atom_ptr->m0 = mag_mom;
            atom_ptr->m = vec_init(0e0, 0e0, -atom_ptr->m0);
              
            atom_ptr->s = vec_divide(atom_ptr->m,-el_g);
            atom_ptr->s0 = vec_length(atom_ptr->s);
        }
    }

    if (ncase == 2){
    //anti-ferromagnetic
        for (int i = 0 ; i < natom ; ++i){
            struct atom_struct *atom_ptr;
            atom_ptr = first_atom_ptr + i ;

            atom_ptr->m0 = mag_mom;
            if (i%2 == 0){ 
                atom_ptr->m = vec_init(0e0, 0e0, -atom_ptr->m0);
            } else {
                atom_ptr->m = vec_init(0e0, 0e0,  atom_ptr->m0);
            }

            atom_ptr->s = vec_divide(atom_ptr->m,-el_g);
            atom_ptr->s0 = vec_length(atom_ptr->s);
        }

    }

}
#endif
#endif
