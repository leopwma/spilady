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

void check_spin_CPU(int current_step){

    ave_s = vec_zero();
    double ave_sx = 0e0;
    double ave_sy = 0e0;
    double ave_sz = 0e0;
    ave_m = vec_zero();

    #pragma omp parallel for reduction(+:ave_sx, ave_sy, ave_sz)
    for (int i = 0; i < natom; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        ave_sx += atom_ptr->s.x;
        ave_sy += atom_ptr->s.y;
        ave_sz += atom_ptr->s.z;
    }
    
    ave_s.x = ave_sx;
    ave_s.y = ave_sy;
    ave_s.z = ave_sz;

    ave_s = vec_divide(ave_s, natom);
    double ave_s0 = vec_length(ave_s);

    ave_m = vec_times(-el_g, ave_s);
    double ave_m0 = vec_length(ave_m);

    char out_spn_front[] = "spn-";
    char out_spn[256];
    strcpy(out_spn,out_spn_front);
    strcat(out_spn,out_body);
    strcat(out_spn,".dat");

    ofstream out_file(out_spn,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step
             << " " << total_time
             << " " << ave_s.x
             << " " << ave_s.y
             << " " << ave_s.z
             << " " << ave_s0
             << " " << ave_m.x
             << " " << ave_m.y
             << " " << ave_m.z
             << " " << ave_m0
             << '\n';
    out_file.close();
}


void check_spin(int current_step){
    check_spin_CPU(current_step);
}

#endif
