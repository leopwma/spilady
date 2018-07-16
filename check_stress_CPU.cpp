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

#if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) &&  defined CPU

#include "spilady.h"

void check_stress_CPU(int current_step){

    ave_stress11 = 0e0;
    ave_stress22 = 0e0;
    ave_stress33 = 0e0;
    ave_stress12 = 0e0;
    ave_stress23 = 0e0;
    ave_stress31 = 0e0;

    #pragma omp parallel for reduction(+:ave_stress11,ave_stress22,ave_stress33,ave_stress12,ave_stress23,ave_stress31)
    for (int i = 0; i < natom ; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        ave_stress11 += atom_ptr->stress11;
        ave_stress22 += atom_ptr->stress22;
        ave_stress33 += atom_ptr->stress33;
        ave_stress12 += atom_ptr->stress12;
        ave_stress23 += atom_ptr->stress23;
        ave_stress31 += atom_ptr->stress31;

    }
    
    ave_stress11 *= 160.217653e0/natom; //convert unit from eV/A^3 to GPa
    ave_stress22 *= 160.217653e0/natom;
    ave_stress33 *= 160.217653e0/natom;
    ave_stress12 *= 160.217653e0/natom;
    ave_stress23 *= 160.217653e0/natom;
    ave_stress31 *= 160.217653e0/natom;

    char out_str_front[] = "str-";
    char out_str[256];
    strcpy(out_str,out_str_front);
    strcat(out_str,out_body);
    strcat(out_str,".dat");

    ofstream out_file(out_str,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step << " " << total_time
             << " " << d.xx << " " << d.yx << " " << d.yy
             << " " << d.zx << " " << d.zy << " " << d.zz
             << " " << density
             << " " << (ave_stress11 + ave_stress22 + ave_stress33)/3e0
             << " " << ave_stress11
             << " " << ave_stress22
             << " " << ave_stress33
             << " " << ave_stress12
             << " " << ave_stress23
             << " " << ave_stress31
             << '\n';

    out_file.close();

}


void check_stress(int current_step){
    check_stress_CPU(current_step);
}

#endif
