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

#if defined CPU

#include "spilady.h"

#if defined extfield

void external_field_CPU(int current_step){

    static bool infile_extfield = 0;

    if (current_step ==  -1){
        ifstream infile("extfield.in");

        if (infile) {
            cout << "Reading external field file!!!" << '\n';
            infile_extfield = 1;

            int temp;
            infile >> natom;
            for (int i = 0; i < natom; ++i){
                struct atom_struct* atom_ptr;
                atom_ptr = first_atom_ptr + i;
                infile >> temp >> atom_ptr->Hext.x >> atom_ptr->Hext.y >> atom_ptr->Hext.z;
                atom_ptr->Hext = vec_times(-el_g*muB, atom_ptr->Hext);  // converted from Tesla into eV

            }
        }
    }

    if (infile_extfield == 0){

        if (current_step == 0) cout <<  "User defined external field apply." << '\n';

        Hext.x = 0e0; //in Tesla
        Hext.y = 0e0; //in Tesla
        Hext.z = 0e0; //in Tesla

        //if (total_time < 1e-10) Hext.z = 0e0;
        //if (total_time >= 1e-10 && total_time < 4e-10) Hext.z = -20e0;
        //if (total_time >= 4e-10 && total_time < 7e-10) Hext.z = 0e0;
        //if (total_time >= 7e-10) Hext.z = -20e0;

        Hext = vec_times(-el_g*muB, Hext);  // converted from Tesla into eV

        #pragma omp parallel for
        for (int i = 0; i < natom; ++i) {
            struct atom_struct* atom_ptr;
            atom_ptr = first_atom_ptr + i;
            atom_ptr->Hext = Hext;
        }
    }
}

void external_field(int current_step){
    external_field_CPU(current_step);
}


#endif
#endif
