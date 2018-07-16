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

void check_pressure_CPU(int current_step){

    double sum_ke = 0e0;
    #pragma omp parallel for reduction(+:sum_ke)
    for (int i = 0; i < natom ; ++i) sum_ke += (first_atom_ptr+i)->ke;

    double tmp = 2e0/3e0*sum_ke/natom;
    pressure0 = density*(tmp-virial/(3e0*natom));
    pressure0 *=160.217653e0;

    char out_prs_front[] = "prs-";
    char out_prs[256];
    strcpy(out_prs,out_prs_front);
    strcat(out_prs,out_body);
    strcat(out_prs,".dat");

    ofstream out_file(out_prs,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step << " " << total_time
            << " " << d.xx << " " << d.yx << " " << d.yy
            << " " << d.zx << " " << d.zy << " " << d.zz
            << " " << density
            << " " << pressure0
            << '\n';

    out_file.close();

}

void check_pressure(int current_step){
    check_pressure_CPU(current_step);
}

#endif
