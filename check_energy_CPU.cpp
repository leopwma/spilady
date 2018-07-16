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

#ifdef CPU

#include "spilady.h"

void check_energy_CPU(int current_step){

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    double ave_pe   = 0e0;
    double ave_ke   = 0e0;

    #pragma omp parallel for  reduction(+:ave_pe,ave_ke)
    for (int i = 0; i < natom; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        ave_pe += atom_ptr->pe;
        ave_ke += atom_ptr->ke;
    }
    ave_pe /= natom;
    ave_ke /= natom;
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    double ave_me  = 0e0;
    double ave_me0 = 0e0;

    #pragma omp parallel for  reduction(+:ave_me,ave_me0)
    for (int i = 0; i < natom; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        ave_me += atom_ptr->me;
        ave_me0 += atom_ptr->me0;
    }
    ave_me /= natom;
    ave_me0 /= natom;
    #endif

    #ifdef eltemp
    double ave_Ee = 0e0;

    #pragma omp parallel for reduction(+:ave_Ee)
    for (int i = 0; i < ncells; ++i){
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + i;
        cell_ptr->Ee = Te_to_Ee(cell_ptr->Te);
        ave_Ee += (double(natom)/double(ncells))*cell_ptr->Ee;
    }
    ave_Ee /= natom;

    #ifdef renormalizeEnergy
    double numerical_error_ave_energy = 0e0;
    if (current_step == -1){
        initial_ave_energy = 0e0;
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        initial_ave_energy += ave_pe;
        initial_ave_energy += ave_ke;
        #endif
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        initial_ave_energy += ave_me;
        initial_ave_energy += ave_me0;
        #endif
        initial_ave_energy += ave_Ee;
    } else {
        double current_ave_energy = 0e0;
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        current_ave_energy += ave_pe;
        current_ave_energy += ave_ke;
        #endif
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        current_ave_energy += ave_me;
        current_ave_energy += ave_me0;
        #endif
        current_ave_energy += ave_Ee;
        numerical_error_ave_energy = current_ave_energy - initial_ave_energy;
    }

    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i){
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + i;
        cell_ptr->Ee -= numerical_error_ave_energy;
        cell_ptr->Te = Ee_to_Te(cell_ptr->Ee);
    }
    #endif

    #endif /*eltemp*/

    char out_enr_front[] = "enr-";
    char out_enr[256];
    strcpy(out_enr,out_enr_front);
    strcat(out_enr,out_body);
    strcat(out_enr,".dat");

    ofstream out_file(out_enr,ios::app);
    out_file << setiosflags(ios::scientific) << setprecision(15);

    out_file << current_step             //1
             << " " << total_time        //2
             #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
             << " " << ave_pe            //3
             << " " << ave_ke            //4
             #endif
             #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
             << " " << ave_me            //5
             << " " << ave_me0           //6
             #endif
             #ifdef eltemp
             << " " << ave_Ee            //7
             #endif
             << '\n';
                
     out_file.close();
}

void check_energy(int current_step){
    check_energy_CPU(current_step);
}

#endif
