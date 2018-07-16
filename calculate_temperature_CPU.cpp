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

#ifdef eltemp

void calculate_temperature_CPU(){

    //calculate Tl, Ts_R, Ts_l, Ges and  Gel
    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i){
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + i;

        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        double sum_ke = 0e0;
          #ifdef localcolmot
          cell_ptr->ave_p = vec_zero();
          #endif
        #endif
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        cell_ptr->sum_R_up = 0e0;
        cell_ptr->sum_R_dn = 0e0;
        #endif
        #if defined SDHL || defined SLDHL
        cell_ptr->sum_L_up  = 0e0;
        cell_ptr->sum_L_dn  = 0e0;
        #endif

        struct atom_struct *atom_ptr;
        atom_ptr = cell_ptr->head_ptr;

        while(atom_ptr != NULL){

            #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
              sum_ke += atom_ptr->ke;
              #ifdef localcolmot
              cell_ptr->ave_p = vec_add(cell_ptr->ave_p, atom_ptr->p);
              #endif
            #endif

            #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
              #ifdef extfield
              atom_ptr->Heff_H = atom_ptr->Hext;
              #else
              atom_ptr->Heff_H = vec_zero();
              #endif

              inner_spin(atom_ptr);
              
              cell_ptr->sum_R_up += vec_sq(vec_cross(atom_ptr->s, atom_ptr->Heff_H));
              cell_ptr->sum_R_dn += vec_dot(atom_ptr->s,atom_ptr->Heff_H);
            #endif
            
            #if defined SDHL || defined SLDHL
              atom_ptr->Heff_L = vec_zero();

              #ifdef SLDHL
              double A = LandauA(atom_ptr->rho);
              double B = LandauB(atom_ptr->rho);
              double C = LandauC(atom_ptr->rho);
              double D = LandauD(atom_ptr->rho);
              #endif
              #ifdef SDHL
              double A = LandauA(1);
              double B = LandauB(1);
              double C = LandauC(1);
              double D = LandauD(1);
              #endif

              double s_sq = vec_sq(atom_ptr->s);
              atom_ptr->s0 = vec_length(atom_ptr->s);
              atom_ptr->Heff_L = vec_times(-(2e0*A + 4e0*B*s_sq + 6e0*C*pow(s_sq,2) + 8e0*D*pow(s_sq,3)), atom_ptr->s);
              
              #ifdef SLDHL
              atom_ptr->sum_Jij_sj = 0e0;
              inner_sum_Jij_sj(atom_ptr);
              atom_ptr->Heff_HC = vec_times(-atom_ptr->sum_Jij_sj/atom_ptr->s0, atom_ptr->s);
              atom_ptr->Heff_L = vec_add(atom_ptr->Heff_L, atom_ptr->Heff_HC);
              #endif

              cell_ptr->sum_L_up += vec_sq(vec_add(atom_ptr->Heff_H, atom_ptr->Heff_L));
              cell_ptr->sum_L_dn += 6e0*A + 20e0*B*s_sq + 42e0*C*pow(s_sq,2) + 72e0*D*pow(s_sq,3);

              #ifdef SLDHL
              cell_ptr->sum_L_dn += 2e0*atom_ptr->sum_Jij_sj/atom_ptr->s0;
              #endif
            #endif

            atom_ptr = atom_ptr->next_atom_ptr;
        }
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
            #ifdef localcolmot
            cell_ptr->ave_p = vec_divide(cell_ptr->ave_p, cell_ptr->no_of_atoms_in_cell);
            sum_ke -= cell_ptr->no_of_atoms_in_cell*vec_sq(cell_ptr->ave_p)/2e0/atmass;
            cell_ptr->Tl = 2e0/3e0*sum_ke/(cell_ptr->no_of_atoms_in_cell - 1e0)/boltz;
            cell_ptr->Gel = 3e0*(cell_ptr->no_of_atoms_in_cell - 1e0)*boltz
                            *gamma_L/atmass/(double(natom)/double(ncells));
            #else
            cell_ptr->Tl = sum_ke*2e0/3e0/cell_ptr->no_of_atoms_in_cell/boltz;
            cell_ptr->Gel = 3e0*cell_ptr->no_of_atoms_in_cell*boltz*gamma_L
                            /atmass/(double(natom)/double(ncells));
            #endif
        #endif
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        cell_ptr->Ts_R = cell_ptr->sum_R_up/cell_ptr->sum_R_dn/2e0/boltz;
        #endif

        #if defined SDH || defined SLDH
        cell_ptr->Ges = 2e0*boltz*gamma_S_H*cell_ptr->sum_R_dn
                        /hbar/(double(natom)/double(ncells));
        #endif

        #if defined SDHL || defined SLDHL
        cell_ptr->Ts_L = cell_ptr->sum_L_up/cell_ptr->sum_L_dn/boltz;
        cell_ptr->Ges = boltz*gamma_S_HL*cell_ptr->sum_L_dn
                        /(double(natom)/double(ncells));
        #endif
    }
}

void calculate_temperature(){
    calculate_temperature_CPU();
}

#endif

#endif
