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

#ifdef GPU

#include "spilady.h"

#ifdef eltemp

#include "prototype_GPU.h"

/*****************************************************************************
* GPU prototypes
*****************************************************************************/

__global__ void LP1temp(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d
                        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                        , double *Jij_ptr_d
                        #endif
                        #if defined SDHL || defined SLDHL
                        , double *LandauA_ptr_d
                        , double *LandauB_ptr_d
                        , double *LandauC_ptr_d
                        , double *LandauD_ptr_d
                        #endif
                        );

/*****************************************************************************
* CPU codes
*****************************************************************************/

void calculate_temperature_GPU(){

    LP1temp<<<no_of_blocks_cell, no_of_threads>>>(var_ptr_d, first_cell_ptr_d
                                                  #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                                                  , Jij_ptr_d
                                                  #endif
                                                  #if defined SDHL || defined SLDHL
                                                  , LandauA_ptr_d
                                                  , LandauB_ptr_d
                                                  , LandauC_ptr_d
                                                  , LandauD_ptr_d
                                                  #endif
                                                  );
}

void calculate_temperature(){
    calculate_temperature_GPU();
}

/*****************************************************************************
* GPU codes
*****************************************************************************/

__global__ void LP1temp(struct varGPU *var_ptr_d, struct cell_struct *first_cell_ptr_d
                        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                        , double *Jij_ptr_d
                        #endif
                        #if defined SDHL || defined SLDHL
                        , double *LandauA_ptr_d
                        , double *LandauB_ptr_d
                        , double *LandauC_ptr_d
                        , double *LandauD_ptr_d
                        #endif
                        )
{

    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->ncells){

        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr_d + i;

        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        double sum_ke = 0e0;
          #ifdef localcolmot
          cell_ptr->ave_p = vec_zero_d();
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
              cell_ptr->ave_p = vec_add_d(cell_ptr->ave_p, atom_ptr->p);
              #endif
            #endif

            #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
              #ifdef extfield
              atom_ptr->Heff_H = atom_ptr->Hext;
              #else
              atom_ptr->Heff_H = vec_zero_d();
              #endif

              inner_spin_d(var_ptr_d, atom_ptr, first_cell_ptr_d, Jij_ptr_d);

              cell_ptr->sum_R_up += vec_sq_d(vec_cross_d(atom_ptr->s, atom_ptr->Heff_H));
              cell_ptr->sum_R_dn += vec_dot_d(atom_ptr->s,atom_ptr->Heff_H);
            #endif

            #if defined SDHL || defined SLDHL
              atom_ptr->Heff_L = vec_zero_d();

              #ifdef SLDHL
              double A = LandauA_d(atom_ptr->rho, LandauA_ptr_d, var_ptr_d);
              double B = LandauB_d(atom_ptr->rho, LandauB_ptr_d, var_ptr_d);
              double C = LandauC_d(atom_ptr->rho, LandauC_ptr_d, var_ptr_d);
              double D = LandauD_d(atom_ptr->rho, LandauD_ptr_d, var_ptr_d);
              #endif
              #ifdef SDHL
              double A = LandauA_d(1, LandauA_ptr_d, var_ptr_d);
              double B = LandauB_d(1, LandauB_ptr_d, var_ptr_d);
              double C = LandauC_d(1, LandauC_ptr_d, var_ptr_d);
              double D = LandauD_d(1, LandauD_ptr_d, var_ptr_d);
              #endif

              double s_sq = vec_sq_d(atom_ptr->s);
              atom_ptr->s0 = vec_length_d(atom_ptr->s);
              atom_ptr->Heff_L = vec_times_d(-(2e0*A + 4e0*B*s_sq + 6e0*C*pow(s_sq,2) + 8e0*D*pow(s_sq,3)), atom_ptr->s);
 
              #ifdef SLDHL
              atom_ptr->sum_Jij_sj = 0e0;
              inner_sum_Jij_sj_d(var_ptr_d, atom_ptr, first_cell_ptr_d, Jij_ptr_d);
              atom_ptr->Heff_HC = vec_times_d(-atom_ptr->sum_Jij_sj/atom_ptr->s0, atom_ptr->s);
              atom_ptr->Heff_L = vec_add_d(atom_ptr->Heff_L, atom_ptr->Heff_HC);
              #endif

              cell_ptr->sum_L_up += vec_sq_d(vec_add_d(atom_ptr->Heff_H, atom_ptr->Heff_L));
              cell_ptr->sum_L_dn += 6e0*A + 20e0*B*s_sq + 42e0*C*pow(s_sq,2) + 72e0*D*pow(s_sq,3);

              #ifdef SLDHL
              cell_ptr->sum_L_dn += 2e0*atom_ptr->sum_Jij_sj/atom_ptr->s0;
              #endif
            #endif

            atom_ptr = atom_ptr->next_atom_ptr;
        }
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
            #ifdef localcolmot
            cell_ptr->ave_p = vec_divide_d(cell_ptr->ave_p, cell_ptr->no_of_atoms_in_cell);
            sum_ke -= cell_ptr->no_of_atoms_in_cell*vec_sq_d(cell_ptr->ave_p)/2e0/var_ptr_d->atmass;
            cell_ptr->Tl = 2e0/3e0*sum_ke/(cell_ptr->no_of_atoms_in_cell - 1e0)/boltz;
            cell_ptr->Gel = 3e0*(cell_ptr->no_of_atoms_in_cell - 1e0)*boltz
                            *var_ptr_d->gamma_L/var_ptr_d->atmass
                            /(double(var_ptr_d->natom)/double(var_ptr_d->ncells));
            #else
            cell_ptr->Tl = sum_ke*2e0/3e0/cell_ptr->no_of_atoms_in_cell/boltz;
            cell_ptr->Gel = 3e0*cell_ptr->no_of_atoms_in_cell*boltz
                            *var_ptr_d->gamma_L/var_ptr_d->atmass
                            /(double(var_ptr_d->natom)/double(var_ptr_d->ncells));
            #endif
        #endif
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        cell_ptr->Ts_R = cell_ptr->sum_R_up/cell_ptr->sum_R_dn/2e0/boltz;
        #endif

        #if defined SDH || defined SLDH
        cell_ptr->Ges = 2e0*boltz*var_ptr_d->gamma_S_H*cell_ptr->sum_R_dn
                        /hbar/(double(var_ptr_d->natom)/double(var_ptr_d->ncells));
        #endif

        #if defined SDHL || defined SLDHL
        cell_ptr->Ts_L = cell_ptr->sum_L_up/cell_ptr->sum_L_dn/boltz;
        cell_ptr->Ges = boltz*var_ptr_d->gamma_S_HL*cell_ptr->sum_L_dn
                        /(double(var_ptr_d->natom)/double(var_ptr_d->ncells));
        #endif
    }
}


#endif

#endif
