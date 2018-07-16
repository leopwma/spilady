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
********************************************************************************
*
*   Edit notes:
*   Date:    Apr 2016
*   Author:  Pui-Wai (Leo) MA
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) variable "kappa_e" is added. Previously, it was a constant.
*
********************************************************************************/

#if defined CPU

#include "spilady.h"

#if defined eltemp

void core_dTe_CPU_A(double dt);
void core_dTe_CPU_B(double dt);
void core_dTe_CPU_C(double dt);

void core_dTe_CPU(double dt){

    int nloop = 50;
    double dt_over_nloop = dt/nloop;
    
    // C dTe/dt = Kappa Lapacian Te + Ges (Ts-Te) + Gel (Tl - Te)
    for(int i = 0; i < nloop; ++i){
        core_dTe_CPU_A(dt_over_nloop/2e0);   //C dTe/dt = Ges Ts + Gel Tl
        core_dTe_CPU_B(dt_over_nloop/2e0);   //C dTe/dt = -(Ges + Gel) Te
        core_dTe_CPU_C(dt_over_nloop);       //C dTe/dt = Kappa Lapacian Te
        core_dTe_CPU_B(dt_over_nloop/2e0);   //C dTe/dt = -(Ges + Gel) Te
        core_dTe_CPU_A(dt_over_nloop/2e0);   //C dTe/dt = Ges Ts + Gel Tl
    }
}

void core_dTe(double dt){
    core_dTe_CPU(dt);
}

void core_dTe_CPU_A(double dt){

    //change of electron energy and temperature due to lattice and spin
    //Solve: dEe/dt = C dTe/dt = Ges Ts + Gel Tl
    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i){
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + i;
        cell_ptr->Ee = Te_to_Ee(cell_ptr->Te);

        double delta_Ee = 0e0;
        #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined lattlang
          delta_Ee += cell_ptr->Gel*cell_ptr->Tl;
        #endif

        #ifdef spinlang
          #if defined SDH || defined SLDH
            delta_Ee += cell_ptr->Ges*cell_ptr->Ts_R;
          #endif
          #if defined SDHL || defined SLDHL
            delta_Ee += cell_ptr->Ges*cell_ptr->Ts_L;
          #endif
        #endif

        cell_ptr->Ee += dt*delta_Ee;
        cell_ptr->Te = Ee_to_Te(cell_ptr->Ee);
    }
}

void core_dTe_CPU_B(double dt){

    //Solve: dEe/dt = Ce*dTe/dt = -(Ges + Gel) Te
    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i){
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + i;

        double Ges_plus_Gel_dt_over_boltz = 0e0;
        #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined lattlang
        Ges_plus_Gel_dt_over_boltz += cell_ptr->Gel;
        #endif
        #if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined spinlang 
        Ges_plus_Gel_dt_over_boltz += cell_ptr->Ges;
        #endif
        Ges_plus_Gel_dt_over_boltz *= dt/boltz;

        //RK2
        //double Te_temp = cell_ptr->Te;
        //double Te_half = Te_temp*exp(-0.5e0*Ges_plus_Gel_dt_over_boltz/Ce(Te_temp));
        //cell_ptr->Te = Te_temp*exp(-Ges_plus_Gel_dt_over_boltz/Ce(Te_half));
        
        //RK4
        double T0 = cell_ptr->Te;
        double k1 = T0*(exp(-Ges_plus_Gel_dt_over_boltz/Ce(T0)) - 1e0)/dt;
        double T1 = T0 + k1*dt/2e0;
        double k2 = T1*(exp(-Ges_plus_Gel_dt_over_boltz/Ce(T1)) - 1e0)/dt;
        double T2 = T0 + k2*dt/2e0;
        double k3 = T2*(exp(-Ges_plus_Gel_dt_over_boltz/Ce(T2)) - 1e0)/dt;
        double T3 = T0 + k3*dt;
        double k4 = T3*(exp(-Ges_plus_Gel_dt_over_boltz/Ce(T3)) - 1e0)/dt;
        cell_ptr->Te += dt/6e0*(k1 + 2e0*k2 + 2e0*k3 + k4);

    }
}

void core_dTe_CPU_C(double dt){

    double start_total_Ee = 0e0;
    #pragma omp parallel for reduction(+:start_total_Ee)
    for (int i = 0; i < ncells; ++i){
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + i;
        cell_ptr->Ee = Te_to_Ee(cell_ptr->Te);
        start_total_Ee += cell_ptr->Ee;
    }

    double thermal_conductivity = kappa_e/eVtoJ/1e10;    //J s^-1 m^-1 K ^-1
    double dx_sq = pow(box_length.x/no_of_link_cell_x,2);
    double dy_sq = pow(box_length.y/no_of_link_cell_y,2);
    double dz_sq = pow(box_length.z/no_of_link_cell_z,2);
    double atomic_volume = box_volume/double(natom);
    double volume_factor = atomic_volume/(double(natom)/double(ncells));

    double prefactor = dt/2e0*thermal_conductivity*volume_factor/boltz;

    for (int i = 0 ; i < ngroups ; ++i){
        #pragma omp parallel for
        for (int j = 0 ; j < *(allocate_threads_ptr+i); ++j){
            struct cell_struct *cell_ptr;
            cell_ptr = *(allocate_cell_ptr_ptr + i*max_no_of_members + j);

            cell_ptr->Ee = Te_to_Ee(cell_ptr->Te);

            cell_ptr->Ee += prefactor
                        *( ((first_cell_ptr+(cell_ptr->neigh_cell[0]))->Te
                           +(first_cell_ptr+(cell_ptr->neigh_cell[13]))->Te
                           -2e0*cell_ptr->Te)/dx_sq
                          +((first_cell_ptr+(cell_ptr->neigh_cell[2]))->Te
                           +(first_cell_ptr+(cell_ptr->neigh_cell[14]))->Te
                           -2e0*cell_ptr->Te)/dy_sq
                          +((first_cell_ptr+(cell_ptr->neigh_cell[12]))->Te
                           +(first_cell_ptr+(cell_ptr->neigh_cell[15]))->Te
                           -2e0*cell_ptr->Te)/dz_sq);

            cell_ptr->Te = Ee_to_Te(cell_ptr->Ee);
        }
    }

    for (int i = ngroups - 1 ; i >=0 ; --i){
        #pragma omp parallel for
        for (int j = 0 ; j < *(allocate_threads_ptr+i); ++j){
            struct cell_struct *cell_ptr;
            cell_ptr = *(allocate_cell_ptr_ptr + i*max_no_of_members + j);

            cell_ptr->Ee = Te_to_Ee(cell_ptr->Te);

            cell_ptr->Ee += prefactor
                        *( ((first_cell_ptr+(cell_ptr->neigh_cell[0]))->Te
                           +(first_cell_ptr+(cell_ptr->neigh_cell[13]))->Te
                           -2e0*cell_ptr->Te)/dx_sq
                          +((first_cell_ptr+(cell_ptr->neigh_cell[2]))->Te
                           +(first_cell_ptr+(cell_ptr->neigh_cell[14]))->Te
                           -2e0*cell_ptr->Te)/dy_sq
                          +((first_cell_ptr+(cell_ptr->neigh_cell[12]))->Te
                           +(first_cell_ptr+(cell_ptr->neigh_cell[15]))->Te
                           -2e0*cell_ptr->Te)/dz_sq);
        

            cell_ptr->Te = Ee_to_Te(cell_ptr->Ee);
        }
    }

    double end_total_Ee = 0e0;

    #pragma omp parallel for reduction(+:end_total_Ee)
    for (int i = 0; i < ncells; ++i){
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + i;
        cell_ptr->Ee = Te_to_Ee(cell_ptr->Te);
        end_total_Ee += cell_ptr->Ee;
    }

    double renormalize_factor = start_total_Ee/end_total_Ee;

    #pragma omp parallel for
    for (int i = 0; i < ncells; ++i){
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + i;
        cell_ptr->Ee *= renormalize_factor;
        cell_ptr->Te = Ee_to_Te(cell_ptr->Ee);
    }
}

#endif
#endif
