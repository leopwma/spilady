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

#if defined changestep

void scale_step_CPU(){

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    double displace_max = 0e0;
    double displace_max_temp[OMP_threads];

    for (int i = 0; i < OMP_threads; ++i) displace_max_temp[i] = 0e0;

    #pragma omp parallel for
    for (int i = 0; i < natom ; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        double p_sq = vec_sq(atom_ptr->p);
        
        if (p_sq > displace_max_temp[omp_get_thread_num()])  displace_max_temp[omp_get_thread_num()] = p_sq;
    }
    for (int i = 0; i < OMP_threads; ++i){
        if (displace_max_temp[i] > displace_max )  displace_max = displace_max_temp[i];
    }
    displace_max = sqrt(displace_max)/atmass*step;
    #endif
      
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    double phi_max = 0e0;
    double phi_max_temp[OMP_threads];

    for (int i = 0; i < OMP_threads; ++i) phi_max_temp[i] = 0e0;

    #pragma omp parallel for
    for (int i = 0; i < natom ; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        double omega_sq = vec_sq(atom_ptr->Heff_H);

        if (omega_sq > phi_max_temp[omp_get_thread_num()])  phi_max_temp[omp_get_thread_num()] = omega_sq;
    }
    for (int i = 0; i < OMP_threads; ++i){
        if (phi_max_temp[i] > phi_max )  phi_max = phi_max_temp[i];
    }
    phi_max = sqrt(phi_max)/hbar*step;
    #endif

    cout
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    << "displace_max = " << displace_max << "(Angstrom)"
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    << " phi_max = " << phi_max << "(rad.)"
    #endif
    <<'\n';

    int switch_lattice = 0;
    int switch_spin = 0;

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    if (displace_max > displace_limit) switch_lattice = 1;
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
    if (phi_max > phi_limit) switch_spin = 1;
    #endif
    if (switch_spin + switch_lattice > 0){
        step *= 0.80;
    } else {
        step *= 1.05;
    }
}

void scale_step(){
    scale_step_CPU();
}

#endif

#endif
