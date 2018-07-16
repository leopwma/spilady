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
#include "prototype_GPU.h"


void copy_CPU_to_GPU(){

    //copy all information of atoms from CPU to GPU
    cudaMalloc((void**)&first_atom_ptr_d, natom*sizeof(atom_struct));
    cudaMemcpy(first_atom_ptr_d, first_atom_ptr, natom*sizeof(atom_struct), cudaMemcpyHostToDevice);

    //copy all information of linkcells from CPU to GPU
    cudaMalloc((void**)&first_cell_ptr_d, ncells*sizeof(cell_struct));
    cudaMemcpy(first_cell_ptr_d, first_cell_ptr, ncells*sizeof(cell_struct), cudaMemcpyHostToDevice);

    //copy all necessary variables from CPU to GPU, using "struct varGPU".
    var_ptr = (varGPU*)malloc(sizeof(varGPU));
    cudaMalloc((void**)&var_ptr_d, sizeof(varGPU));

    var_ptr->ninput = ninput;
    var_ptr->finput = finput;
    var_ptr->rmax = rmax;
    var_ptr->rhomax = rhomax;
    var_ptr->finput_over_rmax = finput_over_rmax;
    var_ptr->finput_over_rhomax = finput_over_rhomax;

    var_ptr->nperfect = nperfect ;
    var_ptr->natom = natom;

    var_ptr->box_length = box_length;
    var_ptr->box_length_half = box_length_half;
    var_ptr->box_volume = box_volume;
    var_ptr->d = d;
    var_ptr->Inv_d = Inv_d;
    var_ptr->density = density;

    var_ptr->no_of_link_cell_x = no_of_link_cell_x;
    var_ptr->no_of_link_cell_y = no_of_link_cell_y;
    var_ptr->no_of_link_cell_z = no_of_link_cell_z;
    var_ptr->ncells = ncells;

    var_ptr->a_lattice = a_lattice;
    #ifdef hcp0001
    var_ptr->c_lattice = c_lattice;
    #endif
    var_ptr->no_of_unit_cell_x = no_of_unit_cell_x;
    var_ptr->no_of_unit_cell_y = no_of_unit_cell_y;
    var_ptr->no_of_unit_cell_z = no_of_unit_cell_z;
    var_ptr->unit_cell_no_of_atom = unit_cell_no_of_atom;
    var_ptr->unit_cell_edge_x = unit_cell_edge_x;
    var_ptr->unit_cell_edge_y = unit_cell_edge_y;
    var_ptr->unit_cell_edge_z = unit_cell_edge_z;

    var_ptr->atmass = atmass;
    var_ptr->temperature = temperature;

    #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined lattlang
    var_ptr->gamma_L_over_mass = gamma_L_over_mass;
    var_ptr->gamma_L  = gamma_L;
    #endif
    #if (defined SDH || defined SLDH) && defined spinlang
    var_ptr->gamma_S_H = gamma_S_H;
    #endif
    #if (defined SDHL || defined SLDHL) && defined spinlang
    var_ptr->gamma_S_HL = gamma_S_HL;
    #endif
  
    #if defined STRESS
    var_ptr->stress_xx = stress_xx;
    var_ptr->stress_yy = stress_yy;
    var_ptr->stress_zz = stress_zz;
    #endif
    #if defined PRESSURE
    var_ptr->pressure = pressure;
    #endif
    #if defined STRESS || defined PRESSURE
    var_ptr->baro_damping_time = baro_damping_time;
    #endif

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    var_ptr->rcut_pot = rcut_pot;
    var_ptr->rcut_pot_sq = rcut_pot_sq;
    #endif
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    var_ptr->rcut_mag = rcut_mag;
    var_ptr->rcut_mag_sq = rcut_mag_sq;
    #endif
    var_ptr->rcut_max = rcut_max;
    var_ptr->rcut_max_sq = rcut_max_sq;
    #ifdef localvol
    var_ptr->rcut_vol = rcut_vol;
    #endif
    var_ptr->min_length_link_cell = min_length_link_cell;

    #ifdef extfield
    var_ptr->Hext = Hext;
    #endif

    #ifdef changestep
    var_ptr->displace_limit = displace_limit;
    var_ptr->phi_limit = phi_limit;
    #endif
    
    #ifdef SLDNC
    var_ptr->para = para;
    #endif
    
    cudaMemcpy(var_ptr_d, var_ptr, sizeof(varGPU), cudaMemcpyHostToDevice);
}


void copy_atoms_from_GPU_to_CPU(){

     cudaMemcpy(first_atom_ptr, first_atom_ptr_d, natom*sizeof(atom_struct), cudaMemcpyDeviceToHost);

}

void copy_cells_from_GPU_to_CPU(){

     cudaMemcpy(first_cell_ptr, first_cell_ptr_d, ncells*sizeof(cell_struct), cudaMemcpyDeviceToHost);
}

void free_copy_CPU_to_GPU(){

     cudaFree(first_atom_ptr_d);
     cudaFree(first_cell_ptr_d);
     cudaFree(var_ptr_d);
     free(var_ptr);

}
#endif

