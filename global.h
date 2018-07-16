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
*  Other internal variables 
*
********************************************************************************
*
*  Edit notes:
*  Date:    Apr 2016
*  Author:  Pui-Wai (Leo) MA
*  Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*  1) Added following for quantum thermostat.
*
*  EXTERN double *quantum_rand_memory_ptr;
*  EXTERN double *quantum_noise_ptr;
*  EXTERN int    *quantum_count_ptr;
*  EXTERN double *H_ptr;
*
*  and 
*
*  EXTERN double* quantum_rand_memory_ptr_d;
*  EXTERN double* quantum_noise_ptr_d;
*  EXTERN int*    quantum_count_ptr_d;
*  EXTERN double* H_ptr_d;
*
********************************************************************************/

EXTERN double virial INIT(0e0); //The total virial of the system

EXTERN double total_time; //the total physical time (not CPU time); in second
EXTERN double last_total_time_pressure;
EXTERN double last_total_time_stress;

#ifdef extfield
EXTERN vector Hext;
//external field; in Tesla; the magnetic moment is in opposite direction of atomic spin.
//You need to input it in "external_field.cpp" or "external_field_GPU.cu"
#endif

//potential.cpp
EXTERN int ninput INIT(1000000);                  // the number of discrete input points in the potential table
EXTERN double finput INIT(ninput);
EXTERN double rmax INIT(rcut_max);                 // the max of rij in potential table
EXTERN double rhomax INIT(1000e0);                    // the max of rho in potential table
EXTERN double finput_over_rmax;
EXTERN double finput_over_rhomax;
#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
EXTERN double *bf_ptr;         // the first pointer of input big_f
EXTERN double *dbf_ptr;        // the first pointer of input 1st derivative of big_f
EXTERN double *sf_ptr;         // the first pointer of input small_f
EXTERN double *dsf_ptr;        // the first pointer of input 1st derivative of small_f
EXTERN double *pr_ptr;         // the first pointer of input pair potential term
EXTERN double *dpr_ptr;        // the first pointer of input 1st derivative of pair potential term
#endif
#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
EXTERN double *Jij_ptr;        //the first pointer to pointer of input Jij
EXTERN double *dJij_ptr;       //the first pointer to pointer of input 1st derivative of Jij
#endif
#if defined SDHL || defined SLDHL
EXTERN double *LandauA_ptr;
EXTERN double *LandauB_ptr;
EXTERN double *LandauC_ptr;
EXTERN double *LandauD_ptr;
#endif
#if defined SLDHL
EXTERN double *dLandauA_ptr;
EXTERN double *dLandauB_ptr;
EXTERN double *dLandauC_ptr;
EXTERN double *dLandauD_ptr;
#endif

EXTERN double *sqrt_ptr;

#if defined SLDNC
EXTERN struct paraLandau para;
#endif

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
EXTERN vector ave_s;
EXTERN vector ave_m;
#endif

EXTERN int nperfect;
EXTERN int natom;
EXTERN vector box_length;
EXTERN vector box_length_half;
EXTERN box_vector d;
EXTERN box_vector Inv_d;
EXTERN double box_volume;
EXTERN double density;

EXTERN int ncells; //no. of link cell
EXTERN int no_of_link_cell_x;
EXTERN int no_of_link_cell_y;
EXTERN int no_of_link_cell_z;

//the first atom pointer
EXTERN struct atom_struct *first_atom_ptr;

//the first link cell pointer
EXTERN struct cell_struct *first_cell_ptr;

//the pointer to pointer of reallocate cells
EXTERN struct cell_struct **allocate_cell_ptr_ptr;

//the pointer for number of thread of a group
EXTERN int *allocate_threads_ptr;

EXTERN int ngroups; //Total no. of parallel group
EXTERN int max_no_of_members;

EXTERN double ave_stress11;
EXTERN double ave_stress22;
EXTERN double ave_stress33;
EXTERN double ave_stress12;
EXTERN double ave_stress23;
EXTERN double ave_stress31;

EXTERN double pressure0;

#ifdef eltemp
EXTERN double initial_ave_energy; //since the heat transfer equation cannot be solved analytically, we store the inital value of energy per atom, and to achieve energy conservation numerically. After every interval_of_print_out, the total energy would be rescaled according to this value.
#endif

//initial_random_number_CPU.cpp
EXTERN uint *m_w_ptr;
EXTERN uint *m_z_ptr;
EXTERN int *spare_rand_ptr;
EXTERN double *rand1_ptr;
EXTERN double *rand2_ptr;

//quantum_noise_CPU.cpp
EXTERN double *quantum_rand_memory_ptr;
EXTERN double *quantum_noise_ptr;
EXTERN int    *quantum_count_ptr;
EXTERN double *H_ptr;

#ifdef GPU
EXTERN int no_of_blocks;
EXTERN int no_of_MP;

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
EXTERN double *bf_ptr_d; // the first pointer of input big_f
EXTERN double *dbf_ptr_d; // the first pointer of input 1st derivative of big_f
EXTERN double *sf_ptr_d; // the first pointer of input small_f
EXTERN double *dsf_ptr_d; // the first pointer of input 1st derivative of small_f
EXTERN double *pr_ptr_d; // the first pointer of input pair potential term
EXTERN double *dpr_ptr_d; // the first pointer of input 1st derivative of pair potential term
#endif
#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
EXTERN double *Jij_ptr_d; //the first pointer of input Jij
EXTERN double *dJij_ptr_d; //the first pointer of input 1st derivative of Jij
#endif
#if defined SDHL || defined SLDHL
EXTERN double *LandauA_ptr_d;
EXTERN double *LandauB_ptr_d;
EXTERN double *LandauC_ptr_d;
EXTERN double *LandauD_ptr_d;
#endif
#if defined SLDHL
EXTERN double *dLandauA_ptr_d;
EXTERN double *dLandauB_ptr_d;
EXTERN double *dLandauC_ptr_d;
EXTERN double *dLandauD_ptr_d;
#endif

EXTERN struct atom_struct *first_atom_ptr_d;
EXTERN struct cell_struct *first_cell_ptr_d;

EXTERN struct cell_struct **allocate_cell_ptr_ptr_d;
EXTERN int *allocate_threads_ptr_d;
EXTERN int *max_no_of_members_ptr_d;

EXTERN int no_of_blocks_cell;
EXTERN int no_of_blocks_members;

EXTERN struct varGPU *var_ptr;
EXTERN struct varGPU *var_ptr_d;

//random_number_GPU.cu
EXTERN curandState *rand_state_ptr_d;

//quantum_noise_GPU.cu
EXTERN double* quantum_rand_memory_ptr_d;
EXTERN double* quantum_noise_ptr_d;
EXTERN int*    quantum_count_ptr_d;
EXTERN double* H_ptr_d;

#endif /*GPU*/

