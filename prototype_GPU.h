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
*  Prototypes for GPU device codes
*
********************************************************************************
*
*   Edit notes:
*   Date:    Apr 2016
*   Author:  Pui-Wai (Leo) MA
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) Edited the __device__ quantum_noise_d();
*
********************************************************************************/



#ifdef GPU

//potential_GPU.cu
#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
__device__ double bigf_d(double rho, double *bf_ptr_d, struct varGPU *var_ptr_d);
__device__ double dbigf_d(double rho, double *dbf_ptr_d, struct varGPU *var_ptr_d);
__device__ double smallf_d(double rij, double *sf_ptr_d, struct varGPU *var_ptr_d);
__device__ double dsmallf_d(double rij, double *dsf_ptr_d, struct varGPU *var_ptr_d);
__device__ double pair_d(double rij, double *pr_ptr_d, struct varGPU *var_ptr_d);
__device__ double dpair_d(double rij, double *dpr_ptr_d, struct varGPU *var_ptr_d);
#endif
#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
__device__ double Jij_d(double rij, double *Jij_ptr_d, struct varGPU *var_ptr_d);
__device__ double dJij_d(double rij, double *dJij_ptr_d, struct varGPU *var_ptr_d);
#endif
#if defined SDHL || defined SLDHL
__device__ double LandauA_d(double rho, double *LandauA_ptr_d, struct varGPU *var_ptr_d);
__device__ double LandauB_d(double rho, double *LandauB_ptr_d, struct varGPU *var_ptr_d);
__device__ double LandauC_d(double rho, double *LandauC_ptr_d, struct varGPU *var_ptr_d);
__device__ double LandauD_d(double rho, double *LandauD_ptr_d, struct varGPU *var_ptr_d);
#endif
#ifdef SLDHL
__device__ double dLandauA_d(double rho, double *dLandauA_ptr_d, struct varGPU *var_ptr_d);
__device__ double dLandauB_d(double rho, double *dLandauB_ptr_d, struct varGPU *var_ptr_d);
__device__ double dLandauC_d(double rho, double *dLandauC_ptr_d, struct varGPU *var_ptr_d);
__device__ double dLandauD_d(double rho, double *dLandauD_ptr_d, struct varGPU *var_ptr_d);
#endif

//periodic_GPU.cu
__device__ void periodic_d(vector &r, struct varGPU *var_ptr_d);

//find_image_GPU.cu
__device__ void find_image_d(vector &rij, struct varGPU *var_ptr_d);

//calculate_rho_GPU.cu
#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
__device__ void embedded_rho_d(struct varGPU *var_ptr_d, struct atom_struct *atom_ptr,
                               struct cell_struct *first_cell_ptr_d, double *sf_ptr_d);
#endif


//calculate_force_energy_GPU.cu
__device__ void inner_loop_d(struct varGPU *var_ptr_d, struct atom_struct *atom_ptr, struct cell_struct *first_cell_ptr_d
                      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                      , double *bf_ptr_d, double *sf_ptr_d, double *pr_ptr_d
                      , double *dbf_ptr_d, double *dsf_ptr_d, double *dpr_ptr_d
                      #endif
                      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                      ,double *Jij_ptr_d, double *dJij_ptr_d
                      #endif
                      #if defined SDHL || defined SLDHL
                      ,double *LandauA_ptr_d, double *LandauB_ptr_d, double *LandauC_ptr_d, double *LandauD_ptr_d
                      #endif
                      #if defined SLDHL
                      ,double *dLandauA_ptr_d, double *dLandauB_ptr_d, double *dLandauC_ptr_d, double *dLandauD_ptr_d
                      #endif
                      );


#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
//core_dS_GPU.cu
__device__ vector spin_rotation_d(vector Heff, vector s, double dt);

//calculate_spin_GPU.cu
__device__ void calculate_spin_d(int j, curandState *rand_state_ptr_d,
                          struct varGPU *var_ptr_d,
                          struct atom_struct *atom_ptr,
                          struct cell_struct *first_cell_ptr_d,
                          double dt,
                          double *Jij_ptr_d
                          #if defined SDHL || defined SLDHL
                          , double *LandauA_ptr_d
                          , double *LandauB_ptr_d
                          , double *LandauC_ptr_d
                          , double *LandauD_ptr_d
                          #endif
                          );

__device__ void inner_spin_d(struct varGPU *var_ptr_d,
                             struct atom_struct *atom_ptr,
                             struct cell_struct *first_cell_ptr_d,
                             double *Jij_ptr_d);
#endif

#ifdef SLDHL
__device__ void inner_sum_Jij_sj_d(struct varGPU *var_ptr_d,
                             struct atom_struct *atom_ptr,
                             struct cell_struct *first_cell_ptr_d,
                             double *Jij_ptr_d);
#endif

#ifdef eltemp
//heatcapacity_GPU.cu
__device__ double Ce_d(double Te);
__device__ double Te_to_Ee_d(double Te);
__device__ double Ee_to_Te_d(double Ee);
#endif

//vec_utils_GPU.cu
__device__ vector vec_add_d(vector a, vector b);
__device__ vector vec_sub_d(vector a, vector b);
__device__ vector vec_cross_d(vector a, vector b);
__device__ vector vec_times_d(double a, vector b);
__device__ vector vec_divide_d(vector a, double b);
__device__ double vec_dot_d(vector a, vector b);
__device__ double vec_sq_d(vector a);
__device__ double vec_length_d(vector a);
__device__ vector vec_zero_d();
__device__ vector vec_init_d(double x, double y, double z);
__device__ double vec_volume_d(vector a);
__device__ box_vector inverse_box_vector_d(box_vector d);

//random_number_GPU.cu
__device__ double normal_rand_d(curandState *rand_state_ptr_d); 

//quantum_noise_GPU.cu
#ifdef quantumnoise
__device__ double quantum_noise_d(int n, int m,  double* quantum_rand_memory_ptr_d,
                                  curandState *rand_state_ptr_d, double* H_ptr_d,
                                  double* quantum_noise_ptr_d, int* quantum_count_ptr_d,
                                  int Msteps_quantum, int Nfrequency_quantum);
#endif


#endif /*GPU*/
