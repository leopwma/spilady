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
*   Prototypes
*
********************************************************************************/

//initial.cpp
void initialize();

//read_variables.cpp
void read_variables();

//potential_CPU.cpp
void potential_table();
#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
double bigf(double rho);
double dbigf(double rho);
double smallf(double rij);
double dsmallf(double rij);
double pairij(double rij);
double dpair(double rij);
#endif
#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
double Jij(double rij);
double dJij(double rij);
#endif
#if defined SDHL || defined SLDHL
double LandauA(double rho);
double LandauB(double rho);
double LandauC(double rho);
double LandauD(double rho);
#endif
#if defined SLDHL
double dLandauA(double rho);
double dLandauB(double rho);
double dLandauC(double rho);
double dLandauD(double rho);
#endif
void free_potential_ptr();

//build_lattice.cpp
void build_lattice();

//read_conf.cpp
#ifdef readconf
void read_config();
#endif

//read_vsim.cpp
#ifdef readvsim
void read_vsim();
#endif

//bcc100.cpp
#ifdef bcc100
void bcc100bulk();
#endif

//bcc111.cpp
#ifdef bcc111
void bcc111bulk();
#endif

//fcc100.cpp
#ifdef fcc100
void fcc100bulk();
#endif

//hcp0001.cpp
#ifdef hcp0001
void hcp0001bulk();
#endif

#if defined bcc100 || defined bcc111 || defined fcc100 || defined hcp0001
void initial_element();
#endif

//periodic_CPU.cpp
void periodic(vector &r);

#if (defined MD ||  defined SLDH || defined SLDHL || defined SLDNC) && defined initmomentum
//initial_momentum.cpp
void initial_momentum();

//scale_temperature.cpp
void scale_temperature();
#endif

//initial_spin.cpp
#if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined initspin
void initial_spin();
#endif

//map_cells.cpp
void map_cells();

//links_CPU.cpp
void initial_links();
void links();

//allocate_CPU.cpp
void allocate_cells();
void free_allocate_memory();

//caculated_rho_CPU.cpp
#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
void embedded_rho(atom_struct *atom_ptr);
void calculate_rho();
#endif

//calculated_force_energy_CPU.cpp
void inner_loop(atom_struct *atom_ptr);
void calculate_force_energy();

//calculate_temperature_CPU.cpp
#ifdef eltemp
void calculate_temperature();
#endif

//find_image_CPU.cpp
void find_image(vector &rij);

//random_number_CPU.cpp
void initialize_random_number();
double normal_rand(int thread_index);
void free_random_number();

//quantum_noise_CPU.cpp
void initial_quantum_noise();
void free_quantum_noise();
double quantum_noise(int n, int thread_index);


//check.cpp
void check(int current_step);

//check_energy_CPU.cpp
void check_energy(int current_step);

//check_temperature_CPU.cpp
void check_temperature(int current_step);

#ifdef eltemp
//heatcapacity.cpp
double Ce(double Te);
double Ee_to_Te(double Ee);
double Te_to_Ee(double Te);
#endif

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
//check_stress_CPU.cpp
void check_stress(int current_step);

//check_pressure_CPU.cpp
void check_pressure(int current_step);
#endif

//check_spin.cpp
#if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
void check_spin(int current_step);
#endif

//write.cpp
void write(int current_step);

//write_config.cpp
void write_config(int current_step);

//write_vsim.cpp
#ifdef writevsim
void write_vsim(int current_step);
#endif

//core.cpp
void core(int current_step);

//external_field_CPU.cpp
#ifdef extfield
void external_field(int current_step);
#endif

//external_force_CPU.cpp
#ifdef extforce
void external_force(int current_step);
#endif

//core_dTe_CPU.cpp
#ifdef eltemp
void core_dTe(double dt);
#endif

//core_ds_CPU.cpp
#if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
vector spin_rotation(vector Heff, vector s, double dt);
void core_ds(double dt);

//calculate_spin_CPU.cpp
void calculate_spin(atom_struct *atom_ptr, double dt);
void inner_spin(atom_struct *atom_ptr);
#endif

#ifdef SLDHL
void inner_sum_Jij_sj(atom_struct *atom_ptr);
#endif

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
//core_dp_CPU.cpp
void core_dp(double dt);

//core_dr_CPU.cpp
void core_dr(double dt);
#endif

//scale.cpp
void scale(int current_step);

//scale_stress_CPU.cpp
#ifdef STRESS
void scale_stress();
#endif

//scale_pressure_CPU.cpp
#ifdef PRESSURE
void scale_pressure();
#endif

//scale_step_CPU.cpp
#ifdef changestep
void scale_step();
#endif

//free_memory.cpp
void free_memory();

//vec_utils_CPU.cpp
vector vec_add(vector a, vector b);
vector vec_sub(vector a, vector b);
vector vec_cross(vector a, vector b);
vector vec_times(double a, vector b);
vector vec_divide(vector a, double b);
double vec_dot(vector a, vector b);
double vec_sq(vector a);
double vec_length(vector a);
vector vec_zero();
vector vec_init(double x, double y, double z);
double vec_volume(vector a);
box_vector inverse_box_vector(box_vector d);

#ifdef GPU
//initial_GPU.cpp
void initial_GPU();

//potential_GPU.cu
void potential_table_GPU();
void free_potential_ptr_GPU();

//copy_CPU_btw_GPU.cu
void copy_CPU_to_GPU();
void copy_atoms_from_GPU_to_CPU();
void copy_cells_from_GPU_to_CPU();
void free_copy_CPU_to_GPU();

#endif

