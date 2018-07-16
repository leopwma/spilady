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
*   Declare and initialize input variables.
*   Default values are in the bracket of INIT().
*
********************************************************************************
*
*   Edit notes:
*   Date:    Apr 2016
*   Author:  Pui-Wai (Leo) MA
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) variable "kappa_e" is added.
*   2) variable "Msteps_quantum" is added.
*   3) variable "Nfrequency_quantum" is added.
*
********************************************************************************/

#ifdef GPU
EXTERN int current_device INIT(0); //index of the Nvidia GPU device being used
EXTERN int no_of_threads INIT(32); //set the number of threads in a block
#endif

#ifdef OMP
EXTERN int OMP_threads INIT(2); //number of threads for running openmp
#endif

EXTERN char* out_body; //the body of output file names

#ifdef runstep  //either run for a total number of steps or a total time.
EXTERN int no_of_production_steps INIT(1000);
#else
EXTERN double total_production_time INIT(1e-13); // in second
#endif

EXTERN double step INIT(1e-15); // time-step ; in second

EXTERN int interval_of_print_out INIT(1); // print out general information per interval

EXTERN int interval_of_config_out INIT(1000); //output the full configuration file per interval during production steps

#ifdef writevsim
EXTERN int interval_of_vsim INIT(1000);
EXTERN int vsim_prec INIT(4);
#endif

#ifdef readconf
EXTERN char* in_config;  //input configuration file
#endif

#ifdef readvsim
EXTERN char* in_vsim_atom; //input V_sim atomic positions file
#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
EXTERN char* in_vsim_spin; //input V_sim spin vectors file
#endif
#endif

#if defined readTe
EXTERN char* in_eltemp; //input the electron temperatures of link cells
#endif

#ifdef eltemp
EXTERN double kappa_e INIT(80e0);
#endif

#ifdef inittime
EXTERN double start_time INIT(0e0); // To initialize total_time = start_time
#endif

#ifdef bcc100
EXTERN double a_lattice INIT(2.83e0); //a lattice constant in Angstrom, Fe
EXTERN int no_of_unit_cell_x INIT(10); //no. of unit cell in a x side
EXTERN int no_of_unit_cell_y INIT(10); //no. of unit cell in a y side
EXTERN int no_of_unit_cell_z INIT(10); //no. of unit cell in a z side
EXTERN int unit_cell_no_of_atom INIT(2); //no. of atom in an unit call
EXTERN double unit_cell_edge_x INIT(a_lattice); //x dimension of a single cell
EXTERN double unit_cell_edge_y INIT(a_lattice); //y dimension of a single cell
EXTERN double unit_cell_edge_z INIT(a_lattice); //z dimension of a single cell
#endif

#ifdef bcc111
EXTERN double a_lattice INIT(2.83e0); //a lattice constant in Angstrom, Fe
EXTERN int no_of_unit_cell_x INIT(6); //no. of unit cell in a x side
EXTERN int no_of_unit_cell_y INIT(4); //no. of unit cell in a y side
EXTERN int no_of_unit_cell_z INIT(7); //no. of unit cell in a z side
EXTERN int unit_cell_no_of_atom INIT(12); //no. of atom in an unit call
EXTERN double unit_cell_edge_x INIT(a_lattice*sqrt(3e0)); //x dimension of a single cell
EXTERN double unit_cell_edge_y INIT(a_lattice*sqrt(6e0)); //y dimension of a single cell
EXTERN double unit_cell_edge_z INIT(a_lattice*sqrt(2e0)); //z dimension of a single cell
#endif

#ifdef fcc100
EXTERN double a_lattice INIT(3.5e0); //a lattice constant in Angstrom, Fe
EXTERN int no_of_unit_cell_x INIT(10); //no. of unit cell in a x side
EXTERN int no_of_unit_cell_y INIT(10); //no. of unit cell in a y side
EXTERN int no_of_unit_cell_z INIT(10); //no. of unit cell in a z side
EXTERN int unit_cell_no_of_atom INIT(4); //no. of atom in an unit call
EXTERN double unit_cell_edge_x INIT(a_lattice); //x dimension of a single cell
EXTERN double unit_cell_edge_y INIT(a_lattice); //y dimension of a single cell
EXTERN double unit_cell_edge_z INIT(a_lattice); //z dimension of a single cell
#endif

#ifdef hcp0001
EXTERN double a_lattice INIT(3.629e0); //a lattice constant in Angstrom, Gd
EXTERN double c_lattice INIT(5.796e0); //c lattice constant in Angstrom, Gd
EXTERN int no_of_unit_cell_x INIT(10); //no. of unit cell in a x side
EXTERN int no_of_unit_cell_y INIT(10); //no. of unit cell in a y side
EXTERN int no_of_unit_cell_z INIT(10); //no. of unit cell in a z side
EXTERN int unit_cell_no_of_atom INIT(2); //no. of atom in an unit call
EXTERN double unit_cell_edge_x INIT(a_lattice); //x dimension of a single cell
EXTERN double unit_cell_edge_y INIT(a_lattice*sqrt(3e0)/2e0); //y dimension of a single cell
EXTERN double unit_cell_edge_z INIT(c_lattice); //z dimension of a single cell
#endif

#if defined bcc100 || defined bcc111 || defined fcc100 || defined hcp0001
EXTERN char element[2];
#endif

#if defined readconf || defined readvsim
EXTERN double a_lattice;
EXTERN int no_of_unit_cell_x;
EXTERN int no_of_unit_cell_y;
EXTERN int no_of_unit_cell_z;
EXTERN int unit_cell_no_of_atom;
EXTERN double unit_cell_edge_x;
EXTERN double unit_cell_edge_y;
EXTERN double unit_cell_edge_z;
#endif

EXTERN double atmass INIT(55.847e0*utoeV*length_scale*length_scale); // atomic mass in eV/((angstrom /s)^2); input unit is in u; mass of iron (Fe)
EXTERN double temperature INIT(300e0*boltz); //internal energy unit is of 1eV
EXTERN double initTl INIT(-1e0);
EXTERN int random_seed INIT(1234);

#if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined initspin
EXTERN double mag_mom INIT(2.2);
#endif

#ifdef lattlang
//EXTERN double gamma_L_over_mass INIT(1e13);//in s^-1, usually in the range of per picosecond.
EXTERN double gamma_L_over_mass INIT(6e11);//in s^-1, usually in the range of per picosecond.
EXTERN double gamma_L INIT(gamma_L_over_mass*atmass);
//Parameters may be found in J. Phys.:Condens. Matter 6 (1994) 6733-6750; table 4.
#endif

#ifdef spinlang
EXTERN double gamma_S_H INIT(8e-3); //spin langevin thermostat for Heisenberg Hamiltonian; unitless
//EXTERN double gamma_S_HL INIT(8e-3/hbar); //spin langevin thermostat for Heisenberg-Landau Hamiltonian; in hbar^-1
EXTERN double gamma_S_HL INIT(5.88e13); //spin langevin thermostat for Heisenberg-Landau Hamiltonian; in hbar^-1
#endif

//for Berendsen barostat
#ifdef STRESS
EXTERN int interval_of_scale_stress INIT(1); //scale according to stresses
EXTERN double stress_xx INIT(0e0); //in GPa; decided stresses in x,y,z direction
EXTERN double stress_yx INIT(0e0); //in GPa; decided stresses in x,y,z direction
EXTERN double stress_yy INIT(0e0); //in GPa
EXTERN double stress_zx INIT(0e0); //in GPa
EXTERN double stress_zy INIT(0e0); //in GPa
EXTERN double stress_zz INIT(0e0); //in GPa
#endif
#ifdef PRESSURE
EXTERN int interval_of_scale_pressure INIT(1); //scale according to pressure
EXTERN double pressure INIT(0e0); //in GPa
#endif
#if defined STRESS || defined PRESSURE
EXTERN double baro_damping_time INIT(1e-13); //in second;
#endif

#if defined MD || defined SLDH || defined SLDHL || defined SLDNC
EXTERN double rcut_pot INIT(4.1e0); // the cutoff for potential; in Angstrom
EXTERN double rcut_pot_sq;
#endif

#if defined SDH || defined SDHL || defined SLDH || defined SLDHL
EXTERN double rcut_mag INIT(3.75e0); // the cutoff for Jij; in Angstrom
EXTERN double rcut_mag_sq;
#endif

EXTERN double rcut_max INIT(4.1e0); //input the max. of rcut_mag and rcut_pot, please put it by hand here.
EXTERN double rcut_max_sq; //input the max. of rcut_mag and rcut_pot, please put it by hand here.

#ifdef localvol
EXTERN double rcut_vol INIT(a_lattice*1.2e0); // the cutoff for calculating local volume; in Angstrom
EXTERN double rcut_vol_sq;
#endif

EXTERN double min_length_link_cell INIT(rcut_max + 0.01e0); //the minimum length of the edge of a link cell
//The min_length_link_cell should always be larger or equal to rcut_max. In the case of treating a contracting system, the user need to ensure min_length_link_cell is alwasy larger than rcut_max throughout the whole simulation.

#ifdef changestep
//if time step is changing
EXTERN double displace_limit INIT(0.01e0); //in Angstrom; the maximum displacment distance in each time-step
EXTERN double phi_limit INIT(2e0*Pi_num/10e0); //in Radian; the maximum change of angle in each time-step
#endif

#ifdef quantumnoise
EXTERN int Msteps_quantum INIT(50); // the quantum noise does not change for M steps
EXTERN int Nfrequency_quantum INIT(150); // number of intervals between 0 and maximum frequency
EXTERN int Nfrequency_quantum_2 INIT(Nfrequency_quantum*2); // number of intervals between 0 and maximum frequency
#endif
