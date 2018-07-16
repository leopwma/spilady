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

struct vector {
    double x;
    double y;
    double z;
};

struct box_vector {
    double xx;
    double yx;
    double yy;
    double zx;
    double zy;
    double zz;
};

struct atom_struct {
    //all the properties of an atom
    vector r; //position in Cartesian coordinate

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
      vector p; //momentum
      vector f; //force

      #ifdef extforce
      vector fext; // external force
      #endif

      double vir; //the virial of individual atom

      double stress11; //components of atomic stresses
      double stress22;
      double stress33;
      double stress12;
      double stress23;
      double stress31;

      double rho; //the effective electron density for embedded atom method potential

      double ke; // Kinectic energy
      double pe; // Potential energy
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNCa
      vector s; //atomic spin vector
      double s0; // magnitude of spin vector
      vector Heff_H; //the Heisenberg part of the effective field

      vector m; //magnetic moment vector, use magnetic moment as input, instead of atomic spin
      double m0; //magnitdue of magnetic moment vector
      
      #if defined SDHL || SLDHL
      vector Heff_L; //the Landau part of the effective field
      #endif

      #ifdef extfield
      vector Hext; //external field, in unit of eV
      #endif

      #ifdef SLDHL
      vector Heff_HC; //the correction term to the Heisenberg part of the effective field.
      double sum_Jij_sj;  // for the correction term
      #endif

      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
      double me;  //magneitc energy (should be in negative sign)
      double me0; //minus of magnetic energy at ground state (but should be in positive sign)
      #endif
    #endif

    #ifdef SLDNC
    vector Im; //StonerI_i * M_i
    vector Im_ik; // Sum_k StonerI_k * M_k * fik / Sum_k fik
    #endif

    char element[2]; //specify the element of current atom

    #ifdef localvol
    double sum_rij_m1; //Sum rij^-1
    double sum_rij_m2; //Sum rij^-2
    #endif

    double local_volume;
  
    //linking atoms in a link cell
    struct atom_struct *next_atom_ptr;
    struct atom_struct *this_atom_ptr;
    struct atom_struct *prev_atom_ptr;
        
    int old_cell_index; //atom in which link cell before "links()"
    int new_cell_index; //atom in which link cell after  "links()"

};

struct cell_struct {

    struct atom_struct *head_ptr;
    struct atom_struct *this_ptr;
    struct atom_struct *tail_ptr;

    int neigh_cell[26];
    int no_of_atoms_in_cell;

    int type; //to allocate the cell in suitable group for parallel programming
              //used in allocate.cpp; due to symplectic method
    #ifdef eltemp
      double last_total_energy;
      double total_energy;

      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
      double Ts_R;
      double sum_R_up;
      double sum_R_dn;
      double Ges;
      #endif

      #if defined SDHL || defined SLDHL || defined SLDNC
      double Ts_L;
      double sum_L_up;
      double sum_L_dn;
      #endif

      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
      double Tl;
      double Gel;
      #endif

      double Te; //Electron temeprature of a link cell
      double Ee; //Electron energy of a link cell, but in unit of per atom

    #endif
    
    #ifdef localcolmot
    vector ave_fluct_force;
    vector ave_p;
    #endif


};

#if defined SLDNC
struct paraLandau {

    double alpha;
    double beta0;
    double beta1;
    double A0;
    double A1;
    double A2;
    double B0;
    double B1;
    double C0;
    double C1;
    double D0;
    double D1;
    double StonerI;
}
#endif


#ifdef GPU

struct varGPU {

    int ninput;
    double finput;
    double rmax;
    double rhomax;
    double finput_over_rmax;
    double finput_over_rhomax;

    int nperfect;
    int natom;

    vector box_length;
    vector box_length_half;
    box_vector d;
    box_vector Inv_d;
    double box_volume;
    double density;
   
    int no_of_link_cell_x;
    int no_of_link_cell_y;
    int no_of_link_cell_z;
    int ncells;

    double a_lattice;
    double c_lattice;
    int no_of_unit_cell_x;
    int no_of_unit_cell_y;
    int no_of_unit_cell_z;
    int unit_cell_no_of_atom;
    double unit_cell_edge_x;
    double unit_cell_edge_y;
    double unit_cell_edge_z;

    double atmass;
    double temperature;

    double gamma_L_over_mass;
    double gamma_L;
    double gamma_S_H;
    double gamma_S_HL;

    double stress_xx;
    double stress_yy;
    double stress_zz;
    double pressure;
    double baro_damping_time;

    double rcut_pot;
    double rcut_mag;
    double rcut_max;
    double rcut_vol;
    double min_length_link_cell;

    double rcut_pot_sq;
    double rcut_mag_sq;
    double rcut_max_sq;

    vector Hext;

    double displace_limit;
    double phi_limit;
    
    #if defined SLDNC
    struct paraLandau para;
    #endif
};
#endif
