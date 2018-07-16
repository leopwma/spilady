/********************************************************************************
*
*   Copyright (C) 2015 Culham Centre for Fusion Energy
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
*   1) variable "kappa_e" is added.
*   2) variable "Msteps_quantum" is added.
*   3) variable "Nfrequency_quantum" is added.
*
********************************************************************************/

#include "spilady.h"

void read_variables(){

    ifstream infile("variables.in");

    if (infile) {
        cout << "Start reading variables ..." << '\n';
    } else {
        cout << "ERROR: You need to have a variable.in file" << '\n';
        exit(1);
    }

    while (!infile.eof()){

        char variable[256];
        memset(variable, 0, 256);

        char value[256];
        memset(value, 0, 256);

        infile >> variable >> value;
        
        #ifdef GPU
        if (strcmp(variable, "current_device")         == 0) current_device = atoi(value);
        if (strcmp(variable, "no_of_threads")          == 0) no_of_threads = atoi(value);
        #endif

        #ifdef OMP
        if (strcmp(variable, "OMP_threads")            == 0) OMP_threads = atoi(value);
        #endif

        if (strcmp(variable, "out_body")               == 0) {
            out_body = (char *) realloc(out_body, strlen(value)+1);
            strcpy(out_body,value);
            out_body[strlen(value)] = '\0';
        }

        #ifdef runstep
        if (strcmp(variable, "no_of_production_steps") == 0) no_of_production_steps = atoi(value);
        #else
        if (strcmp(variable, "total_production_time")  == 0) total_production_time = atof(value);
        #endif

        if (strcmp(variable, "step")                   == 0) step = atof(value);
        if (strcmp(variable, "interval_of_print_out")  == 0) interval_of_print_out = atoi(value);
        if (strcmp(variable, "interval_of_config_out") == 0) interval_of_config_out = atoi(value);

        #ifdef writevsim
        if (strcmp(variable, "interval_of_vsim")       == 0) interval_of_vsim = atoi(value);
        if (strcmp(variable, "vsim_prec")              == 0) vsim_prec = atof(value);
        #endif

        #ifdef readconf
        if (strcmp(variable, "in_config")              == 0) {
            in_config = (char *) realloc(in_config, strlen(value)+1);
            strcpy(in_config,value);
            in_config[strlen(value)] = '\0';
        }
        #endif
        
        #ifdef readvsim
        if (strcmp(variable, "in_vsim_atom")           == 0){
            in_vsim_atom = (char *) realloc(in_vsim_atom, strlen(value)+1);
            strcpy(in_vsim_atom,value);
            in_vsim_atom[strlen(value)]= '\0';
        }
          #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
          if (strcmp(variable, "in_vsim_spin")           == 0){
              in_vsim_spin = (char *) realloc(in_vsim_spin, strlen(value)+1);
              strcpy(in_vsim_spin,value);
              in_vsim_spin[strlen(value)] = '\0';
          }
         #endif
       #endif
        

        #if defined readTe
        if (strcmp(variable, "in_eltemp")              == 0) {
            in_eltemp = (char *) realloc(in_eltemp, strlen(value)+1);
            strcpy(in_eltemp,value);
            in_eltemp[strlen(value)] = '\0';
        }
        #endif

        #ifdef eltemp
        if (strcmp(variable, "kappa_e")                == 0) kappa_e = atof(value);
        #endif
        
        #ifdef inittime
        if (strcmp(variable, "start_time")             == 0) start_time = atof(value);
        #endif

        if (strcmp(variable, "a_lattice")              == 0) {
            a_lattice = atof(value);
            
            #if defined bcc100 || defined fcc100
            unit_cell_edge_x = a_lattice; //x dimension of a single cell
            unit_cell_edge_y = a_lattice; //y dimension of a single cell
            unit_cell_edge_z = a_lattice; //z dimension of a single cell
            #endif
            
            #ifdef bcc111
            unit_cell_edge_x = a_lattice*sqrt(3e0); //x dimension of a single cell
            unit_cell_edge_y = a_lattice*sqrt(6e0); //y dimension of a single cell
            unit_cell_edge_z = a_lattice*sqrt(2e0); //z dimension of a single cell
            #endif
            
            #ifdef hcp0001
            unit_cell_edge_x = a_lattice; //x dimension of a single cell
            unit_cell_edge_y = a_lattice*sqrt(3e0)/2e0; //y dimension of a single cell
            #endif
        }

        #ifdef hcp0001
        if (strcmp(variable, "c_lattice")              == 0){
            c_lattice = atof(value);
            unit_cell_edge_z = c_lattice; //z dimension of a single cell
        }
        #endif

        if (strcmp(variable, "no_of_unit_cell_x")      == 0) no_of_unit_cell_x = atoi(value);
        if (strcmp(variable, "no_of_unit_cell_y")      == 0) no_of_unit_cell_y = atoi(value);
        if (strcmp(variable, "no_of_unit_cell_z")      == 0) no_of_unit_cell_z = atoi(value);

        #if defined bcc100 || defined bcc111 || defined fcc100 || defined hcp0001
        if (strcmp(variable, "element")                == 0) strcpy(element,value);
        #endif

        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        if (strcmp(variable, "atmass")                 == 0) atmass = atof(value)*utoeV*pow(length_scale,2);
        #endif
 
        if (strcmp(variable, "temperature")            == 0) temperature = atof(value)*boltz;

        #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined initmomentum && !defined lattlang
        if (strcmp(variable, "initTl")                 == 0) initTl = atof(value)*boltz;
        #endif

        #if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined initspin
        if (strcmp(variable, "mag_mom")                == 0) mag_mom = atof(value); 
        #endif


        #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined lattlang
        if (strcmp(variable, "gamma_L_over_mass")      == 0)  gamma_L_over_mass = atof(value);
        #endif
        
        
        #ifdef spinlang
          #if defined SDH || defined SLDH
          if (strcmp(variable, "gamma_S_H")              == 0) gamma_S_H = atof(value);
          #endif
          #if defined SDHL || defined SLDHL
          if (strcmp(variable, "gamma_S_HL")             == 0) gamma_S_HL = atof(value);
          #endif
        #endif

        #ifdef STRESS
        if (strcmp(variable, "interval_of_scale_stress") == 0) interval_of_scale_stress = atoi(value);
        if (strcmp(variable, "stress_xx")                == 0) stress_xx  = atof(value);
        if (strcmp(variable, "stress_yx")                == 0) stress_yx  = atof(value);
        if (strcmp(variable, "stress_yy")                == 0) stress_yy  = atof(value);
        if (strcmp(variable, "stress_zx")                == 0) stress_zx  = atof(value);
        if (strcmp(variable, "stress_zy")                == 0) stress_zy  = atof(value);
        if (strcmp(variable, "stress_zz")                == 0) stress_zz  = atof(value);
        #endif


        #ifdef PRESSURE
        if (strcmp(variable, "interval_of_scale_pressure") == 0) interval_of_scale_pressure = atoi(value);
        if (strcmp(variable, "pressure")               == 0) pressure  = atof(value);
        #endif

        #if defined STRESS || defined PRESSURE
        if (strcmp(variable, "baro_damping_time")      == 0) baro_damping_time = atof(value);
        #endif

        if (strcmp(variable, "random_seed")            == 0) random_seed = atoi(value);

        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        if (strcmp(variable, "rcut_pot")               == 0) rcut_pot = atof(value);
        #endif

        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        if (strcmp(variable, "rcut_mag")               == 0) rcut_mag = atof(value);
        #endif

        if (strcmp(variable, "rcut_max")               == 0)  rcut_max = atof(value);
       
        if (strcmp(variable, "min_length_link_cell")   == 0) min_length_link_cell = atof(value);

        #ifdef localvol
        if (strcmp(variable, "rcut_vol")               == 0) rcut_vol = atof(value);
        #endif

        //if (strcmp(variable, "rhomax")                 == 0) rhomax = atof(value);

        #ifdef changestep
          #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
          if (strcmp(variable, "displace_limit")         == 0) displace_limit  = atof(value);
          #endif
          #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
          if (strcmp(variable, "phi_limit")              == 0) phi_limit = atof(value);
          #endif
        #endif

        #ifdef quantumnoise
        if (strcmp(variable, "Msteps_quantum")           == 0) Msteps_quantum = atoi(value);
        if (strcmp(variable, "Nfrequency_quantum")       == 0) {
            Nfrequency_quantum = atoi(value);
            Nfrequency_quantum_2 = Nfrequency_quantum*2;
        }
        #endif

    }
    infile.close();

/* convert some variables into internal variables */

    #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined lattlang
    gamma_L = gamma_L_over_mass*atmass;
    #endif

    if (rcut_max > min_length_link_cell) min_length_link_cell = rcut_max + 0.001;

    #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined initmomentum && !defined lattlang
    if (initTl < 0e0) initTl = temperature;
    #endif

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    rcut_pot_sq = pow(rcut_pot,2);
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    rcut_mag_sq = pow(rcut_mag,2);
    #endif

    rcut_max_sq = pow(rcut_max,2);

/* output all variables that are using. */

    ofstream outfile("variables.all");
    
    #ifdef GPU
    outfile << "current_device " << current_device << '\n';
    outfile << "no_of_threads " << no_of_threads << '\n';
    #endif

    #ifdef OMP
    outfile << "OMP_threads " << OMP_threads << '\n';;
    #endif

    outfile << "out_body " << out_body << '\n';

    #ifdef runstep
    outfile << "no_of_production_steps " << no_of_production_steps << '\n';
    #else
    outfile << "total_production_time " << total_production_time << '\n';
    #endif

    outfile << "step " <<  step << '\n';
    outfile << "interval_of_print_out " << interval_of_print_out << '\n';
    outfile << "interval_of_config_out " << interval_of_config_out << '\n';

    #ifdef writevsim
    outfile << "interval_of_vsim " << interval_of_vsim << '\n';
    outfile << "vsim_prec " << vsim_prec << '\n';
    #endif

    #ifdef readconf
    outfile << "in_config " << in_config << '\n';
    #endif
    
    #ifdef readvsim
    outfile << "in_vsim_atom " << in_vsim_atom << '\n';
      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
      outfile << "in_vsim_spin " << in_vsim_spin << '\n';
      #endif
    #endif

    #if defined readTe
    outfile << "in_eltemp " << in_eltemp << '\n';
    #endif

    #ifdef eltemp
    outfile << "kappa_e " << kappa_e << '\n';
    #endif

    #if defined bcc100 || defined bcc111 || defined fcc100 || defined hcp0001
      outfile << "element " << element << '\n';
      outfile << "a_lattice " << a_lattice << '\n';
      #ifdef hcp0001
      outfile << "c_lattice " << c_lattice << '\n';
      #endif
      outfile << "no_of_unit_cell_x " << no_of_unit_cell_x << '\n';
      outfile << "no_of_unit_cell_y " << no_of_unit_cell_y << '\n';
      outfile << "no_of_unit_cell_z " << no_of_unit_cell_z << '\n';
    #endif
    
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    outfile << "atmass " << atmass/(utoeV*pow(length_scale,2)) << '\n';
    #endif

    #if defined lattlang || defined spinlang || (defined initmomentum && !defined lattlang)
    outfile << "temperature " << temperature/boltz << '\n';
    #endif

    #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined initmomentum && !defined lattlang
    outfile << "initTl " << initTl/boltz << '\n';
    #endif

    #if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined initspin
    outfile << "mag_mom " << mag_mom << '\n';
    #endif

    #if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined lattlang
    outfile << "gamma_L_over_mass " << gamma_L_over_mass << '\n';
    #endif

    #ifdef spinlang
      #if defined SDH || defined SLDH
      outfile << "gamma_S_H " << gamma_S_H << '\n';
      #endif
      #if defined SDHL || defined SLDHL
      outfile << "gamma_S_HL " << gamma_S_HL << '\n';
      #endif
    #endif

    #ifdef STRESS
    outfile << "interval_of_scale_stress " << interval_of_scale_stress << '\n';
    outfile << "stress_xx " << stress_xx << '\n';
    outfile << "stress_yx " << stress_yx << '\n';
    outfile << "stress_yy " << stress_yy << '\n';
    outfile << "stress_zx " << stress_zx << '\n';
    outfile << "stress_zy " << stress_zy << '\n';
    outfile << "stress_zz " << stress_zz << '\n';
    #endif

    #ifdef PRESSURE
    outfile << "interval_of_scale_pressure " << interval_of_scale_pressure << '\n';
    outfile << "pressure " << pressure  << '\n';
    #endif

    #if defined STRESS || defined PRESSURE
    outfile << "baro_damping_time " << baro_damping_time << '\n';
    #endif

    outfile << "random_seed " << random_seed << '\n';

    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    outfile << "rcut_pot " << rcut_pot << '\n';
    #endif

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    outfile << "rcut_mag " << rcut_mag << '\n';
    #endif

    outfile << "rcut_max " << rcut_max << '\n';

    outfile << "min_length_link_cell " << min_length_link_cell << '\n';

    #ifdef localvol
    outfile << "rcut_vol " << rcut_vol << '\n';
    #endif

    #ifdef changestep
      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
      outfile << "displace_limit " << displace_limit << '\n';
      #endif
      #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
      outfile << "phi_limit " << phi_limit << '\n';
      #endif
    #endif

    #ifdef quantumnoise
      outfile << "Msteps_quantum " << Msteps_quantum  << '\n';
      outfile << "Nfrequency_quantum " << Nfrequency_quantum << '\n';
    #endif

    outfile.close();

    cout << "Finished reading variables." << '\n';

}

