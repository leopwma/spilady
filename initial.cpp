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

#include "spilady.h"

void initialize(){

    cout << "Initializing!!!" << '\n';

    read_variables(); // read in variables that are different from default values.

    #ifdef OMP
    omp_set_num_threads(OMP_threads); //set the number of threads or cores in a  machines
    cout << "Number of OpenMP threads = " << OMP_threads << '\n';
    #endif

    potential_table(); //build the potential table.

    build_lattice(); //build the lattice from scratch or read-in file

    #if defined bcc100 || defined bcc111 || defined fcc100 || defined hcp0001
    initial_element();
    #endif

    #if (defined MD ||  defined SLDH || defined SLDHL || defined SLDNC) && defined initmomentum
    initial_momentum();
    #endif

    #if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined initspin
    initial_spin();
    #endif

    #ifdef extfield
    external_field(-1);
    #endif
    #ifdef extforce
    external_force(-1);
    #endif

    map_cells();           //create the link-cell system and map neighbourhood

    #ifdef GPU
    initial_GPU(); //set the device, no. of blocks and no. of threads, copy potential table, copy lattice and cells
    #endif
 
    //starting from this point. All functions need to have GPU and CPU counterpart.

    initial_links(); //put atoms into link cells

    allocate_cells(); //allocate cells into groups, for parallel programming.
                //It is necessary for spin or local collective motion cases.

    #ifndef readconf
      #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        calculate_rho();
      #endif
      calculate_force_energy();  //for spin dynamics, only energy is calculated
    #endif

    #ifdef eltemp
    calculate_temperature();
    #endif

    #ifdef inittime
    total_time = start_time;
    #endif

    #ifdef PRESSURE
    last_total_time_pressure   = total_time;
    #endif
    #ifdef STRESS
    last_total_time_stress     = total_time;
    #endif

    initialize_random_number();

    #ifdef quantumnoise
    initial_quantum_noise();
    #endif

}
