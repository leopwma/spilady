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

void core(int current_step){


    #ifdef extfield
    external_field(current_step);
    #endif
    #ifdef extforce
    external_force(current_step);
    #endif
  
    #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
    links();
    #endif

    #ifdef eltemp
    core_dTe(step/2e0);
    #endif

    #if defined  SDH || defined SDHL
    core_ds(step);
    if ((current_step + 1)%interval_of_print_out == 0) calculate_force_energy();
    #endif
    
    #ifdef MD
    core_dp(step/2e0);
    core_dr(step);
    calculate_rho();
    calculate_force_energy();
    core_dp(step/2e0);
    #endif
    
    #if defined SLDH || defined SLDHL || defined SLDNC
    core_dp(step/2e0);
    core_ds(step/2e0);
    core_dr(step);
    calculate_rho();
    core_ds(step/2e0);
    calculate_force_energy();
    core_dp(step/2e0);
    #endif

    #ifdef eltemp
    calculate_temperature();
    core_dTe(step/2e0);
    #endif
}

