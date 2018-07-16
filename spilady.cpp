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

#define MAIN
#include "spilady.h"
#undef MAIN

int main(){

    time_t timer = time(NULL);
    cout << "SPILADY starts!!!" << '\n';

    initialize();
    check(-1);
    write(-1);

    cout << "I am running!!!" << '\n';
    #ifdef runstep
    for (int current_step = 0 ; current_step < no_of_production_steps ; ++current_step){
    #else
    int current_step = 0;
    while (total_time <= total_production_time + 1e-18 ){
    #endif

        core(current_step);

        total_time += step; //finished 1 step, so add time first. For print out having correct time.
        cout << current_step << " total time = " <<  total_time << " dt = " << step << '\n';

        check(current_step);
        scale(current_step);
        write(current_step);

        #ifndef runstep
        ++current_step;
        #endif      
    }

    write(-2);

    free_memory();

    cout << "SPILADY ends!!!" << '\n';
    cout << "Total running time = " << difftime(time(NULL),timer) << " seconds."<< '\n';
    
    return(0);
}
