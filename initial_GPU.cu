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

void initial_GPU(){

    cudaSetDevice(current_device); //To set which GPU card is going to use.

    no_of_blocks = (natom + no_of_threads - 1)/no_of_threads; //set the total no. of blocks in a grid

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, current_device);
    no_of_MP = deviceProp.multiProcessorCount; //no. of multi Processor

    cout << "Nvidia GPU card is running." << '\n';
    cout << "Device name                  = " << deviceProp.name << '\n';
    cout << "Device index                 = " << current_device << '\n';
    cout << "Number of multi-processors   = " << no_of_MP << '\n';
    cout << "Number of threads in a block = " << no_of_threads << '\n';
    cout << "Number of blocks             = " << no_of_blocks << '\n';

    potential_table_GPU();
    copy_CPU_to_GPU();

}

#endif
