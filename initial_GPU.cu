/********************************************************************************
*
*   Copyright (C) 2015-2018 Culham Centre for Fusion Energy,
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
*   Version: 1.0.2
*   Date:    27 July 2018
*   Author:  Pui-Wai (Leo) MA
*   Contact: info@spilady.ccfe.ac.uk
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*
********************************************************************************/

#ifdef GPU

#include "spilady.h"
#include "prototype_GPU.h"

int initial_no_of_threads(cudaDeviceProp deviceProp);

void initial_GPU(){

    cudaSetDevice(current_device); //To set which GPU card is going to use.

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, current_device);

    no_of_MP = deviceProp.multiProcessorCount; //no. of multi Processor
    no_of_threads = initial_no_of_threads(deviceProp); //assign the no. of threads automatically according to the architecture of GPU card
    no_of_blocks = (natom + no_of_threads - 1)/no_of_threads; //set the total no. of blocks in a grid

    cout << "Nvidia GPU card is running." << '\n';
    cout << "Device name                  = " << deviceProp.name << '\n';
    cout << "Device index                 = " << current_device << '\n';
    cout << "Number of multi-processors   = " << no_of_MP << '\n';
    cout << "Number of threads in a block = " << no_of_threads << '\n';
    cout << "Number of blocks             = " << no_of_blocks << '\n';

    potential_table_GPU();
    copy_CPU_to_GPU();

}

int initial_no_of_threads(cudaDeviceProp deviceProp){

    int cores_per_MP = 0;

    //Information in this function is written according to the specification of different nvidia cards.
    //It requires modification in the future.

    switch (deviceProp.major){
        case 2: // Fermi
            if (deviceProp.minor == 0) cores_per_MP =32;
            else if (deviceProp.minor == 1) cores_per_MP =48;
            else cores_per_MP = 32;
            break;
        case 3: // Kepler
            cores_per_MP = 192;
            break;
        case 5: // Maxwell
            cores_per_MP = 128;
            break;
        case 6: // Pascal
            if (deviceProp.minor == 0){
                cores_per_MP = 64;
            } else if (deviceProp.minor == 1) {
                cores_per_MP = 128;
            } else {
               cores_per_MP = 64;
               cout << "An unknown type of arch 6.x. Assume the same as 6.0.\n";
            }
            break;
        case 7: // Volta
            if (deviceProp.minor == 0){
                cores_per_MP = 64;
            } else {
               cores_per_MP = 64;
               cout << "An unknown type of arch 7.x. Assume the same as 7.0.\n";
            }
            break;
        default:
            cores_per_MP = 32;
            cout << "An unknown tpye of GPU. Assume cores_per_MP = 32.\n";
            break;
      }
      return cores_per_MP;
}

#endif
