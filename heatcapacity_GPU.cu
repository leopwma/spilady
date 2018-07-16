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

#if defined GPU

#include "spilady.h"
#include "prototype_GPU.h"

#if defined eltemp

// a functional form for the heat capacity C = a*tanh(bT) per atom is chosen.
__device__ double Ce_d(double Te){

    double a = 3e0;
    double b = 2e-4/boltz; //K^-1 / (eV K^-1)
    double C = a*tanh(b*Te);
    return C;
}

__device__ double Te_to_Ee_d(double Te){

    double a = 3e0;
    double b = 2e-4/boltz; //K^-1 / (eV K^-1)
    double Ee = a/b*log(cosh(b*Te));
    return Ee;

}

__device__ double Ee_to_Te_d(double Ee){

    double a = 3e0;
    double b = 2e-4/boltz; //K^-1 / (eV K^-1)
    double Te = acosh(exp(b/a*Ee))/b;
    return Te;
}

#endif

#endif
