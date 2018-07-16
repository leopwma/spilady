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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h> // for CPU OpenMP

#ifdef GPU
#include <curand.h>             // Cuda Random number library
#include <curand_kernel.h>      // Cuda Random number library
#endif

using namespace std;

#include "control.h"

/**********************************************************
*
* Fixed constants
*
***********************************************************/

#define speed_of_light 299792458e0                            // meter per second
#define boltz   8.617343e-5                                   // eV per Kelvin
#define utoeV (931.494028e6/speed_of_light/speed_of_light)    // 1u = 931.494013MeV/c^2
#define length_scale 1e-10                                    // length scale; 1 Angstrom
#define energy_scale 1e0                                      // energy scale; 1 eV
#define hbar    6.58211899e-16                                // in eV s; planck constant over 2 Pi
#define muB     5.7883817555e-5                               // in eV/T; bohr magneton
#define el_g    2.0023193043622e0                             // electronic g factor
#define Pi_num  3.141592653589793238462e0                     // Pi numerical value
#define eVtoJ   1.602176487e-19                               // eV to Joule


/**********************************************************
*
* Other .h files
*
***********************************************************/

#include "struct.h"
#include "prototype_CPU.h"

/* MAIN is defined only once in the main.cpp  */
#ifdef MAIN
#define EXTERN			/* define variables in main */
#define INIT(data) = data	/* initialize data only in main */
#else
#define EXTERN extern		/* declare them extern otherwise */
#define INIT(data)		/* skip initialization otherwise */
#endif

#include "default.h"
#include "global.h"


