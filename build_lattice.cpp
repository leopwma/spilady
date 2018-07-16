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

void build_lattice(){

    #ifdef readconf
      read_config();
    #endif
    #ifdef readvsim
      read_vsim();
    #endif
    #ifdef bcc100
      bcc100bulk();
    #endif
    #ifdef bcc111
      bcc111bulk();
    #endif
    #ifdef fcc100
      fcc100bulk();
    #endif
    #ifdef hcp0001
      hcp0001bulk();
    #endif

    cout << "Lattice is built!!!" << '\n';
    cout << "Box vector: " << '\n';
    cout << "x-> " << d.xx << " " << 0e0  << " " << 0e0  << '\n';
    cout << "y-> " << d.yx << " " << d.yy << " " << 0e0  << '\n';
    cout << "z-> " << d.zx << " " << d.zy << " " << d.zz << '\n';


}

