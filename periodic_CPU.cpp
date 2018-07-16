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

void periodic(vector &r){

    //transform the real coordinate into box vectors coordinate
    vector q;
    q.x = Inv_d.xx*r.x + Inv_d.yx*r.y + Inv_d.zx*r.z;
    q.y =                Inv_d.yy*r.y + Inv_d.zy*r.z;
    q.z =                               Inv_d.zz*r.z;

    int index = 0;

    if (q.x <  0e0 || q.x >= 1e0){ q.x -= int(q.x); ++index;}
    if (q.y <  0e0 || q.y >= 1e0){ q.y -= int(q.y); ++index;}
    if (q.z <  0e0 || q.z >= 1e0){ q.z -= int(q.z); ++index;}
 
    if (index > 0){
        r.x = d.xx*q.x + d.yx*q.y + d.zx*q.z;
        r.y =            d.yy*q.y + d.zy*q.z;
        r.z =                       d.zz*q.z;
    }
}


