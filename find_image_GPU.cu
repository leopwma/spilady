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


__device__ void find_image_d(vector &rij, struct varGPU *var_ptr_d){

    vector qij;
    qij.x = var_ptr_d->Inv_d.xx*rij.x + var_ptr_d->Inv_d.yx*rij.y + var_ptr_d->Inv_d.zx*rij.z;
    qij.y =                             var_ptr_d->Inv_d.yy*rij.y + var_ptr_d->Inv_d.zy*rij.z;
    qij.z =                                                         var_ptr_d->Inv_d.zz*rij.z;

    int index = 0;

    //find image of j closest to i
    if (qij.x <  -0.5e0){ qij.x += 1e0; ++index;}
    if (qij.x >=  0.5e0){ qij.x -= 1e0; ++index;}
    if (qij.y <  -0.5e0){ qij.y += 1e0; ++index;}
    if (qij.y >=  0.5e0){ qij.y -= 1e0; ++index;}
    if (qij.z <  -0.5e0){ qij.z += 1e0; ++index;}
    if (qij.z >=  0.5e0){ qij.z -= 1e0; ++index;}

    if (index > 0){
        rij.x = var_ptr_d->d.xx*qij.x + var_ptr_d->d.yx*qij.y + var_ptr_d->d.zx*qij.z;
        rij.y =                         var_ptr_d->d.yy*qij.y + var_ptr_d->d.zy*qij.z;
        rij.z =                                                 var_ptr_d->d.zz*qij.z;
    }
}
#endif


    

