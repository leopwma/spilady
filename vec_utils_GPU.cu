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
#include "prototype_GPU.h"

__device__ vector vec_add_d(vector a, vector b){

    vector c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    
    return c;
}

__device__ vector vec_sub_d(vector a, vector b){

    vector c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;

    return c;
}

__device__ vector vec_cross_d(vector a, vector b){

    vector c;
    c.x = a.y*b.z - b.y*a.z;
    c.y = a.z*b.x - b.z*a.x;
    c.z = a.x*b.y - b.x*a.y;

    return c;
}

__device__ vector vec_times_d(double a, vector b){

    vector c;
    c.x = a*b.x;
    c.y = a*b.y;
    c.z = a*b.z;

    return c;
}

__device__ vector vec_divide_d(vector a, double b){

    vector c;
    c.x = a.x/b;
    c.y = a.y/b;
    c.z = a.z/b;

    return c;
}

__device__ double vec_dot_d(vector a, vector b){

    double c;
    c = a.x*b.x + a.y*b.y + a.z*b.z;
    
    return c;
}

__device__ double vec_sq_d(vector a){

    double c;
    c = a.x*a.x + a.y*a.y + a.z*a.z;

    return c;
}

__device__ double vec_length_d(vector a){

    double c;
    c = a.x*a.x + a.y*a.y + a.z*a.z;
    c = sqrt(c);

    return c;
}

__device__ vector vec_zero_d(){

    vector c;
    c.x = 0e0;
    c.y = 0e0;
    c.z = 0e0;

    return c;

}

__device__ vector vec_init_d(double x, double y, double z){

    vector c;
    c.x = x;
    c.y = y;
    c.z = z;

    return c;

}

__device__ double vec_volume_d(vector a){

    return a.x*a.y*a.z;

}

__device__ box_vector inverse_box_vector_d(box_vector d){

    box_vector Inv_d;

    Inv_d.xx = 1e0/d.xx;
    Inv_d.yx = -d.yx/(d.xx*d.yy);
    Inv_d.yy = 1e0/d.yy;
    Inv_d.zx = (d.yx*d.zy - d.yy*d.zx)/(d.xx*d.yy*d.zz);
    Inv_d.zy = -d.zy/(d.yy*d.zz);
    Inv_d.zz = 1e0/d.zz;

    return Inv_d;

}

