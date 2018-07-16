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

vector vec_add(vector a, vector b){

    vector c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    
    return c;
}

vector vec_sub(vector a, vector b){

    vector c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;

    return c;
}

vector vec_cross(vector a, vector b){

    vector c;
    c.x = a.y*b.z - b.y*a.z;
    c.y = a.z*b.x - b.z*a.x;
    c.z = a.x*b.y - b.x*a.y;

    return c;
}

vector vec_times(double a, vector b){

    vector c;
    c.x = a*b.x;
    c.y = a*b.y;
    c.z = a*b.z;

    return c;
}

vector vec_divide(vector a, double b){

    vector c;
    c.x = a.x/b;
    c.y = a.y/b;
    c.z = a.z/b;

    return c;
}

double vec_dot(vector a, vector b){

    double c;
    c = a.x*b.x 
      + a.y*b.y 
      + a.z*b.z;
    
    return c;
}

double vec_sq(vector a){

    double c;
    c = a.x*a.x + a.y*a.y + a.z*a.z;

    return c;
}

double vec_length(vector a){

    double c;
    c = sqrt(vec_sq(a));

    return c;
}

vector vec_zero(){

    vector c;
    c.x = 0e0;
    c.y = 0e0;
    c.z = 0e0;
    
    return c;

}

vector vec_init(double x, double y, double z){

    vector c;
    c.x = x;
    c.y = y;
    c.z = z;

    return c;

}

double vec_volume(vector a){

    return a.x*a.y*a.z;
    
}

box_vector inverse_box_vector(box_vector d){

    box_vector Inv_d;

    Inv_d.xx = 1e0/d.xx;
    Inv_d.yx = -d.yx/(d.xx*d.yy);
    Inv_d.yy = 1e0/d.yy;
    Inv_d.zx = (d.yx*d.zy - d.yy*d.zx)/(d.xx*d.yy*d.zz);
    Inv_d.zy = -d.zy/(d.yy*d.zz);
    Inv_d.zz = 1e0/d.zz;

    return Inv_d;

}

