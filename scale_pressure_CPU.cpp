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

#if defined CPU

#include "spilady.h"

#if defined PRESSURE

void scale_pressure_CPU(){

    double delta_time = total_time - last_total_time_pressure;
    last_total_time_pressure = total_time;

    double ftmass = 100e0; //the fiticious mass of pressure piston, in unit GPa
    double factor = pow(1e0 + delta_time/baro_damping_time*(pressure0 - pressure)/ftmass, 1e0/3e0); //1 eV/A^3 = 160.217653 GPa

    d.xx *= factor;
    d.yx *= factor;
    d.yy *= factor;
    d.zx *= factor;
    d.zy *= factor;
    d.zz *= factor;
   
    Inv_d = inverse_box_vector(d);

    box_length.x = fabs(d.xx);
    box_length.y = sqrt(d.yx*d.yx + d.yy*d.yy);
    box_length.z = sqrt(d.zx*d.zx + d.zy*d.zy + d.zz*d.zz);

    box_length_half = vec_divide(box_length, 2e0);
    box_volume = d.xx*d.yy*d.zz;
    density = natom/box_volume;

    #pragma omp parallel for
    for (int i = 0; i < natom; ++i) (first_atom_ptr+i)->r = vec_times(factor, (first_atom_ptr+i)->r);
}

void scale_pressure(){
    scale_pressure_CPU();
}

#endif
#endif
