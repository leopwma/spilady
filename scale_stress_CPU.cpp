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

#if defined STRESS

void scale_stress_CPU(){

    double delta_time = total_time - last_total_time_stress;
    last_total_time_stress = total_time;

    double ftmass = 100e0; //the fiticious mass of pressure piston, in unit GPa
    double pre_fact = delta_time/baro_damping_time/ftmass; //1 eV/A^3 = 160.217653 GPa

    box_vector factor;
    factor.xx = pre_fact*(ave_stress11 - stress_xx);
    factor.yx = pre_fact*(ave_stress12 - stress_yx);
    factor.yy = pre_fact*(ave_stress22 - stress_yy);
    factor.zx = pre_fact*(ave_stress31 - stress_zx);
    factor.zy = pre_fact*(ave_stress23 - stress_zy);
    factor.zz = pre_fact*(ave_stress33 - stress_zz);

    d.xx = (1e0 + factor.xx)*d.xx;
    d.yx = (1e0 + factor.xx)*d.yx + factor.yx*(d.yy+d.xx)/2e0;
    d.yy = (1e0 + factor.yy)*d.yy;
    d.zx = (1e0 + factor.xx)*d.zx + factor.zx*(d.zz+d.xx)/2e0;
    d.zy = (1e0 + factor.yy)*d.zy + factor.zy*(d.zz+d.yy)/2e0;
    d.zz = (1e0 + factor.zz)*d.zz;
    
    box_length.x = fabs(d.xx);
    box_length.y = sqrt(d.yx*d.yx + d.yy*d.yy);
    box_length.z = sqrt(d.zx*d.zx + d.zy*d.zy + d.zz*d.zz);
    box_length_half = vec_divide(box_length, 2e0);
    box_volume = d.xx*d.yy*d.zz;
    density = natom/box_volume;

    #ifdef OMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < natom; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;

        vector q;
        //use the old inverse of d to transform the system into general coordinate
        q.x = Inv_d.xx*atom_ptr->r.x + Inv_d.yx*atom_ptr->r.y + Inv_d.zx*atom_ptr->r.z;      
        q.y =                          Inv_d.yy*atom_ptr->r.y + Inv_d.zy*atom_ptr->r.z;       
        q.z =                                                   Inv_d.zz*atom_ptr->r.z;

        //use the new d to transform the system back to real coordinate
        atom_ptr->r.x = d.xx*q.x + d.yx*q.y + d.zx*q.z;
        atom_ptr->r.y =            d.yy*q.y + d.zy*q.z;
        atom_ptr->r.z =                       d.zz*q.z;
        
    }

    //calculate the new Inverse of d
    Inv_d = inverse_box_vector(d);

}

void scale_stress(){
    scale_stress_CPU();
}

#endif
#endif
