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

#if defined STRESS

#include "prototype_GPU.h"

__global__ void LP1ScSt(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d);

void scale_stress_GPU(){

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
    box_volume = vec_volume(box_length);
    density = natom/box_volume;

    cudaMemcpy(&(var_ptr_d->d),               &d,               sizeof(box_vector), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->box_length),      &box_length,      sizeof(vector), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->box_length_half), &box_length_half, sizeof(vector), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->box_volume),      &box_volume,      sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->density),         &density,         sizeof(double), cudaMemcpyHostToDevice);

    LP1ScSt<<<no_of_blocks,no_of_threads>>>(var_ptr_d, first_atom_ptr_d);

    //calculate the new Inverse of d
    Inv_d = inverse_box_vector(d);

    cudaMemcpy(&(var_ptr_d->Inv_d),           &Inv_d,           sizeof(box_vector), cudaMemcpyHostToDevice);

}

void scale_stress(){
    scale_stress_GPU();
}

__global__ void LP1ScSt(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        vector q;
        //use the old inverse of d to transform the system into general coordinate
        q.x = var_ptr_d->Inv_d.xx * atom_ptr->r.x + var_ptr_d->Inv_d.yx * atom_ptr->r.y + var_ptr_d->Inv_d.zx * atom_ptr->r.z;      
        q.y =                                       var_ptr_d->Inv_d.yy * atom_ptr->r.y + var_ptr_d->Inv_d.zy * atom_ptr->r.z;       
        q.z =                                                                             var_ptr_d->Inv_d.zz * atom_ptr->r.z;

        //use the new d to transform the system back to real coordinate
        atom_ptr->r.x = var_ptr_d->d.xx * q.x + var_ptr_d->d.yx * q.y + var_ptr_d->d.zx * q.z;
        atom_ptr->r.y =                         var_ptr_d->d.yy * q.y + var_ptr_d->d.zy * q.z;
        atom_ptr->r.z =                                                 var_ptr_d->d.zz * q.z;
    }
}




#endif
#endif

