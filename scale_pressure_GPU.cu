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

#if defined PRESSURE

#include "prototype_GPU.h"

/***************************************************************************
* GPU prototype
****************************************************************************/

__global__ void LP1ScPr(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double factor);


/***************************************************************************
* CPU codes
****************************************************************************/

void scale_pressure_GPU(){

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

    cudaMemcpy(&(var_ptr_d->d),               &d,               sizeof(box_vector), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->Inv_d),           &Inv_d,           sizeof(box_vector), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->box_length),      &box_length,      sizeof(vector), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->box_length_half), &box_length_half, sizeof(vector), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->box_volume),      &box_volume,      sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&(var_ptr_d->density),         &density,         sizeof(double), cudaMemcpyHostToDevice);

    LP1ScPr<<<no_of_blocks,no_of_threads>>>(var_ptr_d, first_atom_ptr_d, factor);

}

void scale_pressure(){
    scale_pressure_GPU();
}

/***************************************************************************
* GPU codes
****************************************************************************/

__global__ void LP1ScPr(struct varGPU *var_ptr_d, struct atom_struct *first_atom_ptr_d, double factor)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < var_ptr_d->natom){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr_d + i;
        atom_ptr->r = vec_times_d(factor, atom_ptr->r);
    }
}

#endif
#endif
