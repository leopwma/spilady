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

#ifdef fcc100

void fcc100bulk(){

    nperfect = no_of_unit_cell_x*no_of_unit_cell_y*no_of_unit_cell_z*unit_cell_no_of_atom;
    natom = nperfect;

    box_length.x = double(no_of_unit_cell_x)*unit_cell_edge_x;
    box_length.y = double(no_of_unit_cell_y)*unit_cell_edge_y;
    box_length.z = double(no_of_unit_cell_z)*unit_cell_edge_z;
    box_length_half = vec_times(0.5, box_length);
    d.xx = box_length.x;
    d.yx = 0e0;
    d.yy = box_length.y;
    d.zx = 0e0;
    d.zy = 0e0;
    d.zz = box_length.z;
    Inv_d = inverse_box_vector(d);
    box_volume = vec_volume(box_length);
    density = double(natom)/box_volume;

    first_atom_ptr = (atom_struct*)malloc(natom*sizeof(atom_struct));

    vector dist = vec_init(a_lattice/2e0, a_lattice/2e0, a_lattice/2e0);

// assign positions of atoms in basic cell
    first_atom_ptr->r.x = 0e0;
    first_atom_ptr->r.y = 0e0;
    first_atom_ptr->r.z = 0e0;
    (first_atom_ptr+1)->r.x = 0e0;
    (first_atom_ptr+1)->r.y = dist.y;
    (first_atom_ptr+1)->r.z = dist.z;
    (first_atom_ptr+2)->r.x = dist.x;
    (first_atom_ptr+2)->r.y = 0e0;
    (first_atom_ptr+2)->r.z = dist.z;
    (first_atom_ptr+3)->r.x = dist.x;
    (first_atom_ptr+3)->r.y = dist.y;
    (first_atom_ptr+3)->r.z = 0e0;

// replicate first four positions over nunits
     int m = 0;
     int n = 0;
     for (int i = 0; i < no_of_unit_cell_z; ++i ){
         for (int j = 0; j < no_of_unit_cell_y; ++j ){
             for (int k = 0 ; k < no_of_unit_cell_x; ++k ){
                 for (int ij = 0 ; ij < 4; ++ij){
                     if (n < nperfect){
                         struct atom_struct *atom_ptr;
                         atom_ptr = first_atom_ptr+ij+m;
                         atom_ptr->r.x = ((first_atom_ptr+ij)->r.x)+2e0*dist.x*double(k);
                         atom_ptr->r.y = ((first_atom_ptr+ij)->r.y)+2e0*dist.y*double(j);
                         atom_ptr->r.z = ((first_atom_ptr+ij)->r.z)+2e0*dist.z*double(i);
                     }
                     ++n;
                 }
                 m += 4;
             }
         }
     }

    for (int i = 0 ; i < nperfect; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        atom_ptr->r = vec_add(atom_ptr->r, vec_divide(dist,2e0));
        periodic(atom_ptr->r);
    }
    
}
#endif
