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
#ifdef readconf

void read_config(){

    double tt;

    ifstream in_file(in_config);
    char input_type[256];    
  
    in_file >> ws >> natom >> total_time >> input_type;
    in_file >> ws >> d.xx >> d.yx >> d.yy ;
    in_file >> ws >> d.zx >> d.zy >> d.zz ;

    #ifdef MD
    char current_type[] = "MD";
    #endif
    #ifdef SDH
    char current_type[] = "SDH";
    #endif
    #ifdef SDHL
    char current_type[] = "SDHL";
    #endif
    #ifdef SLDH
    char current_type[] = "SLDH";
    #endif
    #ifdef SLDHL
    char current_type[] = "SLDHL";
    #endif

   
    if (strcmp(current_type, input_type) == 0) {
        cout << "Reading the input configuration file for " << current_type << " simulation." << '\n';
    } else {
        cout << "ERROR: the input configuration file is for " << input_type << ", not for "  << current_type << '\n';
        exit(1);
    }
    


    Inv_d = inverse_box_vector(d);

    first_atom_ptr = (atom_struct*)malloc(natom*sizeof(atom_struct));

    int ndummy; //just dummy

    for (int i = 0 ; i < natom; ++i){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;

        in_file >> ws >> ndummy >> atom_ptr->element
                      >> atom_ptr->r.x >> atom_ptr->r.y >> atom_ptr->r.z
                     #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                     >> atom_ptr->p.x >> atom_ptr->p.y >> atom_ptr->p.z
                     >> atom_ptr->f.x >> atom_ptr->f.y >> atom_ptr->f.z
                     >> atom_ptr->stress11 
                     >> atom_ptr->stress22    
                     >> atom_ptr->stress33
                     >> atom_ptr->stress12 
                     >> atom_ptr->stress23 
                     >> atom_ptr->stress31
                     >> atom_ptr->rho
                     >> atom_ptr->ke
                     >> atom_ptr->pe
                     #endif

                     #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                       #if defined magmom || defined SLDNC
                       >> atom_ptr->m.x >> atom_ptr->m.y >> atom_ptr->m.z >> atom_ptr->m0
                       #else
                       >> atom_ptr->s.x >> atom_ptr->s.y >> atom_ptr->s.z >> atom_ptr->s0
                       #endif

                       >> atom_ptr->Heff_H.x >> atom_ptr->Heff_H.x >> atom_ptr->Heff_H.z

                       #if defined SDHL || defined SLDHL
                       >> atom_ptr->Heff_L.x >> atom_ptr->Heff_L.x >> atom_ptr->Heff_L.z
                       #endif

                       #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                       >> atom_ptr->me
                       >> atom_ptr->me0
                       #endif
                     #endif

                     >> atom_ptr->local_volume
                     ;
    }
    in_file.close();

    cout << "Read in atoms configuration file completed." << '\n';

    box_length.x = fabs(d.xx);
    box_length.y = sqrt(d.yx*d.yx + d.yy*d.yy);
    box_length.z = sqrt(d.zx*d.zx + d.zy*d.zy + d.zz*d.zz);
    box_length_half = vec_times(0.5, box_length);
    box_volume = d.xx*d.yy*d.zz;
    density = double(natom)/box_volume;

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC

    #ifdef OMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < natom; ++i) {
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        #if defined magmom || defined SLDNC
        atom_ptr->s = vec_divide(atom_ptr->m, -el_g);
        atom_ptr->s0 = vec_length(atom_ptr->s);
        #else
        atom_ptr->m = vec_times(-el_g,atom_ptr->s);
        atom_ptr->m0 = vec_length(atom_ptr->m);
        #endif
    }
    
    #endif
}

#endif
