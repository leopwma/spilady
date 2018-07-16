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

#ifdef readvsim
void read_vsim(){

    char temp[256];

    ifstream in_file_atom(in_vsim_atom);

    in_file_atom >> natom >> total_time;
    in_file_atom >> d.xx >> d.yx >> d.yy;
    in_file_atom >> d.zx >> d.zy >> d.zz;

    Inv_d = inverse_box_vector(d);

    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    ifstream in_file_spin(in_vsim_spin);
    in_file_spin >> natom;
    #endif

    first_atom_ptr = (atom_struct*)malloc(natom*sizeof(atom_struct));

    for (int i = 0 ; i < natom; ++i){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
        in_file_atom >> atom_ptr->r.x >> atom_ptr->r.y >> atom_ptr->r.z >> atom_ptr->element
                     #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                     >> atom_ptr->p.x >> atom_ptr->p.y >> atom_ptr->p.z
                     #endif
                     ;
        #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
        atom_ptr->ke = vec_sq(atom_ptr->p)/2e0/atmass;
        #endif
        
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
          double ss, theta, phi;
          in_file_spin >> temp >> ss >> theta >> phi;

          theta *= Pi_num/180e0;
          phi   *= Pi_num/180e0;
          vector s;
          s.x = ss*sin(theta)*cos(phi);
          s.y = ss*sin(theta)*sin(phi);
          s.z = ss*cos(theta);

          #if defined magmom || defined SLDNC
            atom_ptr->m = s;
            atom_ptr->m0  = ss;

            atom_ptr->s = vec_divide(atom_ptr->m,-el_g);
            atom_ptr->s0 = vec_length(atom_ptr->s);
          #else
            atom_ptr->s = s;
            atom_ptr->s0  = ss;
          #endif
        #endif

    }

    in_file_atom.close();
    #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
    in_file_spin.close();
    #endif

    cout << "Read in V_sim file(s) completed." << '\n';

    box_length.x = fabs(d.xx);
    box_length.y = sqrt(d.yx*d.yx + d.yy*d.yy);
    box_length.z = sqrt(d.zx*d.zx + d.zy*d.zy + d.zz*d.zz);
    box_length_half = vec_times(0.5, box_length);
    box_volume = d.xx*d.yy*d.zz;
    density = double(natom)/box_volume;

}
#endif
