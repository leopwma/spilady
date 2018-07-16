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

#ifdef writevsim

void write_vsim(int current_step){

    if (current_step == -1 || current_step == -2 || (current_step + 1)%interval_of_vsim == 0 ){

        #ifdef GPU
        copy_atoms_from_GPU_to_CPU();
        #endif

        int n_output;
        if (current_step == -1){
            n_output = 0;
        } else if (current_step == -2){
            n_output = 9999;
        } else {
            n_output = current_step/interval_of_vsim + 1;
        }

        char out_vsim_part1[] = "vsm-";

        char out_vsim_part2[strlen(out_body)];
        strcpy(out_vsim_part2,out_body);

        char out_vsim_part3[6];
        out_vsim_part3[0] = '_';
        out_vsim_part3[1] =  n_output/1000      + '0';
        out_vsim_part3[2] = (n_output%1000)/100 + '0';
        out_vsim_part3[3] = (n_output%100)/10   + '0';
        out_vsim_part3[4] = (n_output%10)       + '0';
        out_vsim_part3[5] = '\0';

        char out_vsim_atom[256];
        strcpy(out_vsim_atom,out_vsim_part1);
        strcat(out_vsim_atom,out_vsim_part2);
        strcat(out_vsim_atom,out_vsim_part3);
        strcat(out_vsim_atom,".ascii");
        ofstream out_file_atom(out_vsim_atom);
        out_file_atom << setiosflags(ios::scientific) << setprecision(vsim_prec);

        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        char out_vsim_spin[256];
        strcpy(out_vsim_spin,out_vsim_part1);
        strcat(out_vsim_spin,out_vsim_part2);
        strcat(out_vsim_spin,out_vsim_part3);
        strcat(out_vsim_spin,".spin");
        ofstream out_file_spin(out_vsim_spin);
        out_file_spin << setiosflags(ios::scientific) << setprecision(vsim_prec);
        #endif

        char out_vsim_color[256];
        strcpy(out_vsim_color,out_vsim_part1);
        strcat(out_vsim_color,out_vsim_part2);
        strcat(out_vsim_color,out_vsim_part3);
        strcat(out_vsim_color,".dat");
        ofstream out_file_color(out_vsim_color);
        out_file_color << setiosflags(ios::scientific) << setprecision(vsim_prec);

        out_file_atom << natom        << " " << total_time << '\n';
        out_file_atom << d.xx << " " << d.yx << " " << d.yy << '\n';
        out_file_atom << d.zx << " " << d.zy << " " << d.zz << '\n';
        
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        out_file_spin << natom << '\n';
        #endif

        for (int i = 0 ; i < natom; ++i){
            struct atom_struct *atom_ptr;
            atom_ptr = first_atom_ptr + i;
            out_file_atom  << atom_ptr->r.x << " " << atom_ptr->r.y << " " << atom_ptr->r.z << " "
                           << atom_ptr->element << " "
                           #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                           << atom_ptr->p.x << " " << atom_ptr->p.y << " " << atom_ptr-> p.z
                           #endif
                           << '\n';

            #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
              #if defined magmom || defined SLDNC
                vector s = atom_ptr->m;
              #else
                vector s = atom_ptr->s;
              #endif
              double ss    = vec_length(s);;
              double theta = acos(s.z/ss)*180e0/Pi_num;
              double phi   = atan2(s.y,s.x)*180e0/Pi_num;
              out_file_spin << i+1 << " " << ss << " " << theta << " " << phi << '\n';
            #endif
            #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
            out_file_color << atom_ptr->pe << '\n';
            #endif
            #if defined SDH || SDHL
            out_file_color << ss << '\n';
            #endif

        }

        out_file_atom.close();
        #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
        out_file_spin.close();
        #endif
        out_file_color.close();

    }
}

#endif


