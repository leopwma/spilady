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

void write_config(int current_step){

    if (current_step == -1 || current_step == -2 || (current_step + 1)%interval_of_config_out == 0 ){

        #ifdef GPU
        copy_atoms_from_GPU_to_CPU();
        #endif

        int n_output;
        if (current_step == -1){
            n_output = 0;
        } else if (current_step == -2){
            n_output = 9999;
        } else {
            n_output = current_step/interval_of_config_out + 1;
        }

        char out_config_part1[] = "con-";

        char out_config_part2[strlen(out_body)]; // strlen(".dat") = 4
        strcpy(out_config_part2,out_body);
        
        char out_config_part3[6];
        out_config_part3[0] = '_';
        out_config_part3[1] =  n_output/1000      + '0';
        out_config_part3[2] = (n_output%1000)/100 + '0';
        out_config_part3[3] = (n_output%100)/10   + '0';
        out_config_part3[4] = (n_output%10)       + '0';
        out_config_part3[5] = '\0';

        char out_config[256];
        strcpy(out_config,out_config_part1);
        strcat(out_config,out_config_part2);
        strcat(out_config,out_config_part3);
        strcat(out_config,".dat");

        ofstream out_file(out_config);
        out_file << setiosflags(ios::scientific) << setprecision(15);

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

        out_file << natom << " " << total_time << " " << current_type << '\n';
        out_file << d.xx << " " << d.yx << " " << d.yy  << '\n' ;
        out_file << d.zx << " " << d.zy << " " << d.zz  << '\n' ;

        struct atom_struct *atom_ptr;

        for (int i = 0 ; i < natom; ++i){
            atom_ptr = first_atom_ptr + i;
            out_file << i << " " << atom_ptr->element << " "
                     << atom_ptr->r.x << " " << atom_ptr->r.y << " " << atom_ptr->r.z << " "
                     #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                     << atom_ptr->p.x << " " << atom_ptr->p.y << " " << atom_ptr->p.z << " "
                     << atom_ptr->f.x << " " << atom_ptr->f.y << " " << atom_ptr->f.z << " "
                     << atom_ptr->stress11 << " "
                     << atom_ptr->stress22 << " "
                     << atom_ptr->stress33 << " "
                     << atom_ptr->stress12 << " "
                     << atom_ptr->stress23 << " "
                     << atom_ptr->stress31 << " "
                     << atom_ptr->rho << " "
                     << atom_ptr->ke << " "
                     << atom_ptr->pe << " "
                     #endif

                     #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                       #if defined magmom || defined SLDNC
                       << atom_ptr->m.x << " " << atom_ptr->m.y << " " << atom_ptr->m.z << " " << atom_ptr->m0 << " "
                       #else
                       << atom_ptr->s.x << " " << atom_ptr->s.y << " " << atom_ptr->s.z << " " << atom_ptr->s0 << " "
                       #endif

                       << atom_ptr->Heff_H.x << " " << atom_ptr->Heff_H.x << " " << atom_ptr->Heff_H.z << " "

                       #if defined SDHL || defined SLDHL
                       << atom_ptr->Heff_L.x << " " << atom_ptr->Heff_L.x << " " << atom_ptr->Heff_L.z << " "
                       #endif

                       #if defined SDH || defined SDHL || defined SLDH || defined SLDHL
                       << atom_ptr->me << " "
                       << atom_ptr->me0 << " "
                       #endif
                     #endif

                     << atom_ptr->local_volume
                     << '\n';
                     
        }
        out_file.close();
        
        #ifdef eltemp
        
        #ifdef GPU
        copy_cells_from_GPU_to_CPU();
        #endif

        char out_cell_part1[] = "cel-";

        char out_cell_part2[strlen(out_body)]; // strlen(".dat") = 4
        strcpy(out_cell_part2,out_body);

        char out_cell_part3[6];
        out_cell_part3[0] = '_';
        out_cell_part3[1] =  n_output/1000      + '0';
        out_cell_part3[2] = (n_output%1000)/100 + '0';
        out_cell_part3[3] = (n_output%100)/10   + '0';
        out_cell_part3[4] = (n_output%10)       + '0';
        out_cell_part3[5] = '\0';

        char out_cell[256];
        strcpy(out_cell,out_cell_part1);
        strcat(out_cell,out_cell_part2);
        strcat(out_cell,out_cell_part3);
        strcat(out_cell,".dat");

        ofstream out_file_cell(out_cell);
        out_file_cell << setiosflags(ios::scientific) << setprecision(15);
        
        out_file_cell << ncells << " " << total_time << '\n' ;
        out_file_cell << d.xx << " " << d.yx << " " << d.yy  << '\n' ;
        out_file_cell << d.zx << " " << d.zy << " " << d.zz  << '\n' ;
        for (int ix = 0; ix < no_of_link_cell_x; ++ix){
            for (int iy = 0; iy < no_of_link_cell_y; ++iy){
                for (int iz = 0; iz < no_of_link_cell_z; ++iz){

                struct cell_struct *cell_ptr;
                cell_ptr = first_cell_ptr + ix
                                          + iy*no_of_link_cell_x
                                          + iz*no_of_link_cell_x*no_of_link_cell_y;

                vector cell_q;
                cell_q.x = (ix+0.5)/no_of_link_cell_x;
                cell_q.y = (iy+0.5)/no_of_link_cell_y;
                cell_q.z = (iz+0.5)/no_of_link_cell_z; 

                vector cell_r;
                cell_r.x = d.xx * cell_q.x + d.yx * cell_q.y + d.zx * cell_q.z;
                cell_r.y =                   d.yy * cell_q.y + d.zy * cell_q.z;
                cell_r.z =                                     d.zz * cell_q.z;

                out_file_cell << cell_r.x << " " << cell_r.y << " " << cell_r.z << " "
                              << cell_ptr->Te/boltz << " "
                              #if defined MD || defined SLDH || defined SLDHL || defined SLDNC
                              << cell_ptr->Tl << " "
                              #endif
                              #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC
                              << cell_ptr->Ts_R << " "
                              #endif
                              #if defined SDHL || defined SLDHL
                              << cell_ptr->Ts_L << " "
                              #endif
                              << '\n';


                }
            }
        }
        out_file_cell.close();
        
        #endif
    }
}




