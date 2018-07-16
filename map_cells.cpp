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

void map_cells(){

    #if defined eltemp && defined readTe
    double temp_double;
    ifstream in_file(in_eltemp);
    in_file >> ws >> ncells >> total_time;
    in_file >> ws >> d.xx >> d.yx >> d.yy;
    in_file >> ws >> d.zx >> d.zy >> d.zz;
    #endif

    double angle1 = sin(atan2(d.yy,d.yx));
    double angle2 = sin(atan2(d.zz,d.zx));
    double angle3 = sin(atan2(d.zz,d.zy));

    double link_cell_edge_x = max(min_length_link_cell/angle1, min_length_link_cell/angle2);
    double link_cell_edge_y = max(min_length_link_cell, min_length_link_cell/angle3);
    double link_cell_edge_z = min_length_link_cell;

    no_of_link_cell_x = d.xx/link_cell_edge_x;
    no_of_link_cell_y = d.yy/link_cell_edge_y;
    no_of_link_cell_z = d.zz/link_cell_edge_z;

    #if defined eltemp && defined readTe
    if (ncells == no_of_link_cell_x*no_of_link_cell_y*no_of_link_cell_z) {
       ncells = no_of_link_cell_x*no_of_link_cell_y*no_of_link_cell_z;
    } else {
       cout << "no. of link cells error. Number of link cells in read-in file is different from calculated. Change min_length_link_cell." << '\n';
    }
    #else
    ncells = no_of_link_cell_x*no_of_link_cell_y*no_of_link_cell_z;
    #endif

    cout << "No. of link cells in the edges of box:" << '\n';
    cout << "In the x axis = " << no_of_link_cell_x << '\n';
    cout << "In the y axis = " << no_of_link_cell_y << '\n';
    cout << "In the z axis = " << no_of_link_cell_z << '\n';
    cout << "Total link cells = " << ncells << '\n';
    
    int n_depth  = no_of_link_cell_x;
    int n_height = no_of_link_cell_x*no_of_link_cell_y;

    first_cell_ptr = (cell_struct*)malloc(ncells*sizeof(cell_struct));

    #ifdef OMP
    #pragma omp parallel for
    #endif
    for (int ix = 0; ix < no_of_link_cell_x; ++ix){
        for (int iy = 0; iy < no_of_link_cell_y; ++iy){
            for (int iz = 0; iz < no_of_link_cell_z; ++iz){

                struct cell_struct *cell_ptr;
                cell_ptr = first_cell_ptr + ix + iy*n_depth + iz*n_height;

                cell_ptr->head_ptr = NULL;
                cell_ptr->this_ptr = NULL;
                cell_ptr->tail_ptr = NULL;
                cell_ptr->no_of_atoms_in_cell = 0;

                int ix_plus_1  = (ix + 1)%no_of_link_cell_x;
                int iy_plus_1  = (iy + 1)%no_of_link_cell_y;
                int iz_plus_1  = (iz + 1)%no_of_link_cell_z;
                int ix_minus_1 = (ix - 1 + no_of_link_cell_x)%no_of_link_cell_x;
                int iy_minus_1 = (iy - 1 + no_of_link_cell_y)%no_of_link_cell_y;
                int iz_minus_1 = (iz - 1 + no_of_link_cell_z)%no_of_link_cell_z;

                cell_ptr->neigh_cell[0] = ix_plus_1  + iy*n_depth         + iz*n_height;
                cell_ptr->neigh_cell[1] = ix_plus_1  + iy_plus_1*n_depth  + iz*n_height;
                cell_ptr->neigh_cell[2] = ix         + iy_plus_1*n_depth  + iz*n_height;
                cell_ptr->neigh_cell[3] = ix_minus_1 + iy_plus_1*n_depth  + iz*n_height;
                cell_ptr->neigh_cell[4] = ix_plus_1  + iy*n_depth         + iz_minus_1*n_height;
                cell_ptr->neigh_cell[5] = ix_plus_1  + iy_plus_1*n_depth  + iz_minus_1*n_height;
                cell_ptr->neigh_cell[6] = ix         + iy_plus_1*n_depth  + iz_minus_1*n_height;
                cell_ptr->neigh_cell[7] = ix_minus_1 + iy_plus_1*n_depth  + iz_minus_1*n_height;
                cell_ptr->neigh_cell[8] = ix_plus_1  + iy*n_depth         + iz_plus_1*n_height;
                cell_ptr->neigh_cell[9] = ix_plus_1  + iy_plus_1*n_depth  + iz_plus_1*n_height;
                cell_ptr->neigh_cell[10]= ix         + iy_plus_1*n_depth  + iz_plus_1*n_height;
                cell_ptr->neigh_cell[11]= ix_minus_1 + iy_plus_1*n_depth  + iz_plus_1*n_height;
                cell_ptr->neigh_cell[12]= ix         + iy*n_depth         + iz_plus_1*n_height;
                cell_ptr->neigh_cell[13]= ix_minus_1 + iy*n_depth         + iz*n_height;
                cell_ptr->neigh_cell[14]= ix         + iy_minus_1*n_depth + iz*n_height;
                cell_ptr->neigh_cell[15]= ix         + iy*n_depth         + iz_minus_1*n_height;
                cell_ptr->neigh_cell[16]= ix_plus_1  + iy_minus_1*n_depth + iz*n_height;
                cell_ptr->neigh_cell[17]= ix_minus_1 + iy_minus_1*n_depth + iz*n_height;
                cell_ptr->neigh_cell[18]= ix_minus_1 + iy*n_depth         + iz_plus_1*n_height;
                cell_ptr->neigh_cell[19]= ix_minus_1 + iy*n_depth         + iz_minus_1*n_height;
                cell_ptr->neigh_cell[20]= ix         + iy_minus_1*n_depth + iz_plus_1*n_height;
                cell_ptr->neigh_cell[21]= ix         + iy_minus_1*n_depth + iz_minus_1*n_height;
                cell_ptr->neigh_cell[22]= ix_plus_1  + iy_minus_1*n_depth + iz_plus_1*n_height;
                cell_ptr->neigh_cell[23]= ix_plus_1  + iy_minus_1*n_depth + iz_minus_1*n_height;
                cell_ptr->neigh_cell[24]= ix_minus_1 + iy_minus_1*n_depth + iz_plus_1*n_height;
                cell_ptr->neigh_cell[25]= ix_minus_1 + iy_minus_1*n_depth + iz_minus_1*n_height;
           
            }
        }
    }


    #ifdef eltemp
    for (int ix = 0; ix < no_of_link_cell_x; ++ix){
        for (int iy = 0; iy < no_of_link_cell_y; ++iy){
            for (int iz = 0; iz < no_of_link_cell_z; ++iz){

                struct cell_struct *cell_ptr;
                cell_ptr = first_cell_ptr + ix + iy*n_depth + iz*n_height;

                #ifdef readTe
                  in_file >> ws >> temp_double >> temp_double >> temp_double
                          >> cell_ptr->Te 
                          #if defined MD || defined SLDH || defined SLDHL || defined SLDNC                             
                          >> cell_ptr->Tl
                          #endif
                          #if defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC   
                          >> cell_ptr->Ts_R
                          #endif
                          #if defined SDHL || defined SLDHL
                          >> cell_ptr->Ts_L
                          #endif
                          ;
                  cout << ix << " " << iy << " " << iz << " " << cell_ptr->Te << '\n';
                  cell_ptr->Te *= boltz;
                #else
                  cell_ptr->Te = temperature;
                #endif
                cell_ptr->Ee = Te_to_Ee(cell_ptr->Te);
            
            }
         }
    }
    #endif

    #if defined eltemp && defined readTe
    in_file.close();
    cout << "Read in link cells electron temperatures is completed." << '\n';
    #endif
}
