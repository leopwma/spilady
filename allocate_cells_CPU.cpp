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

#ifdef CPU

#include "spilady.h"

void allocate_cells_CPU(){


    #pragma omp parallel for
    for (int i = 0 ; i < ncells; ++i) (first_cell_ptr+i)->type = 0;
    
    struct cell_struct *cell_ptr;
    int no_of_groups = 0;
    int no_of_cells_left = ncells;

    max_no_of_members = 0;

    while(1){

        int no_of_members = 0;
        for (int i = 0 ; i < ncells; ++i){

            cell_ptr = first_cell_ptr + i;

            if (cell_ptr->type == 0 ){

                ++no_of_members;
                --no_of_cells_left;
                cell_ptr->type = 1;

                for (int j_neigh = 0 ; j_neigh < 26; ++j_neigh){

                    if ((first_cell_ptr+(cell_ptr->neigh_cell[j_neigh]))->type != 1)
                        (first_cell_ptr+(cell_ptr->neigh_cell[j_neigh]))->type = -1;

                }

            }
        }
        ++no_of_groups;

        #pragma omp parallel for
        for (int i = 0 ; i < ncells; ++i){
            if ((first_cell_ptr+i)->type == -1) (first_cell_ptr+i)->type = 0;
        }

        cout << "Group " << no_of_groups << "; No. of members = "<< no_of_members << "."<< '\n';
        if (no_of_members > max_no_of_members) max_no_of_members = no_of_members;
        if (no_of_cells_left == 0) break;

    }

    // no_of_groups is the total no. of group
    cout << "No. of groups for parallel run = " << no_of_groups << '\n';
    
    allocate_cell_ptr_ptr = (cell_struct**)malloc(no_of_groups*max_no_of_members*sizeof(cell_struct*));
    allocate_threads_ptr = (int*)malloc(no_of_groups*sizeof(int));


    #pragma omp parallel for
    for (int i = 0 ; i < ncells; ++i) (first_cell_ptr+i)->type = 0;
    
    no_of_groups = 0; //the no_of_groups is reinitialized
    no_of_cells_left = ncells;

    while(1){

        int no_of_members = 0;
        for (int i = 0 ; i < ncells; ++i){

            cell_ptr = first_cell_ptr + i;

            if ( cell_ptr->type == 0 ){

                *(allocate_cell_ptr_ptr + no_of_groups*max_no_of_members + no_of_members) = cell_ptr;
                cell_ptr->type = 1;
                ++no_of_members;
                --no_of_cells_left;

                for (int j_neigh = 0 ; j_neigh < 26; ++j_neigh){
                    if ((first_cell_ptr+(cell_ptr->neigh_cell[j_neigh]))->type != 1)
                        (first_cell_ptr+(cell_ptr->neigh_cell[j_neigh]))->type = -1;

                }
            }
            *(allocate_threads_ptr + no_of_groups) = no_of_members;
        }
        ++no_of_groups;

        #pragma omp parallel for
        for (int i = 0 ; i < ncells; ++i){
            if ((first_cell_ptr+i)->type == -1) (first_cell_ptr+i)->type = 0;
        }
        if (no_of_cells_left == 0) break;
    }

    ngroups = no_of_groups;
}

void free_allocate_memory_CPU(){

    free(allocate_cell_ptr_ptr);
    free(allocate_threads_ptr);

}

void allocate_cells(){
    allocate_cells_CPU();
}

void free_allocate_memory(){
    free_allocate_memory_CPU();
}

#endif
