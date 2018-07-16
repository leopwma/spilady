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

void initial_links_CPU(){

    #pragma omp parallel for
    for (int i = 0 ; i < natom ; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;
         
        //initialize some variable for the following
        atom_ptr->this_atom_ptr = atom_ptr;
        atom_ptr->next_atom_ptr = NULL;
        atom_ptr->prev_atom_ptr = NULL;

        atom_ptr->old_cell_index = -1; // give the cell a fake value
        atom_ptr->new_cell_index = -2; // give the cell a fake value

        periodic(atom_ptr->r);
    }

    //Creating double link list system.
    for (int i = 0 ; i < natom ; ++i){
        struct atom_struct *atom_ptr;
        struct cell_struct *cell_ptr;

        atom_ptr = first_atom_ptr + i;

        vector q;
        q.x = Inv_d.xx*atom_ptr->r.x + Inv_d.yx*atom_ptr->r.y + Inv_d.zx*atom_ptr->r.z;
        q.y =                          Inv_d.yy*atom_ptr->r.y + Inv_d.zy*atom_ptr->r.z;
        q.z =                                                   Inv_d.zz*atom_ptr->r.z;

        int icell = int(q.x*no_of_link_cell_x)
                  + int(q.y*no_of_link_cell_y)*no_of_link_cell_x
                  + int(q.z*no_of_link_cell_z)*no_of_link_cell_x*no_of_link_cell_y;

        cell_ptr = first_cell_ptr + icell;
       
        atom_ptr->new_cell_index = icell; // On which cell that such atom is a member


        atom_ptr->prev_atom_ptr = NULL;
        atom_ptr->next_atom_ptr = cell_ptr->head_ptr;
        cell_ptr->head_ptr      = atom_ptr;
        ++(cell_ptr->no_of_atoms_in_cell);

        if (atom_ptr->next_atom_ptr == NULL)
            cell_ptr->tail_ptr = atom_ptr;
        else
            atom_ptr->next_atom_ptr->prev_atom_ptr = atom_ptr;

    }
}

void links_CPU(){

    int no_of_relink = 0; //no. of atoms that need to be relink;

    #pragma omp parallel for reduction(+:no_of_relink)
    for (int i = 0 ; i < natom ; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;

        atom_ptr->old_cell_index = atom_ptr->new_cell_index;
       
        vector q;
        q.x = Inv_d.xx*atom_ptr->r.x + Inv_d.yx*atom_ptr->r.y + Inv_d.zx*atom_ptr->r.z;
        q.y =                          Inv_d.yy*atom_ptr->r.y + Inv_d.zy*atom_ptr->r.z;
        q.z =                                                   Inv_d.zz*atom_ptr->r.z;

        int linkx = int(q.x*no_of_link_cell_x);
        int linky = int(q.y*no_of_link_cell_y);
        int linkz = int(q.z*no_of_link_cell_z);

        if (linkx < 0 ) linkx = 0;
        if (linky < 0 ) linky = 0;
        if (linkz < 0 ) linkz = 0;
        if (linkx >= no_of_link_cell_x) linkx = no_of_link_cell_x - 1;
        if (linky >= no_of_link_cell_y) linky = no_of_link_cell_y - 1;
        if (linkz >= no_of_link_cell_z) linkz = no_of_link_cell_z - 1;

        int icell = linkx
                  + linky*no_of_link_cell_x
                  + linkz*no_of_link_cell_x*no_of_link_cell_y;
      
        atom_ptr->new_cell_index = icell; // On which cell that such atom is a member
        if (icell != atom_ptr->old_cell_index)  no_of_relink += 1;
    }

    if (no_of_relink != 0) {

        int *delinking_atom_index_ptr[OMP_threads];
        int delinking_atom_number[OMP_threads];

        for (int i = 0; i < OMP_threads; ++i){
            delinking_atom_index_ptr[i]  = (int*)malloc(no_of_relink*sizeof(int));
            delinking_atom_number[i] = 0;
        }

        #pragma omp parallel for
        for (int i = 0 ; i < natom ; ++i){
            struct atom_struct *atom_ptr;
            atom_ptr = first_atom_ptr + i;

            if (atom_ptr->new_cell_index != atom_ptr->old_cell_index){
                int thread_index = omp_get_thread_num();
                *(delinking_atom_index_ptr[thread_index] + delinking_atom_number[thread_index]) = i;
                ++delinking_atom_number[thread_index];
            }

        }
      
        //Run in serial only.
        for (int i = 0 ; i < OMP_threads ; ++i){
            for (int j = 0; j < delinking_atom_number[i]; ++j){
                struct atom_struct *atom_ptr;
                atom_ptr = first_atom_ptr + *(delinking_atom_index_ptr[i] + j);
            
                //detach the previous and next atom pointers
                if (atom_ptr->prev_atom_ptr == NULL)
                    (first_cell_ptr + (atom_ptr->old_cell_index))->head_ptr = atom_ptr->next_atom_ptr;
                else
                    atom_ptr->prev_atom_ptr->next_atom_ptr = atom_ptr->next_atom_ptr;
 
                if (atom_ptr->next_atom_ptr == NULL)
                    (first_cell_ptr + (atom_ptr->old_cell_index))->tail_ptr = atom_ptr->prev_atom_ptr;
                else
                    atom_ptr->next_atom_ptr->prev_atom_ptr = atom_ptr->prev_atom_ptr;
 
                //attach to new cell at the front
                atom_ptr->prev_atom_ptr = NULL;
                atom_ptr->next_atom_ptr = (first_cell_ptr + (atom_ptr->new_cell_index))->head_ptr;
                (first_cell_ptr + (atom_ptr->new_cell_index))->head_ptr  = atom_ptr;
                   
                if (atom_ptr->next_atom_ptr == NULL)
                    (first_cell_ptr + (atom_ptr->new_cell_index))->tail_ptr = atom_ptr;
                else
                    atom_ptr->next_atom_ptr->prev_atom_ptr = atom_ptr;
 
                --((first_cell_ptr+(atom_ptr->old_cell_index))->no_of_atoms_in_cell);
                ++((first_cell_ptr+(atom_ptr->new_cell_index))->no_of_atoms_in_cell);
            }
        }
    
        for (int i = 0; i < OMP_threads; ++i) free(delinking_atom_index_ptr[i]);
    }
}

void initial_links(){
      initial_links_CPU();
}

void links(){
      links_CPU();
}

#endif
