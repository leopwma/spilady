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
*********************************************************************************
*
*   Edit notes:
*   Date:    Apr 2016
*   Author:  Pui-Wai (Leo) MA
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) variable "Msteps_quantum" is added.
*
********************************************************************************/

#if (defined MD || defined SLDH || defined SLDHL || defined SLDNC) && defined CPU

#include "spilady.h"

#ifdef lattlang
void core_dp_A(double dt);
void core_dp_B(double dt);
#ifdef localcolmot
void core_dp_C1(double dt);
void core_dp_C2(double dt);
#endif
#endif
void rescale_momentum();

void core_dp_CPU(double dt){

    #ifdef lattlang
      #ifdef localcolmot
      core_dp_C1(dt/2e0);  //subtract average momentum in a cell
      #endif
      core_dp_B(dt/2e0);   // solution of dp/dt = -gamma/mass*p
      core_dp_A(dt);       // add (forces + noise)*dt and substract average noise in a cel ifdef localcolmot
      core_dp_B(dt/2e0);   // solution of dp/dt = -gamma/mass*p
      #ifdef localcolmot
      core_dp_C2(dt/2e0);  //subtract average momentum in a cell
      #endif
    #else
      #pragma omp parallel for
      for (int i = 0; i < natom; ++i){
          struct atom_struct *atom_ptr;
          atom_ptr = first_atom_ptr + i;

          #ifdef extforce
          atom_ptr->f = vec_add(atom_ptr->f, atom_ptr->fext);
          #endif

          atom_ptr->p = vec_add(atom_ptr->p, vec_times(dt,atom_ptr->f));
      }
    #endif

    rescale_momentum(); // make sure the total linear momentum = 0, and rescale it if necessary, due to numerical error 

    #pragma omp parallel for
    for (int i = 0; i < natom ; ++i){
         struct atom_struct *atom_ptr;
         atom_ptr = first_atom_ptr + i;
         atom_ptr->ke = vec_sq(atom_ptr->p)/2e0/atmass;
    }
}

void core_dp(double dt){
    core_dp_CPU(dt);
}



#ifdef lattlang
void core_dp_A(double dt){

    #ifdef localcolmot
    for (int i = 0 ; i < ngroups ; ++i){
        #pragma omp parallel for
        for (int j = 0 ; j < *(allocate_threads_ptr+i); ++j){
            struct atom_struct *atom_ptr;
            atom_ptr = (*(allocate_cell_ptr_ptr + i*max_no_of_members + j))->head_ptr;

            struct cell_struct *cell_ptr;

            bool ave_activated = 0;

            if (atom_ptr != NULL){
                cell_ptr = first_cell_ptr + atom_ptr->new_cell_index;
                cell_ptr->ave_fluct_force = vec_zero();
                ave_activated = 1;
            }

            while(atom_ptr != NULL){

    #else

    #pragma omp parallel for
    for (int i = 0; i < natom ; ++i){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;

        #if defined localcolmot || defined eltemp
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + atom_ptr->new_cell_index;
        #endif

    #endif //localcolmot

                #ifdef extforce
                atom_ptr->f = vec_add(atom_ptr->f, atom_ptr->fext);
                #endif
                //generating random numbers
                int thread_index = omp_get_thread_num();
                vector fluct_force;

                #ifdef quantumnoise
                  double h = dt*Msteps_quantum;
                  double fluct_force_length = sqrt(2e0*gamma_L/h);
                  int n = atom_ptr - first_atom_ptr;
                  int n1 = 3*n;
                  int n2 = 3*n+1;
                  int n3 = 3*n+2;
                  fluct_force.x = quantum_noise(n1, thread_index);
                  fluct_force.y = quantum_noise(n2, thread_index);
                  fluct_force.z = quantum_noise(n3, thread_index);
                #else
                  #ifdef eltemp
                  double fluct_force_length = sqrt(2e0*cell_ptr->Te*gamma_L/dt);
                  #else
                  double fluct_force_length = sqrt(2e0*temperature*gamma_L/dt);
                  #endif
                  fluct_force.x = normal_rand(thread_index);
                  fluct_force.y = normal_rand(thread_index);
                  fluct_force.z = normal_rand(thread_index);
                #endif

                fluct_force = vec_times(fluct_force_length, fluct_force);
                atom_ptr->p = vec_add(atom_ptr->p, vec_times(dt, vec_add(atom_ptr->f, fluct_force)));

        #ifdef localcolmot
                cell_ptr->ave_fluct_force = vec_add(cell_ptr->ave_fluct_force, fluct_force);
                atom_ptr = atom_ptr->next_atom_ptr;
            }
            if (ave_activated)
                cell_ptr->ave_fluct_force = vec_divide(cell_ptr->ave_fluct_force, cell_ptr->no_of_atoms_in_cell);
        }
        #endif
        
    }
    
    #ifdef localcolmot
    #pragma omp parallel for
    for (int i = 0; i < natom ; ++i){

        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr + i;

        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + atom_ptr->new_cell_index;

        atom_ptr->p = vec_sub(atom_ptr->p, vec_times(dt, cell_ptr->ave_fluct_force));
    }

    #endif


}

void core_dp_B(double dt){

    #pragma omp parallel for
    for (int i = 0; i < natom; ++i){
        struct atom_struct *atom_ptr;
        atom_ptr = first_atom_ptr+i;

        #ifdef localcolmot
        struct cell_struct *cell_ptr;
        cell_ptr = first_cell_ptr + atom_ptr->new_cell_index;
        double exp_dt = exp(-gamma_L/atmass*dt*(1e0-1e0/cell_ptr->no_of_atoms_in_cell));
        #else
        double exp_dt = exp(-gamma_L/atmass*dt);
        #endif

        atom_ptr->p = vec_times(exp_dt, atom_ptr->p);
    }
}


#ifdef localcolmot
void core_dp_C1(double dt){

    for (int i = 0 ; i < ngroups ; ++i){
        #pragma omp parallel for
        for (int j = 0 ; j < *(allocate_threads_ptr+i); ++j){
            struct atom_struct *atom_ptr;
            atom_ptr = (*(allocate_cell_ptr_ptr + i*max_no_of_members + j))->head_ptr;
            struct cell_struct *cell_ptr;
            cell_ptr = first_cell_ptr + atom_ptr->new_cell_index;
            while(atom_ptr != NULL){
                vector sum_p;
                sum_p = vec_zero();
                struct atom_struct *work_ptr;
                work_ptr = (*(allocate_cell_ptr_ptr + i*max_no_of_members + j))->head_ptr;
                while(work_ptr != NULL){
                    sum_p = vec_add(sum_p, work_ptr->p);
                    work_ptr = work_ptr->next_atom_ptr;
                }
                double factor = gamma_L/atmass*dt/(first_cell_ptr + atom_ptr->new_cell_index)->no_of_atoms_in_cell;
                
                atom_ptr->p = vec_add(atom_ptr->p, vec_times(factor, vec_sub(sum_p, atom_ptr->p)));
                atom_ptr = atom_ptr->next_atom_ptr;
            }
        }
    }
}

void core_dp_C2(double dt){

    for (int i = ngroups - 1 ; i >=0 ; --i){
        #pragma omp parallel for
        for (int j = 0 ; j < *(allocate_threads_ptr+i); ++j){
            struct atom_struct *atom_ptr;
            atom_ptr = (*(allocate_cell_ptr_ptr + i*max_no_of_members + j))->tail_ptr;
            struct cell_struct *cell_ptr;
            cell_ptr = first_cell_ptr + atom_ptr->new_cell_index;
            while(atom_ptr != NULL){
                vector sum_p;
                sum_p = vec_zero();
                struct atom_struct *work_ptr;
                work_ptr = (*(allocate_cell_ptr_ptr + i*max_no_of_members + j))->head_ptr;
                while(work_ptr != NULL){
                    sum_p = vec_add(sum_p, work_ptr->p);
                    work_ptr = work_ptr->next_atom_ptr;
                }
                double factor = gamma_L/atmass*dt/(first_cell_ptr + atom_ptr->new_cell_index)->no_of_atoms_in_cell;

                atom_ptr->p = vec_add(atom_ptr->p, vec_times(factor, vec_sub(sum_p, atom_ptr->p)));
                atom_ptr = atom_ptr->prev_atom_ptr;
            }
        }
    }

}
#endif
#endif

void rescale_momentum(){

    double ave_p_x = 0e0;
    double ave_p_y = 0e0;
    double ave_p_z = 0e0;
    #pragma omp parallel for reduction(+:ave_p_x,ave_p_y,ave_p_z)
    for (int i = 0; i < natom ; ++i){
         struct atom_struct *atom_ptr;
         atom_ptr = first_atom_ptr + i;
         ave_p_x += atom_ptr->p.x;
         ave_p_y += atom_ptr->p.y;
         ave_p_z += atom_ptr->p.z;
    }
    ave_p_x /= natom;
    ave_p_y /= natom;
    ave_p_z /= natom;

    double total_ke_old = 0e0;
    double total_ke_new = 0e0;
    #pragma omp parallel for reduction(+:total_ke_old,total_ke_new)
    for (int i = 0; i < natom ; ++i){
         struct atom_struct *atom_ptr;
         atom_ptr = first_atom_ptr + i;

         atom_ptr->ke = vec_sq(atom_ptr->p)/2e0/atmass;
         total_ke_old += atom_ptr->ke;

         atom_ptr->p.x -= ave_p_x;
         atom_ptr->p.y -= ave_p_y;
         atom_ptr->p.z -= ave_p_z;

         atom_ptr->ke = vec_sq(atom_ptr->p)/2e0/atmass;
         total_ke_new += atom_ptr->ke;
    }

    double factor = sqrt(total_ke_old/total_ke_new);
    if(total_ke_new < 1e-10) factor = 1e0;

    #pragma omp parallel for
    for (int i = 0; i < natom ; ++i){
         struct atom_struct *atom_ptr;
         atom_ptr = first_atom_ptr + i;
         atom_ptr->p = vec_times(factor,atom_ptr->p);
    }
}

#endif
