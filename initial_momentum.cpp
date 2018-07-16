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

#if defined MD ||  defined SLDH || defined SLDHL || defined SLDNC

#include "spilady.h"

#if defined initmomentum

void initial_momentum(){

    #ifndef readconf
      for (int i = 0; i < natom; ++i){
          struct atom_struct *atom_ptr;
          atom_ptr = first_atom_ptr + i;
          atom_ptr->f = vec_zero();
      }
    #endif

    #ifdef lattlang
      for (int i = 0; i < natom; ++i){
          struct atom_struct *atom_ptr;
          atom_ptr = first_atom_ptr + i;
          atom_ptr->p = vec_zero();
      }
    #else          

      int mseed = 39;  
      srand(mseed);
          
      for (int i = 0; i < natom; ++i){
          struct atom_struct *atom_ptr;
          atom_ptr = first_atom_ptr + i;
          atom_ptr->p = vec_init(rand(),rand(),rand());
      }

      //scale velocities so that total linear momentum is zero
      vector ave_p;
      ave_p = vec_zero();

      for (int i = 0; i < natom ; ++i){
          struct atom_struct *atom_ptr;
          atom_ptr = first_atom_ptr + i;
          ave_p = vec_add(ave_p, atom_ptr->p);
      }
      ave_p = vec_times(1e0/natom,ave_p);

      for (int i = 0; i < natom ; ++i){
          struct atom_struct *atom_ptr;
          atom_ptr = first_atom_ptr + i;
          atom_ptr->p = vec_sub(atom_ptr->p, ave_p);
      }
 
      //scale velocities to set-point temperature
      scale_temperature();
    #endif

}


#endif
#endif
