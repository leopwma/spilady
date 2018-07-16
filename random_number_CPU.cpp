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
********************************************************************************
*
* This thread safe random number generator is written according to George 
* Marsaglia's MWC (multiply with carry) algorithm, and Box-Muller transform.
*
********************************************************************************/

#ifdef CPU

#include "spilady.h"

void initial_random_number_MWC(){

    m_w_ptr = (uint*)malloc(OMP_threads*sizeof(uint));
    m_z_ptr = (uint*)malloc(OMP_threads*sizeof(uint));
    spare_rand_ptr = (int*)malloc(OMP_threads*sizeof(int));
    rand1_ptr = (double*)malloc(OMP_threads*sizeof(double));
    rand2_ptr = (double*)malloc(OMP_threads*sizeof(double));
    
    for(int i = 0; i < OMP_threads; ++i) *(m_w_ptr + i) = i + random_seed;
    for(int i = 0; i < OMP_threads; ++i) *(m_z_ptr + i) = i*3 + random_seed*2;
    for(int i = 0; i < OMP_threads; ++i) *(spare_rand_ptr + i) = 0;
}

double get_uniform(int thread_index){

    uint m_z = *(m_z_ptr + thread_index);
    uint m_w = *(m_w_ptr + thread_index);

    m_z = 36969*(m_z & 65535) + (m_z >> 16);
    m_w = 18000*(m_w & 65535) + (m_w >> 16);

    uint un = (m_z << 16) + m_w;
    
    *(m_z_ptr + thread_index) = m_z;
    *(m_w_ptr + thread_index) = m_w;

    return (un + 1.0)*2.328306435454494e-10;
}

double normal_Box_Muller(int thread_index){

    if(*(spare_rand_ptr + thread_index) == 1){

        *(spare_rand_ptr + thread_index) = 0;
	return sqrt(*(rand1_ptr + thread_index))*sin(*(rand2_ptr + thread_index));

    } else {

        *(spare_rand_ptr + thread_index) = 1;
	*(rand1_ptr + thread_index) = get_uniform(thread_index);
	*(rand1_ptr + thread_index) = -2e0*log(*(rand1_ptr + thread_index));
	*(rand2_ptr + thread_index) = get_uniform(thread_index)*6.283185307179586;
	return sqrt(*(rand1_ptr + thread_index))*cos(*(rand2_ptr + thread_index));
    }

}

void initialize_random_number_CPU(){

   initial_random_number_MWC();

}

void free_random_number_CPU(){

    free(m_w_ptr);
    free(m_z_ptr);
    free(spare_rand_ptr);
    free(rand1_ptr);
    free(rand2_ptr);

}

double normal_rand(int thread_index){

    return normal_Box_Muller(thread_index);
}


void initialize_random_number(){
      initialize_random_number_CPU();
}

void free_random_number(){
      free_random_number_CPU();
}

#endif
