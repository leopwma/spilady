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

#ifdef GPU

#include "spilady.h"
#include "prototype_GPU.h"

__global__ void initialize_random_number_GPU(curandState *rand_state_ptr_d, int random_seed){

    int id = blockIdx.x*blockDim.x + threadIdx.x;
    curand_init(random_seed, id, 0, rand_state_ptr_d + id);

}

void free_random_number_GPU(){

    cudaFree(rand_state_ptr_d);

}

void initialize_random_number(){

      cudaMalloc((void**)&rand_state_ptr_d, no_of_threads*no_of_blocks*sizeof(curandState));
      initialize_random_number_GPU<<<no_of_blocks, no_of_threads>>>(rand_state_ptr_d, random_seed);

}

void free_random_number(){
      free_random_number_GPU();
}



__device__ double normal_rand_d(curandState *state_ptr)
{
    return curand_normal(state_ptr);
}
#endif
