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
*   Quantum thermostat
*   Hichem Dammal et al. Phy. Rev. Lett. 103, 190601 (2009)
*   Jean-Louis Barrat and David Rodney, J. Stat. Phys (2011) 144:679-689
*
********************************************************************************
*
*   Edit notes:
*   Date:    Oct 2015
*   Author:  Pui-Wai (Leo) MA
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) An error in void initial_quantum_noise() is fixed. 
*   The max_frequency should be calculated with only half time-step,
*   becasue the Suzuki-Trotter decomposition is used.
*   The time-step using in core_dp_CPU and core_dp_GPU, dt = step/2e0;
*
********************************************************************************
*
*   Edit notes:
*   Date:    Apr 2016
*   Author:  Pui-Wai (Leo) MA
*   Address: Culham Centre for Fusion Energy, OX14 3DB, United Kingdom
*   1) Added variable "Msteps_quantum"
*   2) Changed "#define Nf" and "#define Nf2" into 
       variables "Nfrequency_quantum" and "Nfrequency_quantum_2"
*
********************************************************************************/

#ifdef GPU

#include "spilady.h"
#include "prototype_GPU.h"

#ifdef quantumnoise
__device__ double quantum_noise_d(int n, int m,  double* quantum_rand_memory_ptr_d,
                                  curandState *rand_state_ptr_d, double* H_ptr_d,
                                  double* quantum_noise_ptr_d, int* quantum_count_ptr_d,
                                  int Msteps, int Nf2){

    //Msteps = Msteps_quantum
    //Nf2 = Nfrequency_quantum_2
  
    if ( (*(quantum_count_ptr_d + n)) % Msteps == 0){

        double noise = 0e0;

        for (int i = 0; i < Nf2; ++i)
            noise += *(H_ptr_d + i) * *(quantum_rand_memory_ptr_d + (Nf2 - 1) - i + n*Nf2);

        for (int i = 0; i < Nf2-1 ; ++i)
            *(quantum_rand_memory_ptr_d + i  + n*Nf2) = *(quantum_rand_memory_ptr_d + (i + 1) + n*Nf2) ;

        *(quantum_rand_memory_ptr_d + (Nf2 - 1) + n*Nf2) =  normal_rand_d(rand_state_ptr_d + m);

        *(quantum_noise_ptr_d + n) = noise;
    }

    ++(*(quantum_count_ptr_d + n));
    return *(quantum_noise_ptr_d + n);
}

void initial_quantum_noise(){

    double h = Msteps_quantum*step/2e0; // divided by 2e0 is because of Suzuki-Trotter decomposition.
    int n = 3*natom;
    double max_frequency = Pi_num/h;    // This maximum frequancy needs to match the time step,
                                        // so adaptive time-step cannot be used.
    int Nf = Nfrequency_quantum;
    int Nf2 = Nfrequency_quantum_2;

    H_ptr = (double*)malloc(Nf2*sizeof(double));

    double H_tilda[Nf2];
    double delta_frequency = max_frequency/double(Nf);
    for (int i = 0; i < Nf2 ; ++i){
        int k = i - Nf;
        double frequency = delta_frequency*double(k);
        H_tilda[i] = sqrt(hbar*fabs(frequency)*(0.5e0 + 1e0/expm1(hbar*fabs(frequency)/temperature)));
        H_tilda[i] *= (frequency*h/2e0)/sin(frequency*h/2e0);
        if (k == 0) H_tilda[i] = sqrt(temperature);
    }    
    for (int i = 0; i < Nf2 ; ++i){
        *(H_ptr + i) = 0e0;
        for (int j = 0; j < Nf2 ; ++j){
            int k = j - Nf;
            *(H_ptr + i) += H_tilda[j]*cos(Pi_num/double(Nf)*double(k)*double(i-Nf));
        }
        *(H_ptr + i) /= double(Nf2);
    }

    cudaMalloc((void**)&H_ptr_d, Nf2*sizeof(double));
    cudaMemcpy(H_ptr_d, H_ptr, Nf2*sizeof(double), cudaMemcpyHostToDevice);

    quantum_rand_memory_ptr = (double*)malloc(n*Nf2*sizeof(double));
    //for (int i = 0; i < n*Nf2; ++i) *(quantum_rand_memory_ptr + i) =  0e0; //just initialize
    for (int i = 0; i < n*Nf2; ++i) *(quantum_rand_memory_ptr + i) = (rand() % 1000)/500e0 - 1.0; //just initialize

    quantum_noise_ptr = (double*)malloc(n*sizeof(double));
    for (int i = 0; i < n; ++i) *(quantum_noise_ptr + i) = 0e0;

    quantum_count_ptr = (int*)malloc(n*sizeof(int));
    for (int i = 0; i < n; ++i) *(quantum_count_ptr + i) = 0;

    cudaMalloc((void**)&quantum_rand_memory_ptr_d, n*Nf2*sizeof(double));
    cudaMemcpy(quantum_rand_memory_ptr_d, quantum_rand_memory_ptr, n*Nf2*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&quantum_noise_ptr_d, n*sizeof(double));
    cudaMemcpy(quantum_noise_ptr_d, quantum_noise_ptr, n*sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&quantum_count_ptr_d, n*sizeof(int));
    cudaMemcpy(quantum_count_ptr_d, quantum_count_ptr, n*sizeof(int), cudaMemcpyHostToDevice);


}

void free_quantum_noise(){

      free(H_ptr);
      free(quantum_rand_memory_ptr);
      free(quantum_noise_ptr);
      free(quantum_count_ptr);

      cudaFree(H_ptr_d);
      cudaFree(quantum_rand_memory_ptr_d);
      cudaFree(quantum_noise_ptr);
      cudaFree(quantum_count_ptr);

}

#endif //quantumnoise

#endif //GPU

