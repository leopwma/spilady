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
*   Quantum thermostat:
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
*      variables "Nfrequency_quantum" and "Nfrequency_quantum_2"
*
********************************************************************************/

#ifdef CPU

#include "spilady.h"

#ifdef quantumnoise

double quantum_noise(int n, int thread_index){

    if ( (*(quantum_count_ptr + n)) % Msteps_quantum == 0){

        int Nf2 = Nfrequency_quantum_2;

        double noise = 0e0;

        for (int i = 0; i < Nf2; ++i)
            noise += *(H_ptr + i)* *(quantum_rand_memory_ptr + (Nf2 - i - 1) + n*Nf2);

        for (int i = 0; i < Nf2-1 ; ++i)
            *(quantum_rand_memory_ptr + i  + n*Nf2) = *(quantum_rand_memory_ptr + (i + 1) + n*Nf2) ;

        *(quantum_rand_memory_ptr + Nf2 - 1 + n*Nf2) =  normal_rand(thread_index);

        *(quantum_noise_ptr + n) = noise;
    }
    ++(*(quantum_count_ptr + n));

    return *(quantum_noise_ptr + n);

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

    #pragma omp parallel for
    for (int i = 0; i < Nf2 ; ++i){
        int k = i - Nf;
        double frequency = delta_frequency*double(k);
        H_tilda[i] = sqrt(hbar*fabs(frequency)*(0.5e0 + 1e0/expm1(hbar*fabs(frequency)/temperature)));
        H_tilda[i] *= (frequency*h/2e0)/sin(frequency*h/2e0);
        if (k == 0) H_tilda[i] = sqrt(temperature);
    }    
    #pragma omp parallel for
    for (int i = 0; i < Nf2 ; ++i){
        *(H_ptr + i) = 0e0;
        for (int j = 0; j < Nf2 ; ++j){
            int k = j - Nf;
            *(H_ptr + i) += H_tilda[j]*cos(Pi_num/double(Nf)*double(k)*double(i-Nf));
        }
        *(H_ptr + i) /= double(Nf2);
        //cout << *(H_ptr + i) << '\n'; 
        //check the last H[i] is small. 
        //At least equal 1e-5. Nf ~= 50 to 200 is enough.
    }

    quantum_rand_memory_ptr = (double*)malloc(n*Nf2*sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n*Nf2; ++i) *(quantum_rand_memory_ptr + i) =  normal_rand(omp_get_thread_num()); //just initialize

    quantum_noise_ptr = (double*)malloc(n*sizeof(double));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) *(quantum_noise_ptr + i) = 0e0;

    quantum_count_ptr = (int*)malloc(n*sizeof(int));
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) *(quantum_count_ptr + i) = 0;
}

void free_quantum_noise(){

      free(H_ptr);
      free(quantum_rand_memory_ptr);
      free(quantum_noise_ptr);
      free(quantum_count_ptr);
}

#endif //quantumnoise

#endif //GPU
