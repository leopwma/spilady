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

#if (defined SDH || defined SDHL || defined SLDH || defined SLDHL || defined SLDNC) && defined GPU

#include "spilady.h"
#include "prototype_GPU.h"

/****************************************************************************
* GPU codes only
****************************************************************************/

__device__ void calculate_spin_d(int j, curandState *rand_state_ptr_d,
                          struct varGPU *var_ptr_d,
                          struct atom_struct *atom_ptr,
                          struct cell_struct *first_cell_ptr_d,
                          double dt, double *Jij_ptr_d
                          #if defined SDHL || defined SLDHL
                          , double *LandauA_ptr_d
                          , double *LandauB_ptr_d
                          , double *LandauC_ptr_d
                          , double *LandauD_ptr_d
                          #endif
                          )
{


    #ifdef extfield
    atom_ptr->Heff_H = atom_ptr->Hext;
    #else
    atom_ptr->Heff_H = vec_zero_d();
    #endif
    #if defined SDHL || defined SLDHL
    atom_ptr->Heff_L = vec_zero_d();
    #endif

    inner_spin_d(var_ptr_d, atom_ptr, first_cell_ptr_d, Jij_ptr_d);//calculate the effective field of current atom

    #ifndef spinlang
    //exact solution; no thermostat; PRL 86, 898 (2001) I. P. Omelyan
    atom_ptr->s = spin_rotation_d(atom_ptr->Heff_H, atom_ptr->s, dt);
    #endif /*no spin langevin*/

    #ifdef spinlang

      double dt_half = dt/2e0;

      #if defined SDH || defined SLDH
      //Pui-Wai Ma and S. L. Dudarev, PHYSICAL REVIEW B 83, 134418 (2011)
      //There are 3 parts. Deterministic -> Stochastic -> Deterministic

      vector s_temp = atom_ptr->s;

      //1st part
      vector s_cross_Heff = vec_cross_d(s_temp, atom_ptr->Heff_H);
      double Heff_H0 = vec_length_d(atom_ptr->Heff_H);

      double cos_a  = cos(Heff_H0*(dt_half/hbar));
      double sin_a  = sin(Heff_H0*(dt_half/hbar));
      double exp_b  = exp(-Heff_H0*atom_ptr->s0*var_ptr_d->gamma_S_H*(dt_half/hbar));
      double exp_2b = exp_b*exp_b;
      double normalized_S_dot_H = 0e0;
      if (atom_ptr->s0 > 0e0 && Heff_H0 > 0e0)
          normalized_S_dot_H = vec_dot_d(s_temp, atom_ptr->Heff_H)/atom_ptr->s0/Heff_H0;

      double denominator = (1e0 + exp_2b + normalized_S_dot_H*(1e0 - exp_2b))*Heff_H0;
      double factor      = (1e0 - exp_2b + normalized_S_dot_H*(1e0 + exp_2b - 2e0*cos_a*exp_b))*atom_ptr->s0;

      atom_ptr->s = vec_divide_d( vec_add_d(vec_add_d(vec_times_d(2e0*cos_a*exp_b*Heff_H0, s_temp),
                                                      vec_times_d(2e0*sin_a*exp_b, s_cross_Heff)),
                                            vec_times_d(factor, atom_ptr->Heff_H)), denominator);

      //2nd part
      #ifdef eltemp
      double random_h = sqrt(2e0*(first_cell_ptr_d+(atom_ptr->new_cell_index))->Te*var_ptr_d->gamma_S_H*hbar/dt);
      #else
      double random_h = sqrt(2e0*var_ptr_d->temperature*var_ptr_d->gamma_S_H*hbar/dt);
      #endif
      vector dh;
      dh.x = random_h*normal_rand_d(rand_state_ptr_d + j);
      dh.y = random_h*normal_rand_d(rand_state_ptr_d + j);
      dh.z = random_h*normal_rand_d(rand_state_ptr_d + j);

      atom_ptr->s = spin_rotation_d(dh, atom_ptr->s, dt);

      //3rd part
      s_temp = atom_ptr->s;
      s_cross_Heff = vec_cross_d(s_temp, atom_ptr->Heff_H);

      normalized_S_dot_H = 0e0;
      if (atom_ptr->s0 > 0e0 && Heff_H0 > 0e0)
          normalized_S_dot_H = vec_dot_d(s_temp, atom_ptr->Heff_H)/atom_ptr->s0/Heff_H0;

      denominator = (1e0 + exp_2b + normalized_S_dot_H*(1e0 - exp_2b))*Heff_H0;
      factor      = (1e0 - exp_2b + normalized_S_dot_H*(1e0 + exp_2b - 2e0*cos_a*exp_b))*atom_ptr->s0;

      atom_ptr->s = vec_divide_d( vec_add_d(vec_add_d(vec_times_d(2e0*cos_a*exp_b*Heff_H0, s_temp),
                                                      vec_times_d(2e0*sin_a*exp_b, s_cross_Heff)),
                                            vec_times_d(factor, atom_ptr->Heff_H)), denominator);

      #endif

      #if defined SDHL || defined SLDHL
      // In 5 parts.
      //part 1
      atom_ptr->s = spin_rotation_d(atom_ptr->Heff_H, atom_ptr->s, dt_half);

      //part 2
      double dt_quad = dt/4e0;
      #ifdef SLDHL
      double A = LandauA_d(atom_ptr->rho, LandauA_ptr_d, var_ptr_d);
      double B = LandauB_d(atom_ptr->rho, LandauB_ptr_d, var_ptr_d);
      double C = LandauC_d(atom_ptr->rho, LandauC_ptr_d, var_ptr_d);
      double D = LandauD_d(atom_ptr->rho, LandauD_ptr_d, var_ptr_d);
      #else
      double A = LandauA_d(1, LandauA_ptr_d, var_ptr_d);
      double B = LandauB_d(1, LandauB_ptr_d, var_ptr_d);
      double C = LandauC_d(1, LandauC_ptr_d, var_ptr_d);
      double D = LandauD_d(1, LandauD_ptr_d, var_ptr_d);
      #endif
      double s_sq;

      //RK2
      s_sq = vec_sq_d(atom_ptr->s);
      atom_ptr->Heff_L = vec_times_d(-(2e0*A + 4e0*B*s_sq + 6e0*C*pow(s_sq,2) + 8e0*D*pow(s_sq,3)), atom_ptr->s);
      #ifdef SLDHL
        atom_ptr->sum_Jij_sj = 0e0;
        inner_sum_Jij_sj_d(var_ptr_d, atom_ptr, first_cell_ptr_d, Jij_ptr_d);
        atom_ptr->Heff_HC = vec_zero_d();
        double s0;
        s0 = sqrt(s_sq);     
        if (s0 > 0e0) atom_ptr->Heff_HC = vec_times_d(-atom_ptr->sum_Jij_sj/s0, atom_ptr->s);
        atom_ptr->Heff_L = vec_add_d(atom_ptr->Heff_L, atom_ptr->Heff_HC);
      #endif
      vector s_temp = vec_add_d(atom_ptr->s, vec_times_d(var_ptr_d->gamma_S_HL*dt_quad, vec_add_d(atom_ptr->Heff_H, atom_ptr->Heff_L)));
      s_sq = vec_sq_d(s_temp);
      atom_ptr->Heff_L = vec_times_d(-(2e0*A + 4e0*B*s_sq + 6e0*C*pow(s_sq,2) + 8e0*D*pow(s_sq,3)), s_temp);
      #ifdef SLDHL
        atom_ptr->Heff_HC = vec_zero_d();
        s0 = sqrt(s_sq);
        if (s0 > 0e0) atom_ptr->Heff_HC = vec_times_d(-atom_ptr->sum_Jij_sj/s0, s_temp);
        atom_ptr->Heff_L = vec_add_d(atom_ptr->Heff_L, atom_ptr->Heff_HC);
      #endif
      atom_ptr->s = vec_add_d(atom_ptr->s, vec_times_d(var_ptr_d->gamma_S_HL*dt_half, vec_add_d(atom_ptr->Heff_H, atom_ptr->Heff_L)));

      //part 3
      #ifdef eltemp
        double random_S = sqrt(2e0*(first_cell_ptr_d+(atom_ptr->new_cell_index))->Te*var_ptr_d->gamma_S_HL/dt);
      #else
        double random_S = sqrt(2e0*var_ptr_d->temperature*var_ptr_d->gamma_S_HL/dt);
      #endif
      vector dS;
      dS.x = random_S*normal_rand_d(rand_state_ptr_d + j);
      dS.y = random_S*normal_rand_d(rand_state_ptr_d + j);
      dS.z = random_S*normal_rand_d(rand_state_ptr_d + j);
      atom_ptr->s = vec_add_d(atom_ptr->s, vec_times_d(dt, dS));

      //part 4; RK2
      s_sq = vec_sq_d(atom_ptr->s);
      atom_ptr->Heff_L = vec_times_d(-(2e0*A + 4e0*B*s_sq + 6e0*C*pow(s_sq,2) + 8e0*D*pow(s_sq,3)), atom_ptr->s);
      #ifdef SLDHL
        atom_ptr->Heff_HC = vec_zero_d();
        s0 = sqrt(s_sq);
        if (s0 > 0e0) atom_ptr->Heff_HC = vec_times_d(-atom_ptr->sum_Jij_sj/s0, atom_ptr->s);
        atom_ptr->Heff_L = vec_add_d(atom_ptr->Heff_L, atom_ptr->Heff_HC);
      #endif
      s_temp = vec_add_d(atom_ptr->s, vec_times_d(var_ptr_d->gamma_S_HL*dt_quad, vec_add_d(atom_ptr->Heff_H, atom_ptr->Heff_L)));

      s_sq = vec_sq_d(s_temp);
      atom_ptr->Heff_L = vec_times_d(-(2e0*A + 4e0*B*s_sq + 6e0*C*pow(s_sq,2) + 8e0*D*pow(s_sq,3)), s_temp);
      #ifdef SLDHL
        atom_ptr->Heff_HC = vec_zero_d();
        s0 = sqrt(s_sq);
        if (s0 > 0e0) atom_ptr->Heff_HC = vec_times_d(-atom_ptr->sum_Jij_sj/s0, s_temp);
        atom_ptr->Heff_L = vec_add_d(atom_ptr->Heff_L, atom_ptr->Heff_HC);
      #endif
      atom_ptr->s = vec_add_d(atom_ptr->s, vec_times_d(var_ptr_d->gamma_S_HL*dt_half, vec_add_d(atom_ptr->Heff_H, atom_ptr->Heff_L)));

      //part 5
      atom_ptr->s = spin_rotation_d(atom_ptr->Heff_H, atom_ptr->s, dt_half);

      #endif
    #endif /*spinlang*/

    atom_ptr->s0 = vec_length_d(atom_ptr->s);
}

__device__ vector spin_rotation_d(vector Heff, vector s, double dt){

    vector omega = vec_divide_d(Heff, -hbar);
    double omega0 = vec_length_d(omega);
    if (omega0 > 0e0){
        omega = vec_divide_d(omega, omega0);
    } else {
        omega = vec_zero_d();
    }
    double omega_12 = omega.x*omega.y;
    double omega_23 = omega.y*omega.z;
    double omega_13 = omega.x*omega.z;

    double omega1_sq = omega.x*omega.x;
    double omega2_sq = omega.y*omega.y;
    double omega3_sq = omega.z*omega.z;

    double A = sin(omega0*dt);
    double B = 1e0 - cos(omega0*dt);

    vector s_temp;
    s_temp.x = s.x
             + (s.x*B*(-omega2_sq - omega3_sq)
             +  s.y*(B*omega_12 - A*omega.z)
             +  s.z*(A*omega.y + B*omega_13));
    s_temp.y = s.y
             + (s.y*B*(-omega1_sq - omega3_sq)
             +  s.z*(B*omega_23 - A*omega.x)
             +  s.x*(A*omega.z + B*omega_12));
    s_temp.z = s.z
             + (s.z*B*(-omega1_sq - omega2_sq)
             +  s.x*(B*omega_13 - A*omega.y)
             +  s.y*(A*omega.x + B*omega_23));

    return s_temp;
}

__device__ void inner_spin_d(struct varGPU *var_ptr_d,
                             struct atom_struct *atom_ptr,
                             struct cell_struct *first_cell_ptr_d,
                             double *Jij_ptr_d)
{
    struct atom_struct *work_ptr;

    struct cell_struct *ccell_ptr;
    struct cell_struct *wcell_ptr;

    ccell_ptr = first_cell_ptr_d + atom_ptr->new_cell_index;

    for (int i = 0; i <= 26; ++i){
        if (i == 26)
            wcell_ptr = ccell_ptr;
        else
            wcell_ptr = first_cell_ptr_d + (ccell_ptr->neigh_cell[i]);

        work_ptr = wcell_ptr->head_ptr;
        while (work_ptr != NULL){

            vector rij = vec_sub_d(atom_ptr->r, work_ptr->r);

            //find image of j closest to i
            find_image_d(rij, var_ptr_d);

            double rsq = vec_sq_d(rij);

            if (rsq < var_ptr_d->rcut_mag_sq && atom_ptr != work_ptr){
                double rij0 = sqrt(rsq);
                double Jij_rij = Jij_d(rij0, Jij_ptr_d, var_ptr_d);
                atom_ptr->Heff_H = vec_add_d(atom_ptr->Heff_H, vec_times_d(Jij_rij, work_ptr->s));
            }
            work_ptr = work_ptr->next_atom_ptr;
        }
    }
}

#ifdef SLDHL
__device__ void inner_sum_Jij_sj_d(struct varGPU *var_ptr_d,
                                   struct atom_struct *atom_ptr,
                                   struct cell_struct *first_cell_ptr_d,
                                   double *Jij_ptr_d)
{
    struct atom_struct *work_ptr;

    struct cell_struct *ccell_ptr;
    struct cell_struct *wcell_ptr;

    ccell_ptr = first_cell_ptr_d + atom_ptr->new_cell_index;

    for (int i = 0; i <= 26; ++i){
        if (i == 26)
            wcell_ptr = ccell_ptr;
        else
            wcell_ptr = first_cell_ptr_d + (ccell_ptr->neigh_cell[i]);

        work_ptr = wcell_ptr->head_ptr;
        while (work_ptr != NULL){

            vector rij = vec_sub_d(atom_ptr->r, work_ptr->r);

            //find image of j closest to i
            find_image_d(rij, var_ptr_d);

            double rsq = vec_sq_d(rij);

            if (rsq < var_ptr_d->rcut_mag_sq && atom_ptr != work_ptr){
                double rij0 = sqrt(rsq);
                double Jij_rij = Jij_d(rij0, Jij_ptr_d, var_ptr_d);
                double sj = vec_length_d(work_ptr->s);
                atom_ptr->sum_Jij_sj += Jij_rij*sj;
            }
            work_ptr = work_ptr->next_atom_ptr;
        }
    }
}
#endif

#endif
