/**********************************************************
*
* Choose control options here.
*
***********************************************************/

/* built lattice */
#define readconf        //read in the configuration
//#define readvsim        //read in V_sim compatible files as configuration file
//#define bcc100
//#define bcc111          //x-> <111> y-> <-211> z-> <0-11>
//#define fcc100
//#define hcp0001

//#define runstep           //runs with a fixed number of steps, instead of a total time
#define changestep        //timestep varies according to max. displacement.

#define inittime           //initial total_time = start_time, default start_time = 0e0;
//#define initspin          //initial spin; need to edit initial_spin.spp
//#define initmomentum      //initial momentum; need to edit initial_momentum.cpp

/* Langevin thermostat */
#define spinlang          //To switch on the Langevin thermostat of spin
#define lattlang          //To switch on the Langevin thermostat of lattice

#define eltemp          //consider electron as a subsystem, with heterogeneous temperature, such that energy conserves for all subsystems being considered.
//#define readTe          //read in the link cell temperature file for the electron temperature of link cells
#define renormalizeEnergy //renormalize the total energy after every time step
#define localcolmot     //consider local collective motion; it is a sub-option to eltemp. It works only when eltemp is switched on.

/* rescale box lengthes according to its own stresses or pressure */
//#define STRESS          //need to edit stress.cpp or stress_GPU.cu
//#define PRESSURE        //need to edit pressure.cpp or pressure_GPU.cu

//#define localvol        //local stress is defined by dividing by local volume, which is calculated by using a quick method, not accurate.
//#define magmom          //use magnetic moment as input and output, instead of atomic spin

//#define extforce        //superimpose external forces onto the system
//#define extfield        //introduce external magnetic field

#define writevsim        //output V_sim compatible files

//#define quantumnoise     //quantum thermostat
