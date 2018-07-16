# delete unused files
rm spilady
#rm spilady.exe

# gcc compiler
#g++ -fopenmp -o spilady -DCPU -DOMP -DSLDHL *.cpp

# -DMD -DSDH #-DSDHL #-DSLDH #-DSLDHL #-DSLDNC

#icc compiler
#module load icc
#module load ifort
#module load openmpi-intel
icc -O3 -ipo -o spilady -DCPU -DOMP -DSLDHL *.cpp -qopenmp

# nvcc compiler, using nvidia GPU
#module load cuda/5.5
#nvcc -g -arch=sm_35 -rdc=true -o spilady -DGPU -DSLDHL *.cpp *.cu

#-arch=sm_20 #-use_fast_math #-deviceemu #-fopenmp #-O0 # -g
