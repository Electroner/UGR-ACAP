#!/bin/bash
#clear

#Comando para ejecutar:
#sbatch -Aacap -p acap -N 1 -c 1 --hint=nomultithread --exclusive --wrap "./comexecatc.sh pi"

#check if the file exists
if [ -f $1.c ]; then
    #run the file
    if [ $# -eq 1 ]; then
        srun -Aacap -p acap gcc $1.c -o $1
        for ((P=0;P<1000000001;P=P+100000000))
        do
            srun -Aacap -p acap ./$1 $(( $P )) >> pi_secuencial_atc.dat
        done
    else
        srun -Aacap -p acap mpicc $1.c -o $1
        for ((P=0;P<1000000001;P=P+100000000))
        do
            srun -Aacap -p acap mpiexec -np $2 --oversubscribe ./$1 $(( $P )) >> pi_parallel_atc.dat
        done
    fi
fi

#parameters: $1 = file name, $2 = number of processes