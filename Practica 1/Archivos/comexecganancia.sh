#!/bin/bash
#clear

#Para conseguir una media se recomienda ejectuar varias veces.

#check if the file exists
if [ -f $1.c ]; then
    #run the file
    if [ $# -eq 1 ]; then
	    gcc $1.c -o $1
        ./pi_sec 1000000000 >> pi_secuencial_ganancia.dat
    else
	    mpicc $1.c -o $1
        mpiexec -np $2 --oversubscribe ./$1 1000000000 >> pi_paralelo_ganancia.dat
    fi
fi

#parameters: $1 = file name, $2 = number of processes