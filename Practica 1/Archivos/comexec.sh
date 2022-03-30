#!/bin/bash
#clear

#check if the file exists
if [ -f $1.c ]; then
    #run the file
    if [ $# -eq 1 ]; then
	gcc $1.c -o $1
        for ((P=0;P<1000000001;P=P+10000000))
        do
            ./pi_sec $(( $P )) >> pi_secuencial.dat
        done
    else
	mpicc $1.c -o $1
        for ((P=0;P<1000000001;P=P+10000000))
        do
            mpiexec -np $2 --oversubscribe ./$1 $(( $P )) >> pi_parallel.dat
        done
    fi
fi

#parameters: $1 = file name, $2 = number of processes