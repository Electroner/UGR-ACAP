#!/bin/bash
#clear

#check if the file exists
if [ -f $1.c ]; then
    #run the file
    if [ $# -eq 1 ]; then
	gcc $1.c -o $1 -lm
        for ((P=0;P<1000000001;P=P+10000000))
        do
            ./$1 $(( $P )) >> Ejer6_sec.dat
        done
    else
	mpicc $1.c -o $1 -lm
        for ((P=2;P<1000000001;P=P+10000000))
        do
            mpirun -np $2 --oversubscribe ./$1 $(( $P )) >> Ejer6_parallel.dat
        done
    fi
fi

#parameters: $1 = file name, $2 = number of processes